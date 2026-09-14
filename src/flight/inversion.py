"""Per-shot inference from the checkpoints, and reconciling flights with a predicted apex and landing.

`invert` is the netted-range problem: a MAP fit of spin, spin-axis tilt and start offsets to the four
checkpoint crossings under the calibrated flight model. `reconcile` fits per-shot drag and lift
multipliers (plus tilt and offsets) so the simulated flight passes through the checkpoints and a given
apex and landing, turning point predictions into physically consistent full trajectories. Both are
solved with `batched_lm`, which runs hundreds of small least-squares problems as one.
"""
import json

import numpy as np
import pandas as pd

from data.dataset import POINTS, ROOT
from flight.simulator import simulate

PARAMS = ["spin", "tilt", "dd", "dl", "dh"]  # spin in thousands of rpm
RECONCILE_PARAMS = ["log_cd", "log_cl", "tilt", "dd", "dl", "dh"]
RECONCILE_PRIOR_SD = np.array([0.5, 0.5, 0.5, 1.0, 0.5, 0.5])
EVENTS = ["apex_t", "apex_d", "apex_l", "apex_h", "landing_t", "landing_d", "landing_l"]


def batched_lm(resid, theta0, step, lower, upper, n_iter=12):
    """Levenberg-Marquardt for many independent problems: minimise sum(resid(theta, idx)**2) per row.

    resid(theta, idx) evaluates problems `idx` at parameters theta (len(idx), P) and returns
    (len(idx), M) residuals; all forward-difference Jacobians are taken in one batched call.
    """
    n, p = theta0.shape
    rows = np.arange(n)
    theta = np.clip(np.array(theta0, dtype=float), lower, upper)
    r = resid(theta, rows)
    cost = (r**2).sum(1)
    lam = np.full(n, 1e-2)
    for _ in range(n_iter):
        pert = (theta[:, None, :] + np.eye(p)[None] * step).reshape(-1, p)
        rp = resid(pert, np.repeat(rows, p)).reshape(n, p, -1)
        J = ((rp - r[:, None, :]) / step[None, :, None]).transpose(0, 2, 1)  # (N, M, P)
        JtJ = J.transpose(0, 2, 1) @ J
        grad = J.transpose(0, 2, 1) @ r[..., None]
        damp = lam[:, None, None] * np.eye(p)[None] * np.diagonal(JtJ, axis1=1, axis2=2)[:, None, :]
        trial = np.clip(theta - np.linalg.solve(JtJ + damp + 1e-9 * np.eye(p), grad)[..., 0], lower, upper)
        r_new = resid(trial, rows)
        cost_new = (r_new**2).sum(1)
        ok = cost_new < cost
        theta[ok], r[ok], cost[ok] = trial[ok], r_new[ok], cost_new[ok]
        lam = np.where(ok, lam * 0.3, lam * 10)
    return theta, cost


def load_calibration():
    cal = json.loads((ROOT / "outputs" / "calibration.json").read_text())
    cal["wind"] = {int(k): v for k, v in cal["wind"].items()}
    return cal


def wind_for(df, wind_by_block):
    """Per-shot wind from its hitting block; unseen blocks get the mean, which carries the calibration's
    constant offset (a zero default would bias the physics for a brand-new session)."""
    default = np.mean(list(wind_by_block.values()), axis=0) if wind_by_block else [0.0, 0.0]
    w = np.array([wind_by_block.get(int(b), default) for b in df.block])
    return np.column_stack([w, np.zeros(len(df))])


def arrays(df):
    return dict(v0=df[["vd", "vl", "vh"]].to_numpy(),
                cp_d=df[[f"{p}_d" for p in POINTS]].to_numpy(),
                obs=np.concatenate([df[[f"{p}_{c}" for p in POINTS]].to_numpy() for c in "tlh"], axis=1))


def scaled_aero(aero, log_cd=0.0, log_cl=0.0):
    return dict(aero, cd0=aero["cd0"] * np.exp(log_cd), cd1=aero["cd1"] * np.exp(log_cd),
                cl_max=aero["cl_max"] * np.exp(log_cl))


def invert(df, aero, wind, prior_mean, prior_sd, sigma_cp=(0.01, 0.1, 0.1), n_iter=12):
    """MAP estimate of PARAMS per shot. prior_mean, prior_sd: (N, 5) in PARAMS order."""
    a = arrays(df)
    obs_sd = np.repeat(sigma_cp, 4)

    def resid(theta, idx):
        spin, tilt, dd, dl, dh = theta.T
        o = simulate(a["v0"][idx], spin * 1000, tilt, aero, wind=wind[idx], cp_d=a["cp_d"][idx] - dd[:, None],
                     stop="cps", t_max=5.0)
        pred = np.concatenate([o["cp_t"], o["cp_l"] + dl[:, None], o["cp_h"] + dh[:, None]], axis=1)
        r = np.concatenate([(pred - a["obs"][idx]) / obs_sd, (theta - prior_mean[idx]) / prior_sd[idx]], axis=1)
        return np.nan_to_num(r, nan=50.0)

    theta, cost = batched_lm(resid, prior_mean, step=np.array([0.05, 0.005, 0.01, 0.01, 0.01]),
                             lower=np.array([0.3, -1.2, -3.0, -2.0, -1.0]),
                             upper=np.array([15.0, 1.2, 3.0, 2.0, 1.0]), n_iter=n_iter)
    return pd.DataFrame(theta, columns=PARAMS, index=df.index), cost


def fly(df, theta, aero, wind, keep_path=False):
    """Full flight from the fitted parameters; returns local-frame predictions (and paths if asked)."""
    spin, tilt, dd, dl, dh = theta[PARAMS].to_numpy().T
    o = simulate(df[["vd", "vl", "vh"]].to_numpy(), spin * 1000, tilt, aero, wind=wind, keep_path=keep_path)
    pred = pd.DataFrame(index=df.index)
    pred["launch_spin_rate"] = spin * 1000
    pred["apex_t"] = o["apex_t"]
    pred["apex_d"], pred["apex_l"], pred["apex_h"] = o["apex_d"] + dd, o["apex_l"] + dl, o["apex_h"] + dh
    pred["landing_t"] = o["landing_t"]
    pred["landing_d"], pred["landing_l"], pred["landing_h"] = o["landing_d"] + dd, o["landing_l"] + dl, 0.0
    return (pred, o) if keep_path else pred


def reconcile(df, targets, spin_rpm, aero, wind, prior_sd=RECONCILE_PRIOR_SD, n_iter=15):
    """Fit a flight through the checkpoints and the apex/landing in `targets` (EVENTS columns)."""
    a = arrays(df)
    obs = np.column_stack([a["obs"]] + [targets[k].to_numpy() for k in EVENTS])
    obs_sd = np.array([0.015] * 4 + [0.15] * 4 + [0.3] * 4 + [0.08, 2.0, 0.8, 0.4, 0.05, 1.0, 1.0])

    def resid(theta, idx):
        log_cd, log_cl, tilt, dd, dl, dh = theta.T
        o = simulate(a["v0"][idx], spin_rpm[idx], tilt, scaled_aero(aero, log_cd, log_cl), wind=wind[idx],
                     cp_d=a["cp_d"][idx] - dd[:, None])
        pred = np.column_stack([o["cp_t"], o["cp_l"] + dl[:, None], o["cp_h"] + dh[:, None],
                                o["apex_t"], o["apex_d"] + dd, o["apex_l"] + dl, o["apex_h"] + dh,
                                o["landing_t"], o["landing_d"] + dd, o["landing_l"] + dl])
        return np.nan_to_num(np.concatenate([(pred - obs[idx]) / obs_sd, theta / prior_sd], axis=1), nan=50.0)

    theta, _ = batched_lm(resid, np.zeros((len(df), 6)), step=np.array([0.01, 0.01, 0.005, 0.01, 0.01, 0.01]),
                          lower=np.array([-1.5, -1.5, -1.2, -3.0, -2.0, -1.0]),
                          upper=np.array([1.5, 1.5, 1.2, 3.0, 2.0, 1.0]), n_iter=n_iter)
    return pd.DataFrame(theta, columns=RECONCILE_PARAMS, index=df.index)
