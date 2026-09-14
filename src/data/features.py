"""Physically motivated features from the launch and the four checkpoint crossings."""
import numpy as np
import pandas as pd

from data.dataset import POINTS
from flight.simulator import AREA, G, MASS

K_NOMINAL = 1.19 * AREA / (2 * MASS)


def kinematic_features(df):
    """Physically motivated summaries of the visible 60 m of flight.

    Each axis is fitted with r(t) = v0 t + a t^2/2 + j t^3/6 through the four checkpoints, anchored
    at the radar launch velocity v0. The implied aerodynamic acceleration (total minus gravity) is
    split into drag along the velocity and lift across it (vertical-plane and sideways parts),
    normalised by the dynamic pressure so they read as effective drag and lift coefficients.
    """
    t = df[[f"{p}_t" for p in POINTS]].to_numpy()
    r = np.stack([df[[f"{p}_{c}" for p in POINTS]].to_numpy() for c in "dlh"], axis=-1)
    v0 = df[["vd", "vl", "vh"]].to_numpy()
    basis = np.stack([t**2 / 2, t**3 / 6], axis=-1)
    lhs = np.einsum("nki,nkj->nij", basis, basis)
    rhs = np.einsum("nki,nkc->nci", basis, r - t[..., None] * v0[:, None, :])
    coef = np.linalg.solve(lhs[:, None], rhs[..., None])[..., 0]  # (N, axis, [a, j])
    resid = r - t[..., None] * v0[:, None, :] - np.einsum("nki,nci->nkc", basis, coef)

    feats = {"fit_rms": np.sqrt((resid**2).mean(axis=(1, 2)))}
    for label, tt in [("mid", t[:, 1]), ("end", t[:, 3])]:
        vel = v0 + coef[..., 0] * tt[:, None] + coef[..., 1] * tt[:, None] ** 2 / 2
        aero = coef[..., 0] + coef[..., 1] * tt[:, None]
        aero[:, 2] += G
        speed = np.linalg.norm(vel, axis=1)
        vhat = vel / speed[:, None]
        q = K_NOMINAL * speed**2
        a_par = (aero * vhat).sum(1)
        perp = aero - a_par[:, None] * vhat
        up = np.array([0.0, 0.0, 1.0]) - vhat[:, 2:3] * vhat
        up /= np.linalg.norm(up, axis=1, keepdims=True)
        side = np.cross(up, vhat)  # + = left
        feats[f"cd_{label}"] = -a_par / q
        feats[f"clu_{label}"] = (perp * up).sum(1) / q
        feats[f"cls_{label}"] = (perp * side).sum(1) / q
        feats[f"v_{label}"] = speed
        feats[f"vh_{label}"] = vel[:, 2]
        feats[f"vl_{label}"] = vel[:, 1]
    t4 = t[:, 3]
    feats["lift_proxy"] = r[:, 3, 2] - (v0[:, 2] * t4 - G * t4**2 / 2)
    feats["time_excess"] = t4 - r[:, 3, 0] / v0[:, 0]
    feats["side_excess"] = r[:, 3, 1] - v0[:, 1] * t4
    return pd.DataFrame(feats, index=df.index)


BASE_FEATURES = ["vd", "vl", "vh", "speed", "vla", "hla", "elevated", "cp1_d"] + [
    f"{p}_{c}" for p in POINTS for c in "tlh"]


def feature_frame(df):
    return pd.concat([df[BASE_FEATURES], kinematic_features(df)], axis=1)
