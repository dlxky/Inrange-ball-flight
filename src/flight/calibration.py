"""Calibrate the flight model on training shots, whose launch spin is known.

Global aerodynamic coefficients are fitted jointly with per-shot nuisance parameters (spin-axis tilt
and a small start offset, since the launch position in the data is the bay's nominal spot) and a
per-block wind, against every measured point: the four checkpoints plus apex and level landing.

    python -m flight.calibration [max_nfev]    writes outputs/calibration.json and calibration_shots.csv
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from data.dataset import POINTS, ROOT, load
from flight.simulator import DEFAULT_AERO, simulate

GLOBAL0 = {"cd0": 0.20, "cd1": 0.25, "cl_max": 0.45, "s_half": 0.12, "decay": 1.0}  # decay in 1e-3/m
GLOBAL_BOUNDS = {"cd0": (0.05, 0.5), "cd1": (0.0, 1.5), "cl_max": (0.05, 1.5), "s_half": (0.01, 1.0),
                 "decay": (0.0, 10.0)}
SHOT = {"tilt": (0.5, 1.2), "dd": (1.0, 3.0), "dl": (0.5, 2.0), "dh": (0.3, 1.0)}  # (prior sd, bound)
WIND_SD, WIND_BOUND = 3.0, 15.0
SIGMA = {"cp_t": 0.01, "cp_l": 0.10, "cp_h": 0.10, "apex_t": 0.08, "apex_d": 2.0, "apex_l": 0.8,
         "apex_h": 0.4, "landing_t": 0.05, "landing_d": 1.0, "landing_l": 1.0}
EVENTS = ["apex_t", "apex_d", "apex_l", "apex_h", "landing_t", "landing_d", "landing_l"]
ROWS = 12 + len(EVENTS)


def aero_from(values):
    aero = dict(DEFAULT_AERO)
    aero.update(dict(zip(GLOBAL0, values)))
    aero["decay"] *= 1e-3
    return aero


class Calibration:
    def __init__(self, df):
        self.n = len(df)
        self.v0 = df[["vd", "vl", "vh"]].to_numpy()
        self.spin = df.launch_spin_rate.to_numpy()
        self.cp_d = df[[f"{p}_d" for p in POINTS]].to_numpy()
        self.cp = {c: df[[f"{p}_{c}" for p in POINTS]].to_numpy() for c in "tlh"}
        self.ev = {k: df[k].to_numpy() for k in EVENTS}
        self.blocks, self.bidx = np.unique(df.block.to_numpy(), return_inverse=True)
        self.nb = len(self.blocks)
        self.ng = len(GLOBAL0)

    def unpack(self, x):
        g = x[:self.ng]
        tilt, dd, dl, dh = x[self.ng:self.ng + 4 * self.n].reshape(4, self.n)
        wind = x[self.ng + 4 * self.n:].reshape(2, self.nb)
        return g, tilt, dd, dl, dh, wind

    def wind_vectors(self, wind):
        return np.stack([wind[0, self.bidx], wind[1, self.bidx], np.zeros(self.n)], axis=1)

    def run(self, x):
        g, tilt, dd, dl, dh, wind = self.unpack(x)
        o = simulate(self.v0, self.spin, tilt, aero_from(g), wind=self.wind_vectors(wind),
                     cp_d=self.cp_d - dd[:, None])
        shift = {"d": dd, "l": dl, "h": dh, "t": 0.0}
        pred = {"cp_t": o["cp_t"], "cp_l": o["cp_l"] + dl[:, None], "cp_h": o["cp_h"] + dh[:, None]}
        for k in EVENTS:
            pred[k] = o[k] + shift[k[-1]]
        return pred, o

    def residuals(self, x):
        pred, _ = self.run(x)
        shot = [(pred[f"cp_{c}"] - self.cp[c]) / SIGMA[f"cp_{c}"] for c in "tlh"]
        shot += [((pred[k] - self.ev[k]) / SIGMA[k])[:, None] for k in EVENTS]
        shot = np.nan_to_num(np.hstack(shot), nan=50.0).ravel()
        _, tilt, dd, dl, dh, wind = self.unpack(x)
        prior = [p / SHOT[name][0] for name, p in zip(SHOT, (tilt, dd, dl, dh))] + [wind.ravel() / WIND_SD]
        return np.concatenate([shot] + prior)

    def sparsity(self):
        nshot = self.n * ROWS
        ncol = self.ng + 4 * self.n + 2 * self.nb
        S = lil_matrix((nshot + ncol - self.ng, ncol), dtype=np.int8)
        for i in range(self.n):
            rows = slice(i * ROWS, (i + 1) * ROWS)
            S[rows, :self.ng] = 1
            for p in range(4):
                S[rows, self.ng + p * self.n + i] = 1
            for c in range(2):
                S[rows, self.ng + 4 * self.n + c * self.nb + self.bidx[i]] = 1
        for k in range(ncol - self.ng):
            S[nshot + k, self.ng + k] = 1
        return S

    def fit(self, max_nfev=40, verbose=2):
        x0 = np.concatenate([list(GLOBAL0.values()), np.zeros(4 * self.n + 2 * self.nb)])
        lo = [b[0] for b in GLOBAL_BOUNDS.values()] + sum([[-SHOT[k][1]] * self.n for k in SHOT], [])
        hi = [b[1] for b in GLOBAL_BOUNDS.values()] + sum([[SHOT[k][1]] * self.n for k in SHOT], [])
        lo += [-WIND_BOUND] * 2 * self.nb
        hi += [WIND_BOUND] * 2 * self.nb
        return least_squares(self.residuals, x0, jac_sparsity=self.sparsity(), bounds=(lo, hi),
                             x_scale="jac", diff_step=1e-4, max_nfev=max_nfev, verbose=verbose)


def report(cal, x, df):
    pred, o = cal.run(x)
    g, tilt, dd, dl, dh, wind = cal.unpack(x)
    print("\nglobal:", {k: round(v, 4) for k, v in zip(GLOBAL0, g)})
    for c in "tlh":
        print(f"cp_{c} rms per checkpoint:", np.sqrt(np.nanmean((pred[f'cp_{c}'] - cal.cp[c]) ** 2, 0)).round(3))
    for k in EVENTS:
        r = pred[k] - cal.ev[k]
        print(f"{k:10s} bias {np.nanmean(r):7.3f}  rms {np.sqrt(np.nanmean(r ** 2)):6.3f}")
    land = np.hypot(pred["landing_d"] - cal.ev["landing_d"], pred["landing_l"] - cal.ev["landing_l"])
    print("landing euclid mean", land.mean().round(3))
    for name, v in zip(SHOT, (tilt, dd, dl, dh)):
        print(f"{name:5s} mean {v.mean():7.3f} sd {v.std():6.3f}")
    print("wind (d, l) per block:\n", pd.DataFrame(wind.T, index=cal.blocks, columns=["wd", "wl"]).round(2).T)
    resid = pd.DataFrame({"landing_d": pred["landing_d"] - cal.ev["landing_d"],
                          "apex_h": pred["apex_h"] - cal.ev["apex_h"], "landing_t": pred["landing_t"] - cal.ev["landing_t"]})
    for col, bins in [("launch_spin_rate", 4), ("speed", 4), ("vla", 4)]:
        print(resid.groupby(pd.qcut(df[col], bins), observed=True).mean().round(3))


if __name__ == "__main__":
    train, _ = load()
    cal = Calibration(train)
    result = cal.fit(max_nfev=int(sys.argv[1]) if len(sys.argv) > 1 else 40)
    report(cal, result.x, train)
    g, tilt, dd, dl, dh, wind = cal.unpack(result.x)
    out = {"aero": aero_from(g), "wind": {int(b): [float(w[0]), float(w[1])] for b, w in zip(cal.blocks, wind.T)}}
    (ROOT / "outputs" / "calibration.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame({"track_id": train.track_id, "tilt": tilt, "dd": dd, "dl": dl, "dh": dh}).to_csv(
        ROOT / "outputs" / "calibration_shots.csv", index=False)
