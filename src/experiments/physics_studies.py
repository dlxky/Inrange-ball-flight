"""Physics studies: how far pure physics gets, and what shape the aerodynamics should take.

    python -m experiments.physics_studies inversion      checkpoint-only inversion: spin prior, wind, oracle spin
    python -m experiments.physics_studies aerodynamics   per-shot drag and lift multipliers from the full flight

Both read outputs/calibration.json from `python -m flight.calibration`. Run from src/ or with PYTHONPATH=src.
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from data.dataset import ROOT, load
from data.features import feature_frame
from flight.inversion import RECONCILE_PARAMS, fly, invert, load_calibration, reconcile, scaled_aero, wind_for
from flight.simulator import RADIUS, RPM_TO_RAD, simulate
from modelling.evaluation import errors, mean_scales, summarise
from modelling.pipeline import lgbm

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)


def inversion(train, cal):
    x = feature_frame(train)
    scales = mean_scales(train)
    shots = pd.read_csv(ROOT / "outputs" / "calibration_shots.csv")
    spin_oof = np.zeros(len(train))
    for fit, val in KFold(5, shuffle=True, random_state=42).split(x):
        spin_oof[val] = lgbm().fit(x.iloc[fit], train.launch_spin_rate.iloc[fit]).predict(x.iloc[val])

    n = len(train)
    nuisance_mean = [0.0] + [shots[k].mean() for k in ("dd", "dl", "dh")]
    nuisance_sd = [0.3] + [shots[k].std() for k in ("dd", "dl", "dh")]
    wind = wind_for(train, cal["wind"])
    loose = (0.015, 0.15, 0.3)
    variants = {
        "ML spin prior, tight checkpoints": (spin_oof, 0.9, wind, (0.01, 0.1, 0.1)),
        "ML spin prior": (spin_oof, 0.9, wind, loose),
        "ML spin prior, no wind": (spin_oof, 0.9, np.zeros_like(wind), loose),
        "ML spin fixed": (spin_oof, 0.01, wind, loose),
        "true spin fixed (oracle)": (train.launch_spin_rate.to_numpy(), 0.01, wind, loose),
    }
    rows = {}
    for name, (spin, spin_sd, w, sigma) in variants.items():
        prior_mean = np.column_stack([spin / 1000] + [np.full(n, m) for m in nuisance_mean])
        prior_sd = np.column_stack([np.full(n, spin_sd)] + [np.full(n, s) for s in nuisance_sd])
        theta, _ = invert(train, cal["aero"], w, prior_mean, prior_sd, sigma_cp=sigma)
        pred = fly(train, theta, cal["aero"], w)
        rows[name] = summarise(errors(train, pred), scales)
        rows[name]["landing_d_bias"] = (pred.landing_d - train.landing_d).mean()
    print(pd.DataFrame(rows).T.round(3))


def aerodynamics(train, cal):
    spin = train.launch_spin_rate.to_numpy()
    wind = wind_for(train, cal["wind"])
    params = reconcile(train, train, spin, cal["aero"], wind, prior_sd=np.array([1.0, 1.0, 0.5, 1.0, 0.5, 0.5]))
    log_cd, log_cl, tilt, dd, dl, dh = params[RECONCILE_PARAMS].to_numpy().T
    o = simulate(train[["vd", "vl", "vh"]].to_numpy(), spin, tilt, scaled_aero(cal["aero"], log_cd, log_cl), wind=wind)
    for k, shift in [("apex_d", dd), ("apex_h", dh), ("landing_d", dd), ("landing_l", dl)]:
        print(f"post-fit rms {k:10s} {np.sqrt(np.nanmean((o[k] + shift - train[k]) ** 2)):.3f} m")

    base = cal["aero"]
    d = params.assign(cd_mult=np.exp(log_cd), cl_mult=np.exp(log_cl), S0=RADIUS * spin * RPM_TO_RAD / train.speed)
    d["cd_eff"] = d.cd_mult * (base["cd0"] + base["cd1"] * d.S0)
    d["cl_eff"] = d.cl_mult * base["cl_max"] * d.S0 / (d.S0 + base["s_half"])
    for col in ["launch_spin_rate", "speed", "vla"]:
        print(d.groupby(pd.qcut(train[col], 6), observed=True)[["cd_mult", "cl_mult", "cd_eff", "cl_eff", "dd", "dh"]]
              .median().round(3))
    d.assign(track_id=train.track_id.values).to_csv(ROOT / "outputs" / "aero_diag.csv", index=False)


if __name__ == "__main__":
    studies = {"inversion": inversion, "aerodynamics": aerodynamics}
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", choices=studies)
    train, _ = load()
    studies[parser.parse_args().study](train, load_calibration())
