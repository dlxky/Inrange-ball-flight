"""Final model: calibrated flight physics plus a Gaussian-process correction layer.

Everything learned from labels (aerodynamic calibration, block wind, spin model, GP layer) is refitted
inside each cross-validation fold, so the CV score is an honest estimate for the test set.

    python -m modelling.pipeline --cv --submit [--layer gp_feature] [--no-wind] [--groups session]

Ablations (CV only): --oracle-spin feeds the true launch-monitor spin instead of the LightGBM estimate;
--no-fingerprints drops the kinematic features from the spin model and the GP layer.
Run from src/ or with PYTHONPATH=src.
"""
import argparse
import json
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from data.dataset import LOCAL_TARGETS, ROOT, load, to_submission
from data.features import BASE_FEATURES, feature_frame
from flight.calibration import Calibration, aero_from
from flight.inversion import fly, invert, reconcile, wind_for
from modelling.evaluation import errors, mean_scales, summarise

warnings.filterwarnings("ignore")
RAW_GP_FEATURES = ["vd", "vl", "vh", "elevated", "cp2_l", "cp2_h", "cp4_t", "cp4_l", "cp4_h"]
GP_FEATURES = RAW_GP_FEATURES + ["lift_proxy", "time_excess", "side_excess"]
POSITION_TARGETS = LOCAL_TARGETS[1:]


def lgbm():
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=10,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.7, n_jobs=1, verbose=-1)


def gp_fit_predict(x_fit, y_fit, x_pred):
    scaler = StandardScaler().fit(x_fit)
    kernel = ConstantKernel() * RBF(np.ones(x_fit.shape[1]), (1e-2, 1e3)) + WhiteKernel(1e-2)
    gp = GaussianProcessRegressor(kernel, normalize_y=True).fit(scaler.transform(x_fit), y_fit)
    return gp.predict(scaler.transform(x_pred))


def calibrate(df, max_nfev=40):
    cal = Calibration(df)
    g, _, dd, dl, dh, wind = cal.unpack(cal.fit(max_nfev=max_nfev, verbose=0).x)
    return dict(aero=aero_from(g), wind={int(b): [float(w[0]), float(w[1])] for b, w in zip(cal.blocks, wind.T)},
                nuisance={k: [float(v.mean()), float(v.std())] for k, v in (("dd", dd), ("dl", dl), ("dh", dh))})


def physics_predictions(df, spin_rpm, cal, use_wind):
    """Invert the checkpoints with spin fixed at the given estimate, then fly each shot to landing."""
    n = len(df)
    wind = wind_for(df, cal["wind"]) if use_wind else np.zeros((n, 3))
    nuis = cal["nuisance"]
    prior_mean = np.column_stack([spin_rpm / 1000, np.zeros(n)] + [np.full(n, nuis[k][0]) for k in ("dd", "dl", "dh")])
    prior_sd = np.tile([0.01, 0.3] + [nuis[k][1] for k in ("dd", "dl", "dh")], (n, 1))
    theta, _ = invert(df, cal["aero"], wind, prior_mean, prior_sd, sigma_cp=(0.015, 0.15, 0.3))
    return fly(df, theta, cal["aero"], wind), wind


def fit_predict(fit_df, pred_df, layer="gp_feature", use_wind=True, oracle_spin=False, fingerprints=True):
    started = time.time()
    gp_features = GP_FEATURES if fingerprints else RAW_GP_FEATURES
    with threadpool_limits(1):
        cal = calibrate(fit_df)
        if fingerprints:
            x_fit, x_pred = feature_frame(fit_df), feature_frame(pred_df)
        else:
            x_fit, x_pred = fit_df[BASE_FEATURES], pred_df[BASE_FEATURES]
        if oracle_spin:  # ablation: as if the launch spin were measured
            spin_fit, spin_pred = fit_df.launch_spin_rate.to_numpy(), pred_df.launch_spin_rate.to_numpy()
        else:
            spin_fit = np.zeros(len(fit_df))
            for a, b in KFold(5, shuffle=True, random_state=0).split(x_fit):
                spin_fit[b] = lgbm().fit(x_fit.iloc[a], fit_df.launch_spin_rate.iloc[a]).predict(x_fit.iloc[b])
            spin_pred = lgbm().fit(x_fit, fit_df.launch_spin_rate).predict(x_pred)
        phys_fit, _ = physics_predictions(fit_df, spin_fit, cal, use_wind)
        phys_pred, wind_pred = physics_predictions(pred_df, spin_pred, cal, use_wind)

        pred = pd.DataFrame({"launch_spin_rate": spin_pred}, index=pred_df.index)
        for target in POSITION_TARGETS:
            y, pf, pp = fit_df[target], phys_fit[target], phys_pred[target]
            if layer == "gp_residual":
                pred[target] = pp.to_numpy() + gp_fit_predict(x_fit[gp_features], y - pf, x_pred[gp_features])
            elif layer == "gp_feature":
                pred[target] = gp_fit_predict(x_fit[gp_features].assign(phys=pf), y,
                                              x_pred[gp_features].assign(phys=pp))
            else:  # physics only
                pred[target] = pp
        pred["landing_h"] = 0.0
    print(f"fold of {len(fit_df)} → {len(pred_df)} shots done in {time.time() - started:.0f}s", flush=True)
    return pred, dict(cal=cal, physics=phys_pred, wind=wind_pred)


def cross_validate(train, layer, use_wind, k=5, groups=None, oracle_spin=False, fingerprints=True):
    splitter = GroupKFold(k) if groups is not None else KFold(k, shuffle=True, random_state=42)
    folds = list(splitter.split(train, groups=groups))
    parts = Parallel(n_jobs=k)(delayed(fit_predict)(train.iloc[a], train.iloc[b], layer, use_wind, oracle_spin,
                                                    fingerprints) for a, b in folds)
    return pd.concat([p for p, _ in parts]).loc[train.index], pd.concat([e["physics"] for _, e in parts]).loc[train.index]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--layer", default="gp_feature", choices=["gp_feature", "gp_residual", "physics"])
    parser.add_argument("--no-wind", action="store_true")
    parser.add_argument("--groups", default="none", choices=["none", "session"])
    parser.add_argument("--oracle-spin", action="store_true")
    parser.add_argument("--no-fingerprints", action="store_true")
    args = parser.parse_args()
    if args.submit and args.oracle_spin:
        parser.error("--oracle-spin needs the true spin, which the test set does not have")
    use_wind, fingerprints = not args.no_wind, not args.no_fingerprints
    tag = (f"{args.layer}{'' if use_wind else '_nowind'}{'_oraclespin' if args.oracle_spin else ''}"
           f"{'' if fingerprints else '_nofingerprints'}{'_bysession' if args.groups == 'session' else ''}")

    train, test = load()
    scales = mean_scales(train)
    if args.cv:
        groups = train.session if args.groups == "session" else None
        oof, phys = cross_validate(train, args.layer, use_wind, groups=groups, oracle_spin=args.oracle_spin,
                                   fingerprints=fingerprints)
        report = pd.DataFrame({"model": summarise(errors(train, oof), scales),
                               "physics_only": summarise(errors(train, phys.assign(launch_spin_rate=oof.launch_spin_rate)), scales)})
        print(report.round(3).T)
        oof.assign(track_id=train.track_id.values).to_csv(ROOT / "outputs" / f"pipeline_oof_{tag}.csv", index=False)
    if args.submit:
        pred, extra = fit_predict(train, test, args.layer, use_wind, fingerprints=fingerprints)
        (ROOT / "submissions").mkdir(exist_ok=True)
        to_submission(test, pred).to_csv(ROOT / "submissions" / "submission.csv", index=False)
        (ROOT / "outputs" / "calibration_final.json").write_text(json.dumps(extra["cal"], indent=2))
        params = reconcile(test, pred, pred.launch_spin_rate.to_numpy(), extra["cal"]["aero"], extra["wind"])
        params.assign(track_id=test.track_id.values).to_csv(ROOT / "outputs" / "test_trajectory_params.csv", index=False)
        pred.assign(track_id=test.track_id.values).to_csv(ROOT / "outputs" / "test_predictions_local.csv", index=False)
        print("submission written:", len(pred), "rows")
