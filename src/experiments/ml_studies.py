"""Machine-learning studies behind the final model choice, and the analysis of its errors.

    python -m experiments.ml_studies explore         kinematic spin signal, block effects, same-club runs
    python -m experiments.ml_studies baselines       ridge, extra trees, LightGBM and GP; random and leave-session-out CV
    python -m experiments.ml_studies neighbours      does the neighbouring training shot in time help?
    python -m experiments.ml_studies hybrid          physics prediction as GP feature vs residual base vs LightGBM feature
    python -m experiments.ml_studies error_analysis  what, why and when of the final model's out-of-fold errors

`hybrid` reads outputs/calibration.json from `python -m flight.calibration`; `error_analysis` also needs
outputs/aero_diag.csv and the pipeline's out-of-fold predictions. Run from src/ or with PYTHONPATH=src.
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kruskal, spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

from data.dataset import LOCAL_TARGETS, ROOT, load
from data.features import BASE_FEATURES, feature_frame, kinematic_features
from flight.inversion import fly, invert, load_calibration, wind_for
from modelling.evaluation import errors, mean_scales, summarise
from modelling.pipeline import GP_FEATURES, POSITION_TARGETS, gp_fit_predict, lgbm

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
RANDOM_FOLDS = KFold(5, shuffle=True, random_state=42)
ERROR_FACTORS = {
    "spin_error": "spin misestimate",
    "aero_anomaly": "unusual drag or lift for this ball",
    "curvature": "curvature (spin-axis tilt)",
    "carry_beyond_net": "carry beyond the net",
    "vla": "launch angle",
    "novelty": "distance to similar training shots",
    "upper_deck": "upper-deck bay",
}


def explore(train):
    kin = kinematic_features(train)
    tr = pd.concat([train, kin], axis=1)
    print("blocks:", train.block.nunique(), "| training shots per block:", train.block.value_counts().sort_index().tolist())
    print("\ncorrelation with launch spin:")
    print(tr[list(kin.columns) + ["speed", "vla", "cp4_t", "cp4_h"]].corrwith(tr.launch_spin_rate).round(3).to_string())

    x = pd.concat([train[BASE_FEATURES], kin], axis=1)
    print("\ntarget            OOF MAE   corr(residual, leave-one-out block mean residual)")
    for target in ["landing_d", "landing_l", "apex_h", "apex_d", "landing_t", "launch_spin_rate"]:
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
        resid = tr[target] - cross_val_predict(model, x, tr[target], cv=RANDOM_FOLDS)
        g = resid.groupby(tr.block)
        loo = (g.transform("sum") - resid) / (g.transform("count") - 1)
        ok = g.transform("count") > 1
        print(f"{target:16s} {resid.abs().mean():8.3f}   {np.corrcoef(resid[ok], loo[ok])[0, 1]:.3f}")

    s = tr.sort_values("launch_time")
    same = s.block.eq(s.block.shift())
    for col in ["launch_spin_rate", "speed", "vla", "landing_l"]:
        print(f"lag-1 corr {col:18s} {np.corrcoef(s[col][same], s[col].shift()[same])[0, 1]:.3f}")


def _oof(train, x, folds, fit_predict):
    pred = pd.DataFrame(index=train.index, columns=LOCAL_TARGETS, dtype=float)
    for fit, val in folds:
        for target in LOCAL_TARGETS:
            pred.iloc[val, pred.columns.get_loc(target)] = fit_predict(x.iloc[fit], train[target].iloc[fit], x.iloc[val])
    pred["landing_h"] = 0.0
    return pred


def _sklearn(make):
    return lambda x_fit, y_fit, x_pred: make().fit(x_fit, y_fit).predict(x_pred)


def baselines(train):
    x = feature_frame(train)
    scales = mean_scales(train)
    models = {
        "ridge_poly2": (_sklearn(lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2), StandardScaler(),
                                                       RidgeCV(alphas=np.logspace(-1, 4, 16)))), None),
        "extratrees": (_sklearn(lambda: ExtraTreesRegressor(500, min_samples_leaf=2, max_features=0.5, n_jobs=-1,
                                                            random_state=0)), None),
        "lightgbm": (_sklearn(lgbm), None),
        "gp_ard": (gp_fit_predict, GP_FEATURES),
    }
    splits = {"kfold": list(RANDOM_FOLDS.split(x)), "session": list(GroupKFold(5).split(x, groups=train.session))}
    rows = {}
    for name, (fit_predict, cols) in models.items():
        for cv_name, folds in splits.items():
            rows[(name, cv_name)] = summarise(errors(train, _oof(train, x[cols] if cols else x, folds, fit_predict)), scales)
            print(name, cv_name, rows[(name, cv_name)].round(3).to_dict(), flush=True)
    print(pd.DataFrame(rows).T.round(3))


def _neighbour_features(query, pool):
    """Closest-in-time pool shots before and after each query shot within the same hitting block."""
    q = query[["launch_time", "block", "speed", "vla"]].reset_index().sort_values("launch_time")
    pl = pool[["launch_time", "block", "speed", "vla", "launch_spin_rate", "curve", "carry_ratio"]]
    pl = pl.assign(t_nb=pl.launch_time).sort_values("launch_time")
    out = pd.DataFrame(index=query.index)
    for side, direction in [("prev", "backward"), ("next", "forward")]:
        m = pd.merge_asof(q, pl, on="launch_time", by="block", direction=direction,
                          allow_exact_matches=False, suffixes=("", "_nb")).set_index("index")
        out[f"{side}_dt"] = (m.launch_time - m.t_nb).abs()
        out[f"{side}_spin"] = m.launch_spin_rate
        out[f"{side}_dspeed"] = m.speed - m.speed_nb
        out[f"{side}_dvla"] = m.vla - m.vla_nb
        out[f"{side}_curve"] = m.curve
        out[f"{side}_carry"] = m.carry_ratio
    return out


def neighbours(train):
    train = train.assign(curve=train.landing_l - train.cp4_l * train.landing_d / train.cp4_d,
                         carry_ratio=train.landing_d / train.cp4_t)
    x = feature_frame(train)
    scales = mean_scales(train)
    rows = {}
    for use in (False, True):
        pred = pd.DataFrame(index=train.index, columns=LOCAL_TARGETS, dtype=float)
        for fit, val in RANDOM_FOLDS.split(x):
            f, v = train.iloc[fit], train.iloc[val]
            xf, xv = x.iloc[fit], x.iloc[val]
            if use:
                xf = pd.concat([xf, _neighbour_features(f, f)], axis=1)
                xv = pd.concat([xv, _neighbour_features(v, f)], axis=1)
            for target in LOCAL_TARGETS:
                pred.loc[v.index, target] = lgbm().fit(xf, f[target]).predict(xv)
        pred["landing_h"] = 0.0
        rows["with neighbours" if use else "plain"] = summarise(errors(train, pred), scales)
    print(pd.DataFrame(rows).T.round(3))


def _hybrid_task(train, x, folds, physics, kind, variant, target, k):
    fit, val = folds[k]
    y = train[target]
    phys = physics[variant][target] if variant else None
    with threadpool_limits(1):
        if kind == "gp_plain":
            pred = gp_fit_predict(x[GP_FEATURES].iloc[fit], y.iloc[fit], x[GP_FEATURES].iloc[val])
        elif kind == "gp_feature":
            cols = x[GP_FEATURES].assign(phys=phys)
            pred = gp_fit_predict(cols.iloc[fit], y.iloc[fit], cols.iloc[val])
        elif kind == "gp_residual":
            pred = phys.iloc[val].to_numpy() + gp_fit_predict(x[GP_FEATURES].iloc[fit], (y - phys).iloc[fit],
                                                              x[GP_FEATURES].iloc[val])
        else:  # lgbm_feature
            cols = pd.concat([x, physics[variant][POSITION_TARGETS].add_prefix("phys_")], axis=1)
            pred = lgbm().fit(cols.iloc[fit], y.iloc[fit]).predict(cols.iloc[val])
    return kind, variant, target, val, pred


def hybrid(train):
    """Uses the all-train calibration, so wind is slightly optimistic; the pipeline's CV is the honest version."""
    x = feature_frame(train)
    scales = mean_scales(train)
    folds = list(RANDOM_FOLDS.split(x))
    cal = load_calibration()
    shots = pd.read_csv(ROOT / "outputs" / "calibration_shots.csv")
    spin_oof = np.zeros(len(train))
    for fit, val in folds:
        spin_oof[val] = lgbm().fit(x.iloc[fit], train.launch_spin_rate.iloc[fit]).predict(x.iloc[val])

    n = len(train)
    prior_mean = np.column_stack([spin_oof / 1000, np.zeros(n)] + [np.full(n, shots[k].mean()) for k in ("dd", "dl", "dh")])
    prior_sd = np.tile([0.01, 0.3] + [shots[k].std() for k in ("dd", "dl", "dh")], (n, 1))
    physics = {}
    for variant, wind in [("wind", wind_for(train, cal["wind"])), ("nowind", np.zeros((n, 3)))]:
        theta, _ = invert(train, cal["aero"], wind, prior_mean, prior_sd, sigma_cp=(0.015, 0.15, 0.3))
        physics[variant] = fly(train, theta, cal["aero"], wind)

    configs = [("gp_plain", None)] + [(k, v) for k in ["gp_feature", "gp_residual", "lgbm_feature"] for v in ["wind", "nowind"]]
    jobs = [delayed(_hybrid_task)(train, x, folds, physics, kind, v, t, k)
            for kind, v in configs for t in POSITION_TARGETS for k in range(5)]
    results = Parallel(n_jobs=20, verbose=5)(jobs)
    rows = {}
    for kind, v in configs:
        pred = pd.DataFrame(index=train.index, columns=POSITION_TARGETS, dtype=float)
        for kd, vr, target, val, p in results:
            if kd == kind and vr == v:
                pred.iloc[val, pred.columns.get_loc(target)] = p
        pred["launch_spin_rate"], pred["landing_h"] = spin_oof, 0.0
        rows[f"{kind}/{v}"] = summarise(errors(train, pred), scales)
    for v in ["wind", "nowind"]:
        rows[f"physics_only/{v}"] = summarise(errors(train, physics[v].assign(launch_spin_rate=spin_oof)), scales)
    print(pd.DataFrame(rows).T.round(3))


def _lag1(values, same):
    return np.corrcoef(values[same], np.roll(values, 1)[same])[0, 1]


def error_analysis(train, reps=2000, seed=0):
    """What, why and when of the final model's out-of-fold landing errors.

    Why: rank correlation of each factor with the relative landing error (error as % of carry), and the
    adjusted effect of each factor from a regression of log relative error on all standardised factors,
    both with bootstrap 95% intervals. When: Kruskal-Wallis across sessions, rank correlation with time of
    day and position in the hitting block, and the lag-1 autocorrelation of signed residuals between
    consecutive training shots against a within-block permutation null.
    Writes outputs/error_analysis.csv (per shot) and outputs/error_analysis.json (statistics).
    """
    rng = np.random.default_rng(seed)
    _, test = load()
    oof = pd.read_csv(ROOT / "outputs" / "pipeline_oof_gp_feature.csv").set_index("track_id").loc[train.track_id]
    diag = pd.read_csv(ROOT / "outputs" / "aero_diag.csv").set_index("track_id").loc[train.track_id]
    oof.index, diag.index = train.index, train.index
    err = errors(train, oof)

    t = pd.DataFrame({"track_id": train.track_id, "landing_error": err.landing,
                      "rel_error": 100 * err.landing / train.landing_d,
                      "res_d": oof.landing_d - train.landing_d, "res_l": oof.landing_l - train.landing_l,
                      "spin_error": (oof.launch_spin_rate - train.launch_spin_rate).abs(),
                      "aero_anomaly": np.hypot(diag.log_cd, diag.log_cl),
                      "curvature": np.degrees(diag.tilt.abs()),
                      "carry_beyond_net": train.landing_d - train.cp4_d,
                      "vla": train.vla, "upper_deck": train.elevated, "session": train.session})
    x = StandardScaler().fit_transform(feature_frame(train)[GP_FEATURES])
    t["novelty"] = NearestNeighbors(n_neighbors=11).fit(x).kneighbors(x)[0][:, 1:].mean(1)
    local = pd.to_datetime(train.launch_time, unit="s") + pd.Timedelta(hours=2)  # South African Standard Time
    t["local_hour"] = local.dt.hour + local.dt.minute / 60
    t["position_in_block"] = pd.concat([train, test]).groupby("block").launch_time.rank(pct=True).iloc[:len(train)].to_numpy()

    # Why
    factors = list(ERROR_FACTORS)
    y = np.log(t.rel_error.to_numpy())
    design = np.column_stack([np.ones(len(t)), StandardScaler().fit_transform(t[factors])])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    r2 = 1 - ((y - design @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    boot_rho, boot_beta = [], []
    ranks = t[factors + ["rel_error"]].rank().to_numpy()
    for _ in range(reps):
        idx = rng.integers(0, len(t), len(t))
        boot_beta.append(np.linalg.lstsq(design[idx], y[idx], rcond=None)[0][1:])
        boot_rho.append([np.corrcoef(ranks[idx, i], ranks[idx, -1])[0, 1] for i in range(len(factors))])
    boot_rho, boot_beta = np.array(boot_rho), np.array(boot_beta)
    why = {f: dict(label=ERROR_FACTORS[f], rho=float(spearmanr(t[f], t.rel_error)[0]),
                   rho_ci=np.nanpercentile(boot_rho[:, i], [2.5, 97.5]).tolist(),
                   pct_per_sd=float(100 * np.expm1(coef[i + 1])),
                   pct_per_sd_ci=(100 * np.expm1(np.percentile(boot_beta[:, i], [2.5, 97.5]))).tolist())
           for i, f in enumerate(factors)}

    # What
    worst = t.landing_error >= t.landing_error.quantile(0.9)
    profile = {f: dict(worst=float(t.loc[worst, f].median()), rest=float(t.loc[~worst, f].median()))
               for f in factors + ["landing_error", "rel_error"]}
    profile["upper_deck"] = dict(worst=float(t.loc[worst, "upper_deck"].mean()), rest=float(t.loc[~worst, "upper_deck"].mean()))
    order = t.sort_values("landing_error")
    cases = {"best": order.track_id.iloc[0], "typical": order.track_id.iloc[len(order) // 2],
             "95th percentile": order.track_id.iloc[int(0.95 * len(order))], "worst": order.track_id.iloc[-1]}

    # When
    sessions = [g.to_numpy() for _, g in t.groupby("session").rel_error]
    s = t.assign(block=train.block, time=train.launch_time).sort_values("time")
    same = s.block.eq(s.block.shift()).to_numpy()
    groups = [np.flatnonzero(s.block.to_numpy() == b) for b in s.block.unique()]
    lag1 = {}
    for col in ("res_d", "res_l"):
        values = s[col].to_numpy()
        null = []
        for _ in range(reps):
            shuffled = values.copy()
            for g in groups:
                shuffled[g] = values[rng.permutation(g)]
            null.append(_lag1(shuffled, same))
        observed = _lag1(values, same)
        lag1[col] = dict(observed=float(observed), null_ci=np.percentile(null, [2.5, 97.5]).tolist(),
                         p=float((np.abs(null) >= abs(observed)).mean()))
    # The same tests on the part of log relative error the why factors (shot type) do not explain,
    # so session and timing effects are not just a proxy for which clubs were being hit.
    t["adjusted_residual"] = y - design @ coef
    adjusted = [g.to_numpy() for _, g in t.groupby("session").adjusted_residual]

    def test(stat, *args):
        return dict(zip(("statistic", "p"), map(float, stat(*args))))

    when = dict(sessions=test(kruskal, *sessions), sessions_adjusted=test(kruskal, *adjusted),
                hour=test(spearmanr, t.local_hour, t.rel_error),
                hour_adjusted=test(spearmanr, t.local_hour, t.adjusted_residual),
                position=test(spearmanr, t.position_in_block, t.rel_error),
                position_adjusted=test(spearmanr, t.position_in_block, t.adjusted_residual),
                lag1=lag1)

    summary = dict(n=len(t), why=why, r2=float(r2), worst_decile=profile, cases=cases, when=when)
    t.to_csv(ROOT / "outputs" / "error_analysis.csv", index=False)
    (ROOT / "outputs" / "error_analysis.json").write_text(json.dumps(summary, indent=2))

    print(f"why (relative landing error, n={len(t)}, adjusted R² {r2:.2f}):")
    for f, w in why.items():
        print(f"  {w['label']:36s} rho {w['rho']:+.2f} [{w['rho_ci'][0]:+.2f}, {w['rho_ci'][1]:+.2f}]   "
              f"{w['pct_per_sd']:+6.1f}% per SD [{w['pct_per_sd_ci'][0]:+.1f}, {w['pct_per_sd_ci'][1]:+.1f}]")
    print("what: worst decile vs the rest (medians; upper deck is a share):")
    for f, p in profile.items():
        print(f"  {f:18s} {p['worst']:9.3f} vs {p['rest']:9.3f}")
    print("cases:", cases)
    print("when:", json.dumps(when, indent=1))


if __name__ == "__main__":
    studies = {"explore": explore, "baselines": baselines, "neighbours": neighbours, "hybrid": hybrid,
               "error_analysis": error_analysis}
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", choices=studies)
    train, _ = load()
    studies[parser.parse_args().study](train)
