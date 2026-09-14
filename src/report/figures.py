"""Report figures: exploratory analysis (`eda`) and model results and error analysis (`results`).

    python -m report.figures eda                 figures 01-05
    python -m report.figures results [--tag T]   figures 06-15 (needs the pipeline's CV and submission outputs
                                                 and `python -m experiments.ml_studies error_analysis`)

Run from src/ or with PYTHONPATH=src.
"""
import argparse
import json

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from data.dataset import CP_PLANES, HEADING, ROOT, U, V, load
from data.features import feature_frame, kinematic_features
from flight.inversion import load_calibration, reconcile, wind_for
from flight.simulator import coefficients, simulate
from modelling.evaluation import errors
from report import style
from report.animation import PHASES, test_trajectories, trajectories

style.apply()
train, test = load()
cal = load_calibration()


def fig_geometry():
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    lat = np.array([-25.0, 20.0])
    for k, s in enumerate(CP_PLANES):
        p = s * U[None, :] + lat[:, None] * V[None, :]
        net = k == 3
        ax.plot(p[:, 0], p[:, 1], color=style.INK2 if net else style.AXIS, lw=2 if net else 1,
                ls="-" if net else "--", zorder=1)
        ax.annotate("net" if net else f"cp{k + 1}", p[0], xytext=(0, -12), textcoords="offset points",
                    ha="center", color=style.INK2, fontsize=8.5)
    sc = ax.scatter(train.landing_x, train.landing_y, c=train.launch_spin_rate, cmap=style.SEQ,
                    s=18, lw=0, zorder=2)
    tees = pd.concat([train, test]).groupby("tee")[["launch_x", "launch_y", "launch_z"]].first()
    ax.scatter(tees.launch_x, tees.launch_y, marker="s", s=40, color=style.SERIES[1], zorder=3)
    raised = tees[tees.launch_z > 1].iloc[0]
    ax.annotate("four bays (one on an\nupper deck, 4.1 m up)", (raised.launch_x, raised.launch_y),
                xytext=(-10, -34), textcoords="offset points", ha="left", color=style.INK2, fontsize=8.5)
    start = tees[["launch_x", "launch_y"]].mean().to_numpy()
    ax.annotate("", xy=start + 250 * U, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=style.MUTED, lw=1), zorder=0)
    ax.text(*(start + 245 * U + 5 * V), "downrange, 28.4° from x", color=style.INK2, fontsize=8.5,
            rotation=np.degrees(HEADING), rotation_mode="anchor", ha="right", va="bottom", zorder=4,
            bbox=dict(boxstyle="round,pad=0.25", fc=style.SURFACE, ec="none", alpha=0.9))
    fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02).set_label("launch spin (rpm)", color=style.INK2)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Range layout: training landings, coloured by spin")
    style.save(fig, "01_range_geometry")


def fig_side_view(n=28, seed=3):
    diag = pd.read_csv(ROOT / "outputs" / "aero_diag.csv").set_index("track_id")
    sub = train.sample(n, random_state=seed)
    d = diag.loc[sub.track_id]
    base = cal["aero"]
    aero = dict(base, cd0=base["cd0"] * d.cd_mult.to_numpy(), cd1=base["cd1"] * d.cd_mult.to_numpy(),
                cl_max=base["cl_max"] * d.cl_mult.to_numpy())
    o = simulate(sub[["vd", "vl", "vh"]].to_numpy(), sub.launch_spin_rate.to_numpy(), d.tilt.to_numpy(),
                 aero, wind=wind_for(sub, cal["wind"]), keep_path=True)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    net = train.cp4_d.mean()
    ax.axvspan(net, 265, color=style.GRID, alpha=0.45, lw=0, zorder=0)
    ax.axvline(net, color=style.INK2, lw=2, zorder=1)
    ax.text(net + 3, 46, "net: everything to the right\nis hidden on a netted range", color=style.INK2,
            fontsize=8.5, va="top")
    for i in range(n):
        live = o["path_t"] <= o["landing_t"][i]
        dd = o["path"][live, i, 0] + d.dd.iloc[i]
        hh = o["path"][live, i, 2] + d.dh.iloc[i]
        seen = dd <= sub.cp4_d.iloc[i]
        ax.plot(dd[seen], hh[seen], color=style.SERIES[0], lw=1.3, zorder=2)
        ax.plot(dd[~seen], hh[~seen], color=style.BLUE_RAMP[3], lw=1.0, zorder=2)
    ax.scatter(sub.apex_d, sub.apex_h, s=22, facecolor="none", edgecolor=style.INK2, lw=1, zorder=3)
    ax.scatter(sub.landing_d, sub.landing_h, s=26, marker="v", color=style.SERIES[1], lw=0, zorder=3)
    frac_d = (train.cp4_d / train.landing_d).mean()
    frac_t = (train.cp4_t / train.landing_t).mean()
    ax.set_xlim(0, 265)
    ax.set_ylim(0, 50)
    ax.set_xlabel("downrange from the bay (m)")
    ax.set_ylabel("height above tee (m)")
    ax.legend(handles=[Line2D([], [], color=style.SERIES[0], lw=1.3, label="seen before the net"),
                       Line2D([], [], color=style.BLUE_RAMP[3], lw=1, label="flight behind the net"),
                       Line2D([], [], marker="o", ls="", mfc="none", mec=style.INK2, label="measured apex"),
                       Line2D([], [], marker="v", ls="", color=style.SERIES[1], label="measured landing")],
              loc="upper right", ncols=2)
    ax.set_title(f"The net sees {frac_d:.0%} of the carry and {frac_t:.0%} of the flight time "
                 f"({n} random training shots)")
    style.save(fig, "02_what_the_net_hides")


def fig_train_test():
    both = pd.concat([feature_frame(train), feature_frame(test)])
    y = np.r_[np.zeros(len(train)), np.ones(len(test))]
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, n_jobs=4, verbose=-1)
    p = cross_val_predict(clf, both, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                          method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)
    cols = {"launch speed (m/s)": "speed", "vertical launch angle (°)": "vla",
            "horizontal launch angle (°)": "hla", "time to reach the net (s)": "cp4_t"}
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.9))
    for ax, (label, col) in zip(axes, cols.items()):
        bins = np.histogram_bin_edges(pd.concat([train[col], test[col]]), 28)
        ax.hist(train[col], bins, density=True, histtype="step", lw=1.5, color=style.SERIES[0],
                label=f"train ({len(train)})")
        ax.hist(test[col], bins, density=True, histtype="step", lw=1.5, color=style.SERIES[1],
                label=f"test ({len(test)})")
        ax.set_xlabel(label)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="upper left")
    fig.suptitle(f"Train and test come from the same sessions: a classifier can barely tell them apart "
                 f"(AUC {auc:.2f})", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    style.save(fig, "03_train_vs_test")
    return auc


def fig_spin_fingerprint():
    kin = kinematic_features(train)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, col, label in [(axes[0], "clu_mid", "lift coefficient measured between checkpoints"),
                           (axes[1], "cd_mid", "drag coefficient measured between checkpoints")]:
        ax.scatter(kin[col], train.launch_spin_rate, s=14, color=style.SERIES[0], alpha=0.55, lw=0)
        r = np.corrcoef(kin[col], train.launch_spin_rate)[0, 1]
        ax.text(0.03, 0.95, f"r = {r:.2f}", transform=ax.transAxes, va="top", color=style.INK2)
        ax.set_xlabel(label)
    axes[0].set_ylabel("launch-monitor spin (rpm)")
    fig.suptitle("Spin leaves a fingerprint in the first 60 m", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    style.save(fig, "04_spin_fingerprint")


def fig_session():
    both = pd.concat([train.assign(split="train"), test.assign(split="test")]).sort_values("launch_time")
    s = train.sort_values("launch_time")
    same = s.block.eq(s.block.shift())
    lag = np.corrcoef(s.launch_spin_rate[same], s.launch_spin_rate.shift()[same])[0, 1]
    blk = train.block.value_counts().idxmax()
    b = both[both.block == blk]
    t0 = b.launch_time.min()
    fig, ax = plt.subplots(figsize=(9, 3.3))
    tr = b[b.split == "train"]
    ax.plot((tr.launch_time - t0) / 60, tr.launch_spin_rate, "-o", color=style.SERIES[0], ms=5, lw=1,
            label="training shot spin")
    te = b[b.split == "test"]
    ax.scatter((te.launch_time - t0) / 60, np.full(len(te), 0.03), marker="|", s=90, color=style.SERIES[1],
               transform=ax.get_xaxis_transform(), label="test shot (spin unknown)")
    ax.set_xlabel(f"minutes into the session ({pd.to_datetime(t0, unit='s'):%d %b %Y})")
    ax.set_ylabel("spin (rpm)")
    ax.legend(loc="upper right", ncols=2)
    ax.set_title(f"Players hit runs with one club: consecutive shots' spin correlates at r = {lag:.2f}")
    style.save(fig, "05_same_club_runs")
    return lag


def fig_model_comparison():
    """Landing error per approach and per ablation of the final model, from outputs/cv_summary.json."""
    summary = json.loads((ROOT / "outputs" / "cv_summary.json").read_text())
    labels, values, headings = [], [], []
    for group, models in summary.items():
        headings.append(len(labels))
        labels.append(group)
        values.append(np.nan)
        for name, metrics in models.items():
            labels.append(name)
            values.append(metrics["landing"])
    values = np.array(values)
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 0.4 * len(labels) + 1.1))
    ax.barh(y, np.nan_to_num(values), height=0.62, color=style.SERIES[0])
    for yi, v in zip(y, values):
        if np.isfinite(v):
            ax.text(v + 0.06, yi, f"{v:.2f} m", va="center", color=style.INK2, fontsize=9)
    ax.set_yticks(y, labels)
    for i in headings:
        ax.get_yticklabels()[i].set_fontweight("bold")
        ax.get_yticklabels()[i].set_color(style.INK)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, np.nanmax(values) * 1.15)
    ax.set_xlabel("mean landing-position error (m), 5-fold cross-validation on train")
    ax.set_title("Approaches and ablations: what each ingredient of the final model contributes")
    style.save(fig, "06_model_comparison")


def _oof(tag):
    return pd.read_csv(ROOT / "outputs" / f"pipeline_oof_{tag}.csv").set_index("track_id").loc[train.track_id]


def fig_predicted_vs_actual(tag):
    oof = _oof(tag)
    err = errors(train, oof)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    for ax, col, label in [(axes[0], "landing_d", "carry downrange (m)"),
                           (axes[1], "landing_l", "landing lateral (m, + = left)")]:
        truth, pred = train[col].to_numpy(), oof[col].to_numpy()
        lo, hi = min(truth.min(), pred.min()), max(truth.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], color=style.AXIS, lw=1)
        ax.scatter(truth, pred, s=12, color=style.SERIES[0], alpha=0.6, lw=0)
        ax.set_xlabel(f"measured {label}")
        ax.set_ylabel("predicted")
        ax.text(0.03, 0.95, f"mean abs error {np.abs(truth - pred).mean():.1f} m", transform=ax.transAxes,
                va="top", color=style.INK2)
    axes[2].hist(err.landing, bins=30, color=style.SERIES[0], rwidth=0.85)
    for q, label, height in [(0.5, "median", 0.95), (0.9, "90th percentile", 0.8)]:
        v = err.landing.quantile(q)
        axes[2].axvline(v, color=style.INK2, lw=1, ls="--")
        axes[2].text(v, axes[2].get_ylim()[1] * height, f" {label} {v:.1f} m", color=style.INK2, fontsize=8.5,
                     va="top")
    axes[2].set_xlabel("landing-position error (m)")
    axes[2].set_ylabel("shots")
    axes[2].grid(axis="x", visible=False)
    fig.suptitle("Out-of-fold predictions for every training shot", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    style.save(fig, "07_predicted_vs_actual")


def fig_error_by_shot(tag):
    err = errors(train, _oof(tag))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for ax, col, label in [(axes[0], "launch_spin_rate", "launch spin (rpm)"), (axes[1], "speed", "launch speed (m/s)")]:
        bins = pd.qcut(train[col], 6)
        stats = err.landing.groupby(bins, observed=True).quantile([0.25, 0.5, 0.75]).unstack()
        centres = train[col].groupby(bins, observed=True).median()
        ax.vlines(centres, stats[0.25], stats[0.75], color=style.BLUE_RAMP[4], lw=4)
        ax.plot(centres, stats[0.5], "o", color=style.SERIES[0], ms=7)
        ax.set_xlabel(f"{label}, sextiles")
    axes[0].set_ylabel("landing error (m)")
    fig.suptitle("Landing error by shot type (dot = median, bar = middle 50%)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    style.save(fig, "08_error_by_shot_type")


def fig_aerodynamics():
    diag = pd.read_csv(ROOT / "outputs" / "aero_diag.csv")
    a = cal["aero"]
    S = np.linspace(0.02, 0.8, 200)
    cd, cl = coefficients(S, a)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, pts, curve, label in [(axes[0], diag.cl_eff, cl, "lift coefficient C_L"),
                                  (axes[1], diag.cd_eff, cd, "drag coefficient C_D")]:
        ax.scatter(diag.S0, pts, s=10, color=style.BLUE_RAMP[4], alpha=0.5, lw=0, label="per-shot fit")
        ax.plot(S, curve, color=style.INK, lw=2, label="calibrated curve")
        ax.set_xlabel("spin factor at launch, S = rω / v")
        ax.set_ylabel(label)
    axes[0].legend(loc="lower right")
    fig.suptitle(f"Calibrated range-ball aerodynamics:  C_D = {a['cd0']:.3f} + {a['cd1']:.3f}·S,   "
                 f"C_L = {a['cl_max']:.3f}·S / (S + {a['s_half']:.3f})", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    style.save(fig, "09_aerodynamics")


def fig_wind():
    starts = pd.concat([train, test]).groupby("block").launch_time.min().sort_values()
    wind = pd.DataFrame(cal["wind"], index=["along", "across"]).T
    wind = wind.loc[[b for b in starts.index if b in wind.index]]
    # launch_time is Unix/POSIX seconds (UTC); the range is in South African Standard Time, UTC+2.
    labels = [(pd.to_datetime(starts[b], unit="s") + pd.Timedelta(hours=2)).strftime("%d %b %H:%M") for b in wind.index]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    y = np.arange(len(wind))
    for ax, col, label in [(axes[0], "along", "along the range (m/s, + = tailwind)"),
                           (axes[1], "across", "across the range (m/s, + = blowing left)")]:
        ax.axvline(0, color=style.AXIS, lw=1)
        ax.hlines(y, 0, wind[col], color=style.BLUE_RAMP[4], lw=2)
        ax.plot(wind[col], y, "o", color=style.SERIES[0], ms=6)
        ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)
    mean_along = wind.along.mean()
    axes[0].axvline(mean_along, color=style.INK2, lw=1, ls="--")
    axes[0].set_xlabel(f"along the range (m/s, + = tailwind)\ndashed: mean {mean_along:.1f} m/s, an offset that "
                       "absorbs unmodelled drag")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle("Effective wind per hitting block (local start time), fitted from training landings: the "
                 "spread between blocks is the weather", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    style.save(fig, "10_session_wind")


def fig_test_flights(tag, n=40):
    pred = pd.read_csv(ROOT / "outputs" / "test_predictions_local.csv").set_index("track_id").loc[test.track_id]
    shots = test_trajectories(test.track_id.sample(n, random_state=1).tolist(), tag)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw=dict(width_ratios=[1, 2.1]))
    sc = ax1.scatter(-pred.landing_l, pred.landing_d, c=pred.launch_spin_rate, cmap=style.SEQ, s=12, lw=0)
    ax1.set(xlabel="right of target (m)", ylabel="downrange (m)", aspect="equal")
    fig.colorbar(sc, ax=ax1, shrink=0.85).set_label("predicted spin (rpm)", color=style.INK2)
    ax1.set_title(f"Predicted landings, all {len(test)} test shots")
    ax2.axvline(test.cp4_d.mean(), color=style.INK2, lw=1.5)
    for s in shots:
        cut = int(np.searchsorted(s["air"][:, 0], s["net_d"]))
        ax2.plot(s["air"][:cut + 1, 0], s["air"][:cut + 1, 2], color=PHASES["tracked"][0], lw=1.2)
        ax2.plot(s["air"][cut:, 0], s["air"][cut:, 2], color=PHASES["modelled"][0], lw=1, alpha=0.85)
        ax2.plot(s["ground"][:, 0], s["ground"][:, 2], color=PHASES["ground"][0], lw=1)
    ax2.legend(handles=[Line2D([], [], color=c, lw=2, label=label) for c, label in PHASES.values()], loc="upper right")
    ax2.set(xlabel="downrange (m)", ylabel="height above ground (m)", ylim=(0, None), xlim=(0, None))
    ax2.set_title(f"Full predicted flights for {n} random test shots")
    fig.tight_layout()
    style.save(fig, "11_test_flights")


def fig_bounce_and_roll(tag):
    spins = pd.read_csv(ROOT / "outputs" / "test_predictions_local.csv").set_index("track_id").launch_spin_rate
    ids = [(spins - spins.quantile(q)).abs().idxmin() for q in (0.03, 0.35, 0.7, 0.97)]
    shots = test_trajectories(ids, tag)
    fig, axes = plt.subplots(len(shots), 1, figsize=(9, 6.5), sharex=True)
    for ax, s in zip(axes, shots):
        land = s["air"][-1]
        v = (s["air"][-1] - s["air"][-2]) / (s["t_air"][-1] - s["t_air"][-2])
        angle = np.degrees(np.arctan2(-v[2], np.hypot(v[0], v[1])))

        def along(p):
            return np.sign(p[:, 0] - land[0]) * np.hypot(p[:, 0] - land[0], p[:, 1] - land[1])

        tail = s["air"][s["air"][:, 0] >= land[0] - 25]
        ax.axhline(0, color=style.AXIS, lw=1)
        ax.plot(along(tail), tail[:, 2], color=style.SERIES[1], lw=1.5)
        ax.plot(along(s["ground"]), s["ground"][:, 2], color=style.SERIES[2], lw=1.8)
        run = along(s["ground"][-1:])[0]
        ax.plot([run], [0], "o", ms=6, mfc="white", mec=style.INK)
        ax.text(0.01, 0.92, f"{s['spin']:,.0f} rpm · lands at {np.linalg.norm(v):.0f} m/s and {angle:.0f}° · "
                f"comes to rest {run:+.1f} m from the pitch mark", transform=ax.transAxes, va="top",
                color=style.INK2, fontsize=9)
        ax.set_ylim(0, 3)
        ax.grid(axis="x", visible=False)
    axes[-1].set_xlabel("distance from the landing point (m)")
    fig.supylabel("height (m)", color=style.INK2, fontsize=9.5)
    fig.suptitle("Bounce and roll: low-spin shots release, high-spin shots check and spin back", x=0.01,
                 ha="left", fontweight="bold")
    fig.tight_layout()
    style.save(fig, "12_bounce_and_roll")


def _error_analysis():
    return (json.loads((ROOT / "outputs" / "error_analysis.json").read_text()),
            pd.read_csv(ROOT / "outputs" / "error_analysis.csv").set_index("track_id"))


def fig_case_studies():
    """What: measured and predicted flights for the best, a typical and the worst out-of-fold shots."""
    summary, table = _error_analysis()
    oof = _oof("gp_feature")
    diag = pd.read_csv(ROOT / "outputs" / "aero_diag.csv").set_index("track_id")
    ids = list(summary["cases"].values())
    pick = train.set_index("track_id").loc[ids].reset_index()
    wind = wind_for(pick, cal["wind"])
    spin_pred = oof.loc[ids].launch_spin_rate.to_numpy()
    measured = trajectories(pick, diag.loc[ids], pick.launch_spin_rate.to_numpy(), cal["aero"], wind)
    predicted = trajectories(pick, reconcile(pick, oof.loc[ids], spin_pred, cal["aero"], wind), spin_pred,
                             cal["aero"], wind)

    fig, axes = plt.subplots(len(ids), 2, figsize=(12, 2.6 * len(ids) + 0.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    for row, (case, tid) in enumerate(summary["cases"].items()):
        shot, m, p, info, d = pick.iloc[row], measured[row], predicted[row], table.loc[tid], diag.loc[tid]
        side, top = axes[row]
        cut = int(np.searchsorted(p["air"][:, 0], p["net_d"]))
        for ax, j in ((side, 2), (top, 1)):
            ax.axvline(shot.cp4_d, color=style.AXIS, lw=1)
            ax.plot(m["air"][:, 0], m["air"][:, j], color=style.INK2, lw=1.2, ls="--")
            ax.plot(p["air"][:cut + 1, 0], p["air"][:cut + 1, j], color=style.SERIES[0], lw=1.8)
            ax.plot(p["air"][cut:, 0], p["air"][cut:, j], color=style.SERIES[1], lw=1.8)
            ax.plot(p["checkpoints"][:, 0], p["checkpoints"][:, j], "o", ms=3.5, color=style.SERIES[0])
        z0 = shot.launch_z
        side.plot(shot.apex_d, shot.apex_h + z0, "o", mfc="none", mec=style.INK2, ms=7)
        side.plot(shot.landing_d, z0, "v", color=style.INK2, ms=7)
        side.plot(oof.loc[tid].landing_d, z0, "v", color=style.SERIES[1], ms=7)
        top.plot(shot.landing_d, shot.landing_l, "v", color=style.INK2, ms=7)
        top.plot(oof.loc[tid].landing_d, oof.loc[tid].landing_l, "v", color=style.SERIES[1], ms=7)
        side.set_title(f"{case}: landing off by {info.landing_error:.1f} m ({info.rel_error:.1f}% of carry)"
                       f"{', upper-deck bay' if shot.elevated else ''}")
        side.text(0.01, 0.95, f"spin {shot.launch_spin_rate:,.0f} rpm, predicted {oof.loc[tid].launch_spin_rate:,.0f}\n"
                  f"{shot.speed:.0f} m/s at {shot.vla:.0f}° · this ball: drag ×{d.cd_mult:.2f}, lift ×{d.cl_mult:.2f}, "
                  f"tilt {np.degrees(d.tilt):+.0f}°", transform=side.transAxes, va="top", fontsize=8.5, color=style.INK2)
        side.set(ylabel="height (m)", xlim=(0, None), ylim=(0, max(m["air"][:, 2].max(), p["air"][:, 2].max()) * 1.45))
        top.set(ylabel="left of target (m)", xlim=(0, None))
    axes[-1, 0].set_xlabel("downrange (m)")
    axes[-1, 1].set_xlabel("downrange (m)")
    fig.legend(handles=[Line2D([], [], color=style.INK2, lw=1.2, ls="--", label="measured flight"),
                        Line2D([], [], color=style.SERIES[0], lw=1.8, label="tracked to the net"),
                        Line2D([], [], color=style.SERIES[1], lw=1.8, label="predicted beyond the net"),
                        Line2D([], [], marker="o", ls="", mfc="none", mec=style.INK2, label="measured apex"),
                        Line2D([], [], marker="v", ls="", color=style.INK2, label="measured landing"),
                        Line2D([], [], marker="v", ls="", color=style.SERIES[1], label="predicted landing")],
               loc="upper right", ncols=3, bbox_to_anchor=(0.99, 1.0), fontsize=8.5)
    fig.suptitle("What the errors look like: measured and predicted flights", x=0.01, y=0.995, ha="left",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    style.save(fig, "13_case_studies")


def fig_error_attribution():
    """Why: association of shot-level factors with the relative landing error."""
    summary, _ = _error_analysis()
    why = summary["why"]
    order = sorted(why, key=lambda f: abs(why[f]["pct_per_sd"]))
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, key, title, xlabel in [
            (axes[0], "rho", "Univariate", "Spearman rank correlation with relative error"),
            (axes[1], "pct_per_sd", "Adjusted for all other factors", "change in relative error per 1 SD of the factor (%)")]:
        ax.axvline(0, color=style.AXIS, lw=1)
        ax.hlines(y, [why[f][f"{key}_ci"][0] for f in order], [why[f][f"{key}_ci"][1] for f in order],
                  color=style.BLUE_RAMP[4], lw=3)
        ax.plot([why[f][key] for f in order], y, "o", color=style.SERIES[0], ms=7)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [why[f]["label"] for f in order])
    fig.suptitle(f"Why shots miss: {why[order[-1]]['label']} has the largest adjusted effect on relative landing error",
                 x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, -0.02, f"n = {summary['n']} out-of-fold shots; relative error = landing error ÷ carry; bars are "
             f"bootstrap 95% intervals; regression R² = {summary['r2']:.2f} on log relative error.",
             color=style.INK2, fontsize=8.5)
    fig.tight_layout()
    style.save(fig, "14_error_attribution")


def _p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def fig_error_timing():
    """When: relative landing error by session, time of day and position in the hitting block, and carry-over
    between consecutive shots. Adjusted tests use the error left unexplained by the why factors (shot type)."""
    summary, table = _error_analysis()
    when = summary["when"]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2), gridspec_kw=dict(width_ratios=[1.3, 1, 1, 0.75]))

    stats = table.groupby("session").rel_error.quantile([0.25, 0.5, 0.75]).unstack().sort_index()
    x = np.arange(len(stats))
    axes[0].vlines(x, stats[0.25], stats[0.75], color=style.BLUE_RAMP[4], lw=4)
    axes[0].plot(x, stats[0.5], "o", color=style.SERIES[0], ms=7)
    axes[0].set_xticks(x, [pd.Timestamp(s).strftime("%d %b") for s in stats.index], rotation=45, ha="right")
    axes[0].set_ylabel("relative landing error (% of carry)")
    axes[0].set_title(f"By session\nKruskal–Wallis {_p(when['sessions']['p'])}; adjusted {_p(when['sessions_adjusted']['p'])}",
                      fontsize=9.5)
    axes[0].grid(axis="x", visible=False)

    axes[1].sharey(axes[2])
    axes[1].set_ylabel("relative landing error (% of carry)")
    for ax, col, width, label, key, xlabel in [
            (axes[1], "local_hour", 0.5, "By time of day", "hour", "local time (SAST, hours)"),
            (axes[2], "position_in_block", 0.1, "Through a hitting block", "position", "position in the block (0 = first, 1 = last)")]:
        ax.scatter(table[col], table.rel_error, s=8, color=style.BLUE_RAMP[3], alpha=0.5, lw=0)
        edges = np.arange(np.floor(table[col].min() / width) * width, table[col].max() + width, width)
        medians = table.groupby(pd.cut(table[col], edges), observed=True).rel_error.median()
        ax.plot([b.mid for b in medians.index], medians.to_numpy(), "-o", color=style.SERIES[0], ms=5)
        ax.set(ylim=(0, table.rel_error.quantile(0.98)), xlabel=xlabel)
        raw, adj = when[key], when[f"{key}_adjusted"]
        ax.set_title(f"{label}\nρ = {raw['statistic']:+.2f} ({_p(raw['p'])}); adjusted ρ = {adj['statistic']:+.2f} ({_p(adj['p'])})",
                     fontsize=9.5)

    for i, col in enumerate(("res_d", "res_l")):
        lag = when["lag1"][col]
        axes[3].vlines(i, *lag["null_ci"], color=style.GRID, lw=14)
        axes[3].plot(i, lag["observed"], "o", color=style.SERIES[0], ms=8)
        axes[3].text(i + 0.14, lag["observed"], _p(lag["p"]), va="center", fontsize=8.5, color=style.INK2)
    axes[3].axhline(0, color=style.AXIS, lw=1)
    axes[3].set(xlim=(-0.5, 1.9), ylabel="lag-1 correlation of residuals")
    axes[3].set_xticks([0, 1], ["downrange", "lateral"])
    axes[3].set_title("Carry-over between\nconsecutive shots", fontsize=9.5)
    axes[3].legend(handles=[Line2D([], [], color=style.GRID, lw=8, label="permutation null, 95%"),
                            Line2D([], [], marker="o", ls="", color=style.SERIES[0], label="observed")],
                   loc="lower right", fontsize=8)
    axes[3].grid(axis="x", visible=False)

    sessions = ("sessions still differ once shot type is accounted for" if when["sessions_adjusted"]["p"] < 0.05
                else "differences between sessions are explained by shot type")
    carry = ("consecutive residuals are correlated" if min(l["p"] for l in when["lag1"].values()) < 0.05
             else "consecutive residuals are independent")
    fig.suptitle(f"When shots miss: {sessions}, and {carry}", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    style.save(fig, "15_error_timing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("group", nargs="?", default="eda", choices=["eda", "results"])
    parser.add_argument("--tag", default="gp_feature")
    args = parser.parse_args()
    if args.group == "eda":
        fig_geometry()
        fig_side_view()
        auc = fig_train_test()
        fig_spin_fingerprint()
        lag = fig_session()
        print(f"adversarial AUC {auc:.3f}, lag-1 spin corr {lag:.3f}")
    else:
        fig_model_comparison()
        fig_predicted_vs_actual(args.tag)
        fig_error_by_shot(args.tag)
        fig_aerodynamics()
        fig_wind()
        fig_test_flights(args.tag)
        fig_bounce_and_roll(args.tag)
        fig_case_studies()
        fig_error_attribution()
        fig_error_timing()
