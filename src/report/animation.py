"""Animated full trajectories: tracked flight to the net, modelled flight beyond it, bounce and roll.

    python -m report.animation [tag]    renders figures/test_shot.gif and figures/test_shots_mix.gif

Run from src/ or with PYTHONPATH=src, after `python -m modelling.pipeline --submit`.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

from data.dataset import POINTS, ROOT, load
from flight.ground import bounce_and_roll
from flight.inversion import RECONCILE_PARAMS, scaled_aero, wind_for
from flight.simulator import RPM_TO_RAD, simulate
from report import style

PHASES = {"tracked": (style.SERIES[0], "tracked up to the net"),
          "modelled": (style.SERIES[1], "modelled beyond the net"),
          "ground": (style.SERIES[2], "bounce and roll")}
GRASS, NET_HEIGHT = "#e9efdf", 30.0


def trajectories(df, params, spin_rpm, aero, wind, dt=0.02):
    """Launch-to-rest path of each shot in its bay's frame: d downrange, l lateral (+ left), z above ground."""
    log_cd, log_cl, tilt, dd, dl, dh = params[RECONCILE_PARAMS].to_numpy().T
    z0 = df.launch_z.to_numpy()
    o = simulate(df[["vd", "vl", "vh"]].to_numpy(), spin_rpm, tilt, scaled_aero(aero, log_cd, log_cl),
                 wind=wind, dt=dt, keep_path=True, land_h=-z0 - dh)
    shots = []
    for i, row in enumerate(df.itertuples()):
        live = o["path_t"] < o["landing_t"][i]
        impact = np.array([o["landing_d"][i] + dd[i], o["landing_l"][i] + dl[i], 0.0])
        air = np.vstack([o["path"][live, i] + [dd[i], dl[i], dh[i] + z0[i]], impact])
        t_air = np.append(o["path_t"][live], o["landing_t"][i])
        rest = bounce_and_roll(impact, o["landing_vel"][i], o["landing_spin"][i] * RPM_TO_RAD * o["axis"][i])
        top = int(np.argmax(air[:, 2]))
        shots.append(dict(
            t_air=t_air, air=air, t_ground=t_air[-1] + rest["t"], ground=rest["path"], net_d=row.cp4_d,
            tee_z=z0[i], apex=air[top], t_apex=t_air[top], spin=float(spin_rpm[i]),
            checkpoints=np.array([[getattr(row, f"{p}_d"), getattr(row, f"{p}_l"), getattr(row, f"{p}_h") + z0[i]]
                                  for p in POINTS])))
    return shots


def test_trajectories(ids, tag="gp_feature"):
    """Trajectories for the given test track ids from the pipeline's saved outputs."""
    _, test = load()
    pred = pd.read_csv(ROOT / "outputs" / "test_predictions_local.csv").set_index("track_id")
    params = pd.read_csv(ROOT / "outputs" / "test_trajectory_params.csv").set_index("track_id")
    final = json.loads((ROOT / "outputs" / "calibration_final.json").read_text())
    final["wind"] = {int(k): v for k, v in final["wind"].items()}
    pick = test.set_index("track_id").loc[ids].reset_index()
    wind = np.zeros((len(pick), 3)) if "nowind" in tag else wind_for(pick, final["wind"])
    return trajectories(pick, params.loc[ids], pred.loc[ids].launch_spin_rate.to_numpy(), final["aero"], wind)


def _at(points, times, t):
    return np.array([np.interp(t, times, points[:, c]) for c in range(3)])


def _state(shot, t):
    """Ball position, speed and phase at time t."""
    in_air = t < shot["t_air"][-1]
    pts, times = (shot["air"], shot["t_air"]) if in_air else (shot["ground"], shot["t_ground"])
    pos = _at(pts, times, t)
    ahead = min(t + 0.02, times[-1])
    speed = np.linalg.norm(_at(pts, times, ahead) - pos) / max(ahead - t, 1e-9) if ahead > t else 0.0
    phase = ("tracked by radar" if pos[0] < shot["net_d"] else "modelled flight") if in_air else "bounce and roll"
    return pos, speed, phase


def _segments(shot, t):
    seen = shot["air"][:np.searchsorted(shot["t_air"], t, side="right")]
    cut = int(np.searchsorted(seen[:, 0], shot["net_d"]))
    on_ground = t >= shot["t_air"][-1]
    ground = shot["ground"][:np.searchsorted(shot["t_ground"], t, side="right")] if on_ground else shot["ground"][:0]
    return {"tracked": seen[:cut + 1], "modelled": seen[cut:] if cut < len(seen) else seen[:0], "ground": ground}


def _scenery(ax3, d_max, l_lim, z_max, net_d, tee_heights):
    d_grid, l_grid = np.meshgrid([0, d_max], [-l_lim, l_lim])
    ax3.plot_surface(d_grid, l_grid, np.zeros_like(d_grid), color=GRASS, alpha=1.0, shade=False, zorder=0)
    for m in range(50, int(d_max) + 1, 50):
        ax3.plot([m, m], [-l_lim, l_lim], [0, 0], color=style.AXIS, lw=0.7, zorder=1)
    width = 0.8 * l_lim
    net_l, net_z = np.meshgrid([-width, width], [0, NET_HEIGHT])
    ax3.plot_surface(np.full_like(net_l, net_d), net_l, net_z, color=style.MUTED, alpha=0.15, shade=False, zorder=1)
    for l in (-width, width):
        ax3.plot([net_d, net_d], [l, l], [0, NET_HEIGHT], color=style.INK2, lw=1.2, zorder=2)
    ax3.plot([net_d, net_d], [-width, width], [NET_HEIGHT, NET_HEIGHT], color=style.INK2, lw=0.8, zorder=2)
    for z in {round(h, 2) for h in tee_heights if h > 1}:  # upper-deck bay
        ax3.plot([0, 0], [0, 0], [0, z], color=style.INK2, lw=2, zorder=2)
        ax3.plot([-3, 3, 3, -3, -3], [-3, -3, 3, 3, -3], [z] * 5, color=style.INK2, lw=1, zorder=2)
    ax3.set(xlim=(0, d_max), ylim=(-l_lim, l_lim), zlim=(0, z_max))
    ax3.set_xlabel("downrange (m)", labelpad=6)
    ax3.set_ylabel("left of target (m)", labelpad=2)
    ax3.set_zlabel("height (m)", labelpad=0)
    ax3.set_xticks(np.arange(0, d_max + 1, 50))
    ax3.set_yticks([-l_lim // 10 * 10, 0, l_lim // 10 * 10])
    ax3.set_zticks(np.arange(0, z_max + 1, 10))
    ax3.tick_params(labelsize=7, pad=0)
    ax3.set_box_aspect((3.2, 1.0, 0.8), zoom=1.2)
    ax3.set_proj_type("persp", focal_length=0.6)
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
    ax3.grid(False)


def animate(shots, title, fps=20, hold=2.0, rotate=True):
    """Build the animation; save it with `save_animation` or show it in a notebook via to_jshtml()."""
    t_end = max(s["t_ground"][-1] for s in shots)
    frames = np.arange(0.0, t_end + hold, 1.0 / fps)
    pts = np.vstack([np.vstack([s["air"], s["ground"]]) for s in shots])
    d_max = max(100.0, np.ceil(pts[:, 0].max() * 1.06 / 25) * 25)
    l_lim = max(20.0, np.ceil(np.abs(pts[:, 1]).max() * 1.25 / 10) * 10)
    z_max = max(NET_HEIGHT, np.ceil(pts[:, 2].max() * 1.15 / 10) * 10)
    net_d = float(np.mean([s["net_d"] for s in shots]))
    labelled = len(shots) == 1

    fig = plt.figure(figsize=(13, 6.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[2.1, 1], hspace=0.5, wspace=0.3, left=0.0, right=0.98)
    ax3 = fig.add_subplot(grid[:, 0], projection="3d", computed_zorder=False)
    side, top = fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    _scenery(ax3, d_max, l_lim, z_max, net_d, [s["tee_z"] for s in shots])

    for ax, ylim, ylabel, name in [(side, (0, z_max), "height (m)", "side view"),
                                   (top, (-l_lim, l_lim), "left of target (m)", "from above")]:
        ax.set(xlim=(0, d_max), ylim=ylim, ylabel=ylabel)
        ax.set_title(name)
        ax.axvline(net_d, color=style.INK2, lw=1.5)
    top.set_xlabel("downrange (m)")
    side.text(net_d + 2, z_max * 0.9, "net", color=style.INK2, fontsize=8)

    ball = dict(ms=6, mfc="white", mec=style.INK, mew=1.2, ls="", zorder=5)
    artists = []
    for s in shots:
        a = {ph: (ax3.plot([], [], [], color=c, lw=2, zorder=3)[0], side.plot([], [], color=c, lw=1.5)[0],
                  top.plot([], [], color=c, lw=1.5)[0]) for ph, (c, _) in PHASES.items()}
        a["shadow"] = ax3.plot([], [], [], "o", ms=4, color=style.MUTED, alpha=0.5, zorder=2)[0]
        a["ball"] = (ax3.plot([], [], [], "o", **ball)[0], side.plot([], [], "o", **ball)[0],
                     top.plot([], [], "o", **ball)[0])
        ax3.plot(*s["checkpoints"].T, "o", ms=3, color=style.SERIES[0], zorder=4)
        side.plot(*s["checkpoints"][:, [0, 2]].T, "o", ms=3, color=style.SERIES[0])
        note = dict(textcoords="offset points", ha="center", fontsize=8, color=style.INK2, visible=False)
        a["apex"] = side.annotate(f"apex {s['apex'][2]:.0f} m", s["apex"][[0, 2]], xytext=(0, 6), **note)
        a["carry"] = top.annotate(f"carry {s['air'][-1][0]:.0f} m", s["air"][-1][[0, 1]], xytext=(0, 8), **note)
        a["rest"] = top.annotate(f"rest {s['ground'][-1][0]:.0f} m", s["ground"][-1][[0, 1]], xytext=(0, -14), **note)
        artists.append(a)
    info = ax3.text2D(0.03, 0.9, "", transform=ax3.transAxes, color=style.INK, fontsize=9.5, va="top")
    ax3.legend(handles=[Line2D([], [], color=c, lw=2, label=label) for c, label in PHASES.values()],
               loc="lower left", bbox_to_anchor=(0.02, 0.04), fontsize=8.5)

    def update(k):
        t = frames[k]
        for s, a in zip(shots, artists):
            for ph, seg in _segments(s, t).items():
                line3, line_side, line_top = a[ph]
                line3.set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])
                line_side.set_data(seg[:, 0], seg[:, 2])
                line_top.set_data(seg[:, 0], seg[:, 1])
            pos, speed, phase = _state(s, t)
            a["ball"][0].set_data_3d([pos[0]], [pos[1]], [pos[2]])
            a["ball"][1].set_data([pos[0]], [pos[2]])
            a["ball"][2].set_data([pos[0]], [pos[1]])
            a["shadow"].set_data_3d([pos[0]], [pos[1]], [0.0])
            if labelled:
                a["apex"].set_visible(t >= s["t_apex"])
                a["carry"].set_visible(t >= s["t_air"][-1])
                a["rest"].set_visible(t >= s["t_ground"][-1])
                info.set_text(f"{t:4.1f} s  ·  {phase}\n{speed:4.1f} m/s   height {pos[2]:4.1f} m   "
                              f"downrange {pos[0]:5.1f} m\nlaunch spin {s['spin']:,.0f} rpm")
        if rotate:
            ax3.view_init(elev=14, azim=-58 + 22 * t / frames[-1])
        return []

    return FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps)


def save_animation(anim, path, fps=20, dpi=90):
    writer = PillowWriter(fps=fps) if str(path).endswith(".gif") else FFMpegWriter(fps=fps, bitrate=2400)
    anim.save(path, writer=writer, dpi=dpi)
    plt.close(anim._fig)


def render_submission(tag="gp_feature", fps=20):
    """Render a single test shot and a mix of five test shots from the pipeline's saved outputs."""
    spins = pd.read_csv(ROOT / "outputs" / "test_predictions_local.csv").set_index("track_id").launch_spin_rate
    single = [(spins - spins.quantile(0.2)).abs().idxmin()]
    save_animation(animate(test_trajectories(single, tag), "A test shot: tracked to the net, predicted all the way to rest"),
                   ROOT / "figures" / "test_shot.gif", fps)
    mix = [(spins - spins.quantile(q)).abs().idxmin() for q in (0.05, 0.3, 0.55, 0.8, 0.97)]
    save_animation(animate(test_trajectories(mix, tag), "Five test shots, from low-spin to wedge-like"),
                   ROOT / "figures" / "test_shots_mix.gif", fps)


if __name__ == "__main__":
    import sys

    style.apply()
    render_submission(sys.argv[1] if len(sys.argv) > 1 else "gp_feature")
