"""Batched point-mass golf-ball flight model: gravity, drag, Magnus lift, spin decay and wind.

Every shot is integrated at once with RK4 in its own launch-relative frame
(d = downrange, l = lateral with + to the left, h = height above the tee), so a few
thousand trajectories cost a handful of numpy operations per time step.
"""
import numpy as np

G = 9.81
MASS = 0.04593          # kg, USGA maximum
RADIUS = 0.021335       # m, USGA minimum diameter / 2
AREA = np.pi * RADIUS**2
RPM_TO_RAD = 2 * np.pi / 60

# Starting aerodynamic model; the free coefficients are calibrated on the training shots.
DEFAULT_AERO = dict(
    rho=1.19,                    # kg/m^3, ~120 m ASL at ~20 C
    cd0=0.20, cd1=0.25,          # CD = cd0 + cd1 * S
    cl_max=0.45, s_half=0.12,    # CL = cl_max * S / (S + s_half)
    decay=1.0e-3,                # d(omega)/dt = -decay * omega * |v_air|   [1/m]
)


def coefficients(S, aero):
    cd = aero["cd0"] + aero["cd1"] * S
    cl = aero["cl_max"] * S / (S + aero["s_half"])
    return cd, cl


def spin_axis(v0, tilt):
    """Unit spin axis perpendicular to the launch velocity; tilt > 0 curves the ball left."""
    vhat = v0 / np.linalg.norm(v0, axis=1, keepdims=True)
    side = np.stack([vhat[:, 1], -vhat[:, 0], np.zeros(len(v0))], axis=1)  # horizontal, to the right
    side /= np.linalg.norm(side, axis=1, keepdims=True)
    up = np.cross(side, vhat)
    return np.cos(tilt)[:, None] * side + np.sin(tilt)[:, None] * up


def _accel(vel, om, axis, wind, aero):
    va = vel - wind
    speed = np.linalg.norm(va, axis=1)
    cd, cl = coefficients(RADIUS * om / speed, aero)
    k = aero["rho"] * AREA / (2 * MASS)
    acc = k * speed[:, None] * (cl[:, None] * np.cross(axis, va) - cd[:, None] * va)
    acc[:, 2] -= G
    return acc, -aero["decay"] * om * speed


def simulate(v0, spin_rpm, tilt=0.0, aero=DEFAULT_AERO, wind=None, cp_d=None,
             dt=0.01, t_max=12.0, keep_path=False, stop="landing", land_h=0.0):
    """Fly every shot until it comes back down to height land_h and report checkpoint, apex and
    landing events. land_h = 0 is the level landing (tee height); -launch_z reaches the ground.

    v0: (N, 3) launch velocity (d, l, h); spin_rpm, tilt, land_h: (N,) or scalar;
    wind: (N, 3) or (3,); cp_d: (N, K) downrange distances at which to report crossings.
    stop="cps" ends the integration once every shot has crossed every checkpoint.
    """
    vel = np.array(v0, dtype=float)
    n = len(vel)
    pos = np.zeros((n, 3))
    om = np.broadcast_to(np.asarray(spin_rpm, float), (n,)) * RPM_TO_RAD
    axis = spin_axis(vel, np.broadcast_to(np.asarray(tilt, float), (n,)))
    wind = np.zeros((n, 3)) if wind is None else np.broadcast_to(np.asarray(wind, float), (n, 3))
    cp_d = np.empty((n, 0)) if cp_d is None else np.asarray(cp_d, float)
    land_h = np.broadcast_to(np.asarray(land_h, float), (n,))

    out = {k: np.full(n, np.nan) for k in
           ["apex_t", "apex_d", "apex_l", "apex_h", "landing_t", "landing_d", "landing_l"]}
    out["landing_vel"] = np.full((n, 3), np.nan)
    out["landing_spin"] = np.full(n, np.nan)
    out["axis"] = axis
    for k in ["cp_t", "cp_l", "cp_h"]:
        out[k] = np.full(cp_d.shape, np.nan)
    path, times = [pos.copy()], [0.0]

    t = 0.0
    while t < t_max and np.isnan(out["cp_t"] if stop == "cps" else out["landing_t"]).any():
        a1, w1 = _accel(vel, om, axis, wind, aero)
        v2, o2 = vel + 0.5 * dt * a1, om + 0.5 * dt * w1
        a2, w2 = _accel(v2, o2, axis, wind, aero)
        v3, o3 = vel + 0.5 * dt * a2, om + 0.5 * dt * w2
        a3, w3 = _accel(v3, o3, axis, wind, aero)
        v4, o4 = vel + dt * a3, om + dt * w3
        a4, w4 = _accel(v4, o4, axis, wind, aero)
        new_pos = pos + dt / 6 * (vel + 2 * v2 + 2 * v3 + v4)
        new_vel = vel + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        new_om = om + dt / 6 * (w1 + 2 * w2 + 2 * w3 + w4)

        for j in range(cp_d.shape[1]):
            m = np.isnan(out["cp_t"][:, j]) & (pos[:, 0] < cp_d[:, j]) & (new_pos[:, 0] >= cp_d[:, j])
            if m.any():
                f = (cp_d[m, j] - pos[m, 0]) / (new_pos[m, 0] - pos[m, 0])
                out["cp_t"][m, j] = t + f * dt
                out["cp_l"][m, j] = pos[m, 1] + f * (new_pos[m, 1] - pos[m, 1])
                out["cp_h"][m, j] = pos[m, 2] + f * (new_pos[m, 2] - pos[m, 2])

        m = np.isnan(out["apex_t"]) & (vel[:, 2] > 0) & (new_vel[:, 2] <= 0)
        if m.any():
            f = vel[m, 2] / (vel[m, 2] - new_vel[m, 2])
            out["apex_t"][m] = t + f * dt
            for c, name in enumerate("dlh"):
                out[f"apex_{name}"][m] = pos[m, c] + f * (new_pos[m, c] - pos[m, c])

        m = (np.isnan(out["landing_t"]) & (pos[:, 2] > land_h) & (new_pos[:, 2] <= land_h)
             & (new_vel[:, 2] < 0))
        if m.any():
            f = (pos[m, 2] - land_h[m]) / (pos[m, 2] - new_pos[m, 2])
            out["landing_t"][m] = t + f * dt
            out["landing_d"][m] = pos[m, 0] + f * (new_pos[m, 0] - pos[m, 0])
            out["landing_l"][m] = pos[m, 1] + f * (new_pos[m, 1] - pos[m, 1])
            out["landing_vel"][m] = vel[m] + f[:, None] * (new_vel[m] - vel[m])
            out["landing_spin"][m] = (om[m] + f * (new_om[m] - om[m])) / RPM_TO_RAD

        pos, vel, om, t = new_pos, new_vel, new_om, t + dt
        if keep_path:
            path.append(pos.copy())
            times.append(t)

    if keep_path:
        out["path"], out["path_t"] = np.stack(path), np.array(times)
    return out
