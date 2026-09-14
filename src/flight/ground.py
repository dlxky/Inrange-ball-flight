"""Bounce and roll after the ball reaches the ground.

Impacts follow the crater model of A. R. Penner, "The run of a golf ball", Can. J. Phys. 80 (2002):
the turf gives way and tilts the effective contact normal back against the direction of travel by an
angle that grows with impact speed and steepness; the normal rebound uses a speed-dependent
coefficient of restitution and the tangential impulse is capped by Coulomb friction, or ends in
rolling. Between impacts the ball hops ballistically with drag; once the rebound is negligible it rolls
to rest under rolling resistance plus a speed-proportional grass drag. Turf constants are generic
range-grass values, not measured at this range.
"""
import numpy as np

from flight.simulator import AREA, G, MASS, RADIUS

TURF = dict(mu=0.43, crater_deg=15.4, crater_speed=18.6, crater_angle=44.4,
            roll_resistance=0.3, grass_drag=0.8, min_rebound=1.2, hop_cd=0.25, rho=1.19)


def restitution(vn):
    """Normal coefficient of restitution against normal impact speed (m/s)."""
    return np.where(vn <= 20.0, 0.510 - 0.0375 * vn + 0.000903 * vn**2, 0.120)


def impact(vel, omega, turf=TURF):
    """One turf impact. vel (3,) m/s and omega (3,) rad/s in a z-up frame; returns both after impact."""
    horiz = np.hypot(vel[0], vel[1])
    angle = np.degrees(np.arctan2(-vel[2], horiz))
    crater = np.radians(turf["crater_deg"] * (np.linalg.norm(vel) / turf["crater_speed"])
                        * (angle / turf["crater_angle"]))
    heading = np.array([vel[0] / horiz, vel[1] / horiz, 0.0]) if horiz > 1e-9 else np.array([1.0, 0, 0])
    n = np.cos(crater) * np.array([0.0, 0.0, 1.0]) - np.sin(crater) * heading
    vn = vel @ n
    if vn >= 0:
        return vel, omega
    jn = (1 + restitution(-vn)) * -vn           # normal impulse per unit mass
    r_c = -RADIUS * n                            # contact point relative to the centre
    contact = vel + np.cross(omega, r_c)
    slip = contact - (contact @ n) * n
    s = np.linalg.norm(slip)
    jt = -min(turf["mu"] * jn, 2 / 7 * s) * slip / s if s > 1e-9 else np.zeros(3)
    return vel + jn * n + jt, omega + np.cross(r_c, jt) / (0.4 * RADIUS**2)


def bounce_and_roll(pos, vel, omega, turf=TURF, dt=0.005, max_impacts=12):
    """Follow the ball from its first ground contact at pos (ground height = pos[2]) until it stops.

    Returns times, path (T, 3), the impact points, and the rest position.
    """
    pos, vel, omega = np.array(pos, float), np.array(vel, float), np.array(omega, float)
    ground = pos[2]
    k = turf["rho"] * AREA / (2 * MASS) * turf["hop_cd"]
    t, times, path, impacts = 0.0, [0.0], [pos.copy()], [pos.copy()]
    for _ in range(max_impacts):
        vel, omega = impact(vel, omega, turf)
        if vel[2] < turf["min_rebound"]:
            break
        while True:  # a hop: gravity and drag, semi-implicit Euler
            vel = vel + dt * (-k * np.linalg.norm(vel) * vel - np.array([0.0, 0.0, G]))
            pos = pos + dt * vel
            t += dt
            landed = pos[2] <= ground and vel[2] < 0
            if landed:
                pos[2] = ground
            times.append(t)
            path.append(pos.copy())
            if landed:
                impacts.append(pos.copy())
                break

    # Roll: dv/dt = -(a + beta v), solved in closed form until the ball stops.
    roll = np.array([vel[0], vel[1], 0.0])
    speed = np.linalg.norm(roll)
    a, beta = turf["roll_resistance"] * G, turf["grass_drag"]
    if speed > 1e-6:
        direction = roll / speed
        stop = np.log1p(beta * speed / a) / beta
        for tau in np.linspace(0.0, stop, 60)[1:]:
            dist = (speed + a / beta) * (1 - np.exp(-beta * tau)) / beta - a / beta * tau
            times.append(t + tau)
            path.append(pos + direction * dist)
    return dict(t=np.array(times), path=np.array(path), impacts=np.array(impacts), rest=path[-1])
