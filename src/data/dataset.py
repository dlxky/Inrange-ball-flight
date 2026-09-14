"""Competition data: loading, range and per-shot coordinate frames, hitting blocks and submission format."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("INRANGE_DATA_DIR", ROOT / "inrange-competition"))

# Downrange heading recovered from the checkpoint lines: every cp_k crossing projects onto this
# direction at exactly the same value no matter which bay the shot was hit from.
HEADING = 0.4950901162353211                      # rad from +x (28.367 deg)
U = np.array([np.cos(HEADING), np.sin(HEADING)])  # downrange unit vector
V = np.array([-U[1], U[0]])                       # lateral unit vector, + = left of target line
CP_PLANES = 10.8621 + 15.0 * np.arange(4)         # range-frame downrange coordinate of the lines

POINTS = ["cp1", "cp2", "cp3", "cp4"]
TARGETS = ["launch_spin_rate", "apex_t", "apex_x", "apex_y", "apex_z",
           "landing_t", "landing_x", "landing_y", "landing_z"]
LOCAL_TARGETS = ["launch_spin_rate", "apex_t", "apex_d", "apex_l", "apex_h",
                 "landing_t", "landing_d", "landing_l"]


def load():
    train = add_local(pd.read_csv(DATA_DIR / "train.csv"))
    test = add_local(pd.read_csv(DATA_DIR / "test.csv"))
    add_blocks(train, test)
    return train, test


def add_local(df):
    """Add launch-relative downrange (d), lateral (l) and height (h) columns plus launch angles."""
    df = df.copy()
    lxy = df[["launch_x", "launch_y"]].to_numpy()
    for p in POINTS + ["apex", "landing"]:
        if f"{p}_x" not in df:
            continue
        rel = df[[f"{p}_x", f"{p}_y"]].to_numpy() - lxy
        df[f"{p}_d"] = rel @ U
        df[f"{p}_l"] = rel @ V
        df[f"{p}_h"] = df[f"{p}_z"] - df["launch_z"]
    vxy = df[["launch_vx", "launch_vy"]].to_numpy()
    df["vd"] = vxy @ U
    df["vl"] = vxy @ V
    df["vh"] = df["launch_vz"]
    df["speed"] = np.sqrt(df.vd**2 + df.vl**2 + df.vh**2)
    df["vla"] = np.degrees(np.arcsin(df.vh / df.speed))
    df["hla"] = np.degrees(np.arctan2(df.vl, df.vd))
    df["tee"] = df.launch_x.round(2).astype(str) + "_" + df.launch_y.round(2).astype(str)
    df["elevated"] = (df.launch_z > 1).astype(int)
    df["session"] = pd.to_datetime(df.launch_time, unit="s").dt.strftime("%Y-%m-%d")
    return df


def add_blocks(train, test, gap=900.0):
    """Label contiguous hitting blocks (no pause longer than `gap` s) on the joint timeline."""
    both = pd.concat([train[["track_id", "launch_time"]], test[["track_id", "launch_time"]]])
    both = both.sort_values("launch_time")
    both["block"] = (both.launch_time.diff() > gap).cumsum()
    block = both.set_index("track_id").block
    for df in (train, test):
        df["block"] = df.track_id.map(block).to_numpy()


def to_global(df, d, l, h):
    """Map launch-relative (d, l, h) back to range x, y, z for each row of df."""
    x = df.launch_x.to_numpy() + d * U[0] + l * V[0]
    y = df.launch_y.to_numpy() + d * U[1] + l * V[1]
    z = df.launch_z.to_numpy() + h
    return x, y, z


def to_submission(df, pred):
    """Convert a local-frame prediction frame (LOCAL_TARGETS columns) to the submission format."""
    out = pd.DataFrame({"track_id": df.track_id.to_numpy()})
    out["launch_spin_rate"] = pred["launch_spin_rate"].to_numpy()
    for p in ["apex", "landing"]:
        h = pred[f"{p}_h"].to_numpy() if f"{p}_h" in pred else np.zeros(len(df))
        x, y, z = to_global(df, pred[f"{p}_d"].to_numpy(), pred[f"{p}_l"].to_numpy(), h)
        out[f"{p}_t"] = pred[f"{p}_t"].to_numpy()
        out[f"{p}_x"], out[f"{p}_y"], out[f"{p}_z"] = x, y, z
    return out[["track_id"] + TARGETS]
