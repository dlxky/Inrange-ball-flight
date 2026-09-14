"""Error components in competition units, and a proxy for the hidden composite score."""
import numpy as np
import pandas as pd

from data.dataset import LOCAL_TARGETS

# Proxy for the hidden composite metric: landing heaviest, then apex, then times, spin least.
WEIGHTS = {"landing": 0.40, "apex": 0.25, "apex_t": 0.125, "landing_t": 0.125, "spin": 0.10}


def errors(truth, pred):
    """Per-shot error components in competition units; distances are rotation-invariant."""
    e = pd.DataFrame(index=truth.index)
    for p in ["landing", "apex"]:
        ph = pred[f"{p}_h"].to_numpy() if f"{p}_h" in pred else 0.0
        e[p] = np.sqrt((truth[f"{p}_d"].to_numpy() - pred[f"{p}_d"].to_numpy()) ** 2
                       + (truth[f"{p}_l"].to_numpy() - pred[f"{p}_l"].to_numpy()) ** 2
                       + (truth[f"{p}_h"].to_numpy() - ph) ** 2)
    e["apex_t"] = np.abs(truth.apex_t.to_numpy() - pred.apex_t.to_numpy())
    e["landing_t"] = np.abs(truth.landing_t.to_numpy() - pred.landing_t.to_numpy())
    e["spin"] = np.abs(truth.launch_spin_rate.to_numpy() - pred.launch_spin_rate.to_numpy())
    return e


def mean_scales(train):
    """Error of the train-mean predictor for each component, used to normalise the composite."""
    mean = pd.DataFrame({c: np.full(len(train), train[c].mean()) for c in LOCAL_TARGETS})
    mean["landing_h"] = 0.0
    return errors(train, mean).mean()


def summarise(err, scales):
    row = err.mean()
    row["composite"] = sum(w * row[k] / scales[k] for k, w in WEIGHTS.items())
    return row
