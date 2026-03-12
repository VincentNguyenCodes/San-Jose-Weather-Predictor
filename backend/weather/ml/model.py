import torch
import torch.nn as nn
import math

# Feature layout (see build_features() below):
#   - 5 same-day historical years × 2 (tmax, tmin) = 10
#   - 5 presence flags (1 if that year's data exists)   =  5
#   - 3 sequential prior days × 2 (tmax, tmin)          =  6
#   - cyclical day-of-year (sin, cos)                   =  2
# Total input size = 23
INPUT_SIZE = 23
HIST_YEARS = 5
SEQ_DAYS   = 3


def WeatherNet():
    return nn.Sequential(
        nn.Linear(INPUT_SIZE, 128), nn.ReLU(),
        nn.Linear(128, 256),        nn.ReLU(),
        nn.Linear(256, 128),        nn.ReLU(),
        nn.Linear(128, 64),         nn.ReLU(),
        nn.Linear(64, 2),
    )


def build_features(
    same_day_hist: list,   # list of (tmax, tmin) tuples, oldest→newest, len ≤ HIST_YEARS
    recent_seq: list,      # list of (tmax, tmin) tuples, [D-1, D-2, D-3], len ≤ SEQ_DAYS
    day_of_year: int,
) -> torch.Tensor:
    """
    Build the 23-element feature tensor.
    Missing values are zero-padded; presence flags encode which slots have data.
    """
    # Historical same-day slots (pad left with zeros if fewer than HIST_YEARS)
    hist_tmax = [0.0] * HIST_YEARS
    hist_tmin = [0.0] * HIST_YEARS
    flags      = [0.0] * HIST_YEARS

    for i, (tx, tn) in enumerate(same_day_hist[-HIST_YEARS:]):
        slot = HIST_YEARS - len(same_day_hist) + i
        hist_tmax[slot] = float(tx)
        hist_tmin[slot] = float(tn)
        flags[slot]      = 1.0

    # Sequential prior-day slots
    seq_tmax = [0.0] * SEQ_DAYS
    seq_tmin = [0.0] * SEQ_DAYS
    for i, (tx, tn) in enumerate(recent_seq[:SEQ_DAYS]):
        seq_tmax[i] = float(tx)
        seq_tmin[i] = float(tn)

    # Cyclical day-of-year
    angle = 2 * math.pi * day_of_year / 365.0
    cyclic = [math.sin(angle), math.cos(angle)]

    vec = (
        [val for pair in zip(hist_tmax, hist_tmin) for val in pair]  # 10
        + flags                                                         #  5
        + [val for pair in zip(seq_tmax, seq_tmin) for val in pair]   #  6
        + cyclic                                                        #  2
    )
    return torch.tensor(vec, dtype=torch.float32)
