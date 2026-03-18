import torch
import torch.nn as nn
import math

# Feature layout (see build_features() below):
#   - 7 same-day historical years × 2 (tmax, tmin), normalized  = 14
#   - 7 presence flags (1 if that year's data exists)            =  7
#   - 7 sequential prior days × 2 (tmax, tmin), normalized      = 14
#   - temperature deltas (tmax_delta, tmin_delta)                =  2
#   - 7-day rolling precip sum, normalized                       =  1
#   - cyclical day-of-year (sin, cos)                            =  2
# Total input size = 40

INPUT_SIZE = 40
HIST_YEARS = 7
SEQ_DAYS   = 7

TEMP_SCALE   = 100.0   # divide °F values by this to normalize
PRECIP_SCALE = 5.0     # divide 7-day precip sum by this to normalize


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
    recent_seq: list,      # list of (tmax, tmin) tuples, [D-1, D-2, ...D-7], len ≤ SEQ_DAYS
    day_of_year: int,
    precip_seq: list = (), # list of precip values [D-1, D-2, ...D-7], len ≤ SEQ_DAYS
) -> torch.Tensor:
    """
    Build the 40-element feature tensor.
    Missing values are zero-padded; presence flags encode which slots have data.
    Temperatures are normalized by dividing by TEMP_SCALE.
    """
    # Historical same-day slots (pad left with zeros if fewer than HIST_YEARS)
    hist_tmax = [0.0] * HIST_YEARS
    hist_tmin = [0.0] * HIST_YEARS
    flags      = [0.0] * HIST_YEARS

    for i, (tx, tn) in enumerate(same_day_hist[-HIST_YEARS:]):
        slot = HIST_YEARS - len(same_day_hist) + i
        hist_tmax[slot] = float(tx) / TEMP_SCALE
        hist_tmin[slot] = float(tn) / TEMP_SCALE
        flags[slot]      = 1.0

    # Sequential prior-day slots (normalized)
    seq_tmax = [0.0] * SEQ_DAYS
    seq_tmin = [0.0] * SEQ_DAYS
    for i, (tx, tn) in enumerate(recent_seq[:SEQ_DAYS]):
        seq_tmax[i] = float(tx) / TEMP_SCALE
        seq_tmin[i] = float(tn) / TEMP_SCALE

    # Temperature deltas: yesterday minus 2 days ago (captures warming/cooling trend)
    if len(recent_seq) >= 2:
        tmax_delta = (recent_seq[0][0] - recent_seq[1][0]) / TEMP_SCALE
        tmin_delta = (recent_seq[0][1] - recent_seq[1][1]) / TEMP_SCALE
    else:
        tmax_delta = 0.0
        tmin_delta = 0.0

    # 7-day rolling precipitation sum (normalized)
    rolling_precip = sum(float(p) for p in precip_seq[:SEQ_DAYS]) / PRECIP_SCALE

    # Cyclical day-of-year
    angle = 2 * math.pi * day_of_year / 365.0
    cyclic = [math.sin(angle), math.cos(angle)]

    vec = (
        [val for pair in zip(hist_tmax, hist_tmin) for val in pair]  # 14
        + flags                                                         #  7
        + [val for pair in zip(seq_tmax, seq_tmin) for val in pair]   # 14
        + [tmax_delta, tmin_delta]                                      #  2
        + [rolling_precip]                                              #  1
        + cyclic                                                        #  2
    )
    return torch.tensor(vec, dtype=torch.float32)
