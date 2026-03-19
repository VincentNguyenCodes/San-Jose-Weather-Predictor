import torch
import torch.nn as nn
import math

INPUT_SIZE = 40
HIST_YEARS = 7
SEQ_DAYS   = 7

TEMP_SCALE   = 100.0
PRECIP_SCALE = 5.0

SEQ_WEIGHTS = [(SEQ_DAYS - i) / SEQ_DAYS for i in range(SEQ_DAYS)]


def WeatherNet():
    return nn.Sequential(
        nn.Linear(INPUT_SIZE, 128), nn.ReLU(),
        nn.Linear(128, 256),        nn.ReLU(),
        nn.Linear(256, 128),        nn.ReLU(),
        nn.Linear(128, 64),         nn.ReLU(),
        nn.Linear(64, 2),
    )


def build_features(
    same_day_hist: list,
    recent_seq: list,
    day_of_year: int,
    precip_seq: list = (),
) -> torch.Tensor:
    hist_tmax = [0.0] * HIST_YEARS
    hist_tmin = [0.0] * HIST_YEARS
    flags      = [0.0] * HIST_YEARS

    for i, (tx, tn) in enumerate(same_day_hist[-HIST_YEARS:]):
        slot = HIST_YEARS - len(same_day_hist) + i
        hist_tmax[slot] = float(tx) / TEMP_SCALE
        hist_tmin[slot] = float(tn) / TEMP_SCALE
        flags[slot]      = 1.0

    seq_tmax = [0.0] * SEQ_DAYS
    seq_tmin = [0.0] * SEQ_DAYS
    for i, (tx, tn) in enumerate(recent_seq[:SEQ_DAYS]):
        seq_tmax[i] = float(tx) / TEMP_SCALE * SEQ_WEIGHTS[i]
        seq_tmin[i] = float(tn) / TEMP_SCALE * SEQ_WEIGHTS[i]

    if len(recent_seq) >= 2:
        tmax_delta = (recent_seq[0][0] - recent_seq[1][0]) / TEMP_SCALE
        tmin_delta = (recent_seq[0][1] - recent_seq[1][1]) / TEMP_SCALE
    else:
        tmax_delta = 0.0
        tmin_delta = 0.0

    rolling_precip = sum(float(p) * SEQ_WEIGHTS[i] for i, p in enumerate(precip_seq[:SEQ_DAYS])) / PRECIP_SCALE

    angle = 2 * math.pi * day_of_year / 365.0
    cyclic = [math.sin(angle), math.cos(angle)]

    vec = (
        [val for pair in zip(hist_tmax, hist_tmin) for val in pair]
        + flags
        + [val for pair in zip(seq_tmax, seq_tmin) for val in pair]
        + [tmax_delta, tmin_delta]
        + [rolling_precip]
        + cyclic
    )
    return torch.tensor(vec, dtype=torch.float32)
