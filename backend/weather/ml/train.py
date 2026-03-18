"""
Train WeatherNet on historical San Jose weather data and save weights.

Features per sample:
  - Same-day temps (tmax, tmin) from up to 7 prior years, normalized  [14 values]
  - Presence flags for each historical year slot                       [ 7 values]
  - Sequential temps from the 7 days immediately before, normalized   [14 values]
  - Temperature deltas (tmax_delta, tmin_delta)                       [ 2 values]
  - 7-day rolling precipitation sum, normalized                       [ 1 value ]
  - Cyclical day-of-year encoding (sin, cos)                          [ 2 values]
  Total: 40 input features → 2 outputs (tmax, tmin)

Usage:
    python backend/weather/ml/train.py
    python backend/weather/ml/train.py --data-dir data/ --epochs 1000
"""

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import WeatherNet, build_features, HIST_YEARS, SEQ_DAYS

DEFAULT_DATA_DIR  = Path(__file__).resolve().parents[2] / "data"
DEFAULT_MODEL_OUT = Path(__file__).resolve().parent / "model_weights.pth"


def load_data(data_dir: Path) -> dict:
    """Return {year: {day_of_year: (tmax, tmin, precip)}}"""
    all_data: dict = {}
    for csv_file in sorted(data_dir.glob("SanJoseWeather*.csv")):
        with open(csv_file, newline="") as f:
            for row in csv.DictReader(f):
                yr  = int(row["year"])
                doy = int(row["day_of_year"])
                tx, tn = row["tmax"], row["tmin"]
                if tx == "" or tn == "":
                    continue
                precip = float(row["precip"]) if row.get("precip", "") != "" else 0.0
                all_data.setdefault(yr, {})[doy] = (float(tx), float(tn), precip)
    return all_data


def build_dataset(all_data: dict):
    years = sorted(all_data.keys())
    features, targets = [], []

    for y_idx, target_year in enumerate(years):
        if y_idx == 0:
            continue  # need at least one historical year
        past_years = years[:y_idx]

        for doy, (tmax_t, tmin_t, _) in sorted(all_data[target_year].items()):
            same_day = [
                (all_data[py][doy][0], all_data[py][doy][1])
                for py in past_years[-HIST_YEARS:]
                if doy in all_data.get(py, {})
            ]
            if not same_day:
                continue

            recent = []
            precip_seq = []
            for off in range(1, SEQ_DAYS + 1):
                prev_doy = doy - off
                if prev_doy >= 1 and prev_doy in all_data[target_year]:
                    tx, tn, pr = all_data[target_year][prev_doy]
                    recent.append((tx, tn))
                    precip_seq.append(pr)

            features.append(build_features(same_day, recent, doy, precip_seq).tolist())
            targets.append([tmax_t, tmin_t])

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets,  dtype=torch.float32)
    return X, y


def train(data_dir: Path, output_path: Path, epochs: int, lr: float):
    print(f"Loading data from {data_dir} ...")
    all_data = load_data(data_dir)
    years = sorted(all_data.keys())
    print(f"  Years: {years[0]}–{years[-1]}")

    X, y = build_dataset(all_data)
    print(f"  Training samples: {len(X)}")

    loader  = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
    model   = WeatherNet()
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.HuberLoss()

    print(f"Training for {epochs} epochs ...")
    for epoch in range(1, epochs + 1):
        total = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if epoch % 200 == 0:
            print(f"  Epoch {epoch:5d}  loss={total / len(loader):.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved model weights → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the WeatherNet model")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output",   type=Path, default=DEFAULT_MODEL_OUT)
    parser.add_argument("--epochs",   type=int,  default=1000)
    parser.add_argument("--lr",       type=float, default=1e-3)
    args = parser.parse_args()
    train(args.data_dir, args.output, args.epochs, args.lr)
