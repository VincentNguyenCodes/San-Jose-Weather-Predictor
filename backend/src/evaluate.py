import csv, math, sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "weather" / "ml"))
from model import WeatherNet, build_features, HIST_YEARS, SEQ_DAYS

DATA_DIR   = Path(__file__).resolve().parents[1] / "data"
MODEL_PATH = Path(__file__).resolve().parents[1] / "weather/ml/model_weights.pth"

TEST_YEARS = [2023, 2024, 2025]


def load_all():
    all_data = {}
    for f in sorted(DATA_DIR.glob("SanJoseWeather*.csv")):
        with open(f, newline="") as fp:
            for row in csv.DictReader(fp):
                yr  = int(row["year"])
                doy = int(row["day_of_year"])
                tx, tn = row["tmax"], row["tmin"]
                if tx == "" or tn == "":
                    continue
                precip = float(row["precip"]) if row.get("precip", "") != "" else 0.0
                all_data.setdefault(yr, {})[doy] = (float(tx), float(tn), precip)
    return all_data


def make_dataset(all_data, use_years, target_years):
    features, targets, meta = [], [], []
    for target_year in target_years:
        if target_year not in all_data:
            continue
        past_years = [y for y in sorted(use_years) if y < target_year]
        for doy in sorted(all_data[target_year]):
            tmax_t, tmin_t, _ = all_data[target_year][doy]
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
            meta.append((target_year, doy, [(all_data[py][doy][0], all_data[py][doy][1])
                                             for py in past_years[-HIST_YEARS:]
                                             if doy in all_data.get(py, {})]))
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets,  dtype=torch.float32)
    return X, y, meta


def train_model(X, y, epochs=1000, lr=1e-3, patience=50, verbose=False):
    model   = WeatherNet()
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.HuberLoss()
    loader  = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

    best_loss    = float('inf')
    best_weights = None
    no_improve   = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        avg = total / len(loader)

        if avg < best_loss:
            best_loss    = avg
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1

        if verbose and epoch % 100 == 0:
            print(f"    epoch {epoch:4d}  loss={avg:.4f}")

        if no_improve >= patience:
            if verbose:
                print(f"    early stop at epoch {epoch}")
            break

    model.load_state_dict(best_weights)
    model.eval()
    return model


def metrics(preds, actuals):
    n = len(preds)
    mae_tx  = sum(abs(p[0] - a[0]) for p, a in zip(preds, actuals)) / n
    mae_tn  = sum(abs(p[1] - a[1]) for p, a in zip(preds, actuals)) / n
    rmse_tx = math.sqrt(sum((p[0]-a[0])**2 for p,a in zip(preds,actuals)) / n)
    rmse_tn = math.sqrt(sum((p[1]-a[1])**2 for p,a in zip(preds,actuals)) / n)
    return mae_tx, mae_tn, rmse_tx, rmse_tn


def predict_batch(model, X):
    with torch.no_grad():
        out = model(X)
    return [(round(r[0].item(), 1), round(r[1].item(), 1)) for r in out]


def baseline_preds(meta):
    preds = []
    for (yr, doy, same_day) in meta:
        avg_tx = sum(tx for tx, tn in same_day) / len(same_day)
        avg_tn = sum(tn for tx, tn in same_day) / len(same_day)
        preds.append((avg_tx, avg_tn))
    return preds


def per_year_metrics(preds, actuals, meta):
    by_year = defaultdict(lambda: {"preds": [], "actuals": []})
    for pred, actual, (yr, doy, _) in zip(preds, actuals, meta):
        by_year[yr]["preds"].append(pred)
        by_year[yr]["actuals"].append(actual)
    results = {}
    for yr in sorted(by_year):
        results[yr] = metrics(by_year[yr]["preds"], by_year[yr]["actuals"])
    return results


def worst_predictions(preds, actuals, meta, n=10):
    errors = []
    for pred, actual, (yr, doy, _) in zip(preds, actuals, meta):
        err = abs(pred[0] - actual[0]) + abs(pred[1] - actual[1])
        errors.append((err, yr, doy, pred, actual))
    return sorted(errors, reverse=True)[:n]


def divider(char="─", width=64):
    print(char * width)

def section(title):
    divider()
    print(f"  {title}")
    divider()


def main():
    all_data = load_all()
    years_available = sorted(all_data.keys())
    train_years = [y for y in years_available if y not in TEST_YEARS]

    print(f"\n  Data loaded: {years_available[0]}-{years_available[-1]}")
    print(f"  Train years: {train_years[0]}-{train_years[-1]}")
    print(f"  Test years:  {TEST_YEARS}\n")

    X_train, y_train, _    = make_dataset(all_data, train_years, train_years)
    X_test,  y_test,  meta = make_dataset(all_data, train_years, TEST_YEARS)
    actuals = [(float(y[0]), float(y[1])) for y in y_test]

    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples:     {len(X_test)}\n")

    section("Training hold-out model (1950-2022, v3 config)")
    holdout_model = train_model(X_train, y_train, epochs=1000, patience=50, verbose=True)

    prod_model = WeatherNet()
    prod_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    prod_model.eval()

    preds_holdout = predict_batch(holdout_model, X_test)
    preds_prod    = predict_batch(prod_model,    X_test)
    preds_base    = baseline_preds(meta)

    section("Overall accuracy on 2023-2025 test set")
    header = f"  {'Model':<36}  {'MAE High':>8}  {'MAE Low':>8}  {'RMSE High':>10}  {'RMSE Low':>9}"
    print(header)
    divider("-")

    for label, preds in [
        ("Hold-out model (1950-2022 train, v3)", preds_holdout),
        ("Production model (all years)",          preds_prod),
        ("Baseline (same-day avg, no NN)",        preds_base),
    ]:
        mx, mn, rx, rn = metrics(preds, actuals)
        print(f"  {label:<36}  {mx:>7.2f}F  {mn:>7.2f}F  {rx:>9.2f}F  {rn:>8.2f}F")

    section("Per-year breakdown - hold-out model")
    print(f"  {'Year':<6}  {'Samples':>7}  {'MAE High':>9}  {'MAE Low':>9}  {'RMSE High':>10}  {'RMSE Low':>9}")
    divider("-")
    by_year = per_year_metrics(preds_holdout, actuals, meta)
    by_year_counts = defaultdict(int)
    for (yr, doy, _) in meta:
        by_year_counts[yr] += 1
    for yr, (mx, mn, rx, rn) in by_year.items():
        print(f"  {yr:<6}  {by_year_counts[yr]:>7}  {mx:>8.2f}F  {mn:>8.2f}F  {rx:>9.2f}F  {rn:>8.2f}F")

    section("10 largest errors - hold-out model")
    print(f"  {'Year':>4}  {'DOY':>4}  {'Pred High':>9}  {'Act High':>8}  {'Pred Low':>9}  {'Act Low':>8}  {'Total Err':>9}")
    divider("-")
    from datetime import date, timedelta
    for err, yr, doy, pred, actual in worst_predictions(preds_holdout, actuals, meta):
        d = date(yr, 1, 1) + timedelta(days=doy - 1)
        date_str = d.strftime("%b %d")
        print(
            f"  {yr:>4}  {date_str:>6}  "
            f"{pred[0]:>8.1f}F  {actual[0]:>7.1f}F  "
            f"{pred[1]:>8.1f}F  {actual[1]:>7.1f}F  "
            f"{err:>8.1f}F"
        )

    section("Monthly MAE - hold-out model (averaged over 2023-2025)")
    from datetime import date as dt, timedelta as td
    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    by_month = defaultdict(lambda: {"preds": [], "actuals": []})
    for pred, actual, (yr, doy, _) in zip(preds_holdout, actuals, meta):
        d = dt(yr, 1, 1) + td(days=doy - 1)
        by_month[d.month]["preds"].append(pred)
        by_month[d.month]["actuals"].append(actual)
    print(f"  {'Month':<6}  {'Samples':>7}  {'MAE High':>9}  {'MAE Low':>9}")
    divider("-")
    for m in range(1, 13):
        if m not in by_month:
            continue
        mx, mn, _, _ = metrics(by_month[m]["preds"], by_month[m]["actuals"])
        n = len(by_month[m]["preds"])
        print(f"  {MONTHS[m-1]:<6}  {n:>7}  {mx:>8.2f}F  {mn:>8.2f}F")

    divider()
    print()


if __name__ == "__main__":
    main()
