# Model Accuracy Report

Evaluation methodology: **hold-out validation**.
The model was retrained exclusively on 2015–2022 data, then tested on the
completely unseen 2023–2025 data (1,096 samples across three full years).
A naive baseline (plain historical same-day average, no neural network) is
included to show the value added by the model.

---

## Model History

### v1 — Original Model

**Input features (23 total):**
| Feature Group | Size | Description |
|---|---|---|
| Same-day historical temps | 10 | Raw tmax + tmin for the same calendar date across the past 5 years |
| Presence flags | 5 | 1 if historical data exists for that year slot, 0 otherwise |
| Sequential prior days | 6 | tmax + tmin from the 3 days immediately preceding the target date |
| Cyclical day-of-year | 2 | sin/cos encoding of day-of-year |

**Architecture:** 5-layer MLP (23 → 128 → 256 → 128 → 64 → 2)
**Loss:** MSE
**Historical years used:** 5
**Sequential days used:** 3

**Known weaknesses:**
- Raw °F values (30–105) mixed with normalized sin/cos features (−1 to 1) — large input scale imbalance hurt gradient flow
- Only 3 prior days of context — too short to capture weather momentum
- Only 5 years of same-day history — leaving data on the table
- Precipitation data available in CSVs but completely ignored
- No trend signal — model couldn't tell if temperatures were rising or falling

---

### v2 — Improved Model

**What changed and why:**

**1. Input normalization (high impact)**
All temperature values are now divided by 100 before being fed into the network. Previously, raw °F values (30–105) sat alongside cyclical sin/cos features clamped to [−1, 1]. This 100× scale difference caused the optimizer to over-weight temperature features while the cyclical seasonal signal was effectively drowned out. Normalizing brings all features into a comparable range and stabilizes training.

**2. Extended historical years: 5 → 7**
The dataset covers 2015–2025 (10 years). The original model only looked back 5 years on the same calendar date. Increasing to 7 gives the model more data points to estimate typical conditions for each day of year, reducing variance in the historical signal.

**3. Extended sequential window: 3 → 7 days**
Weather systems have momentum that extends well beyond 3 days. A cold front, heat wave, or marine layer typically persists for a week or more. Extending the sequential window from 3 to 7 days lets the model see the full arc of the current weather pattern rather than just the last 72 hours.

**4. Temperature delta features (new)**
Two new features were added: `tmax_delta` and `tmin_delta`, defined as yesterday's temperature minus the day before's temperature (normalized). This gives the model an explicit signal for whether temperatures are trending up or down — something that couldn't be inferred from raw day values alone.

**5. Precipitation features (new)**
The CSV data includes daily precipitation (inches) but it was never used. A 7-day rolling precipitation sum (normalized) is now included as a feature. Wet stretches tend to suppress high temps and moderate lows; dry spells allow temperatures to swing further. This feature particularly improved low temperature accuracy.

**6. Huber loss (replaces MSE)**
MSE penalizes large errors quadratically, which causes the model to over-prioritize rare extreme events (heat waves, cold snaps) at the expense of normal-day accuracy. Huber loss behaves like MSE for small errors but clips large errors linearly, making the model more robust on typical days while not completely ignoring outliers.

**Updated input features (40 total):**
| Feature Group | Size | Description |
|---|---|---|
| Same-day historical temps (normalized) | 14 | tmax + tmin for the same calendar date across the past 7 years, divided by 100 |
| Presence flags | 7 | 1 if historical data exists for that year slot, 0 otherwise |
| Sequential prior days (normalized) | 14 | tmax + tmin from the 7 days preceding the target date, divided by 100 |
| Temperature deltas | 2 | (yesterday − 2 days ago) for tmax and tmin, normalized |
| 7-day rolling precipitation | 1 | Sum of prior 7 days of precipitation, normalized |
| Cyclical day-of-year | 2 | sin/cos encoding of day-of-year |

**Architecture:** 5-layer MLP (40 → 128 → 256 → 128 → 64 → 2) — same depth, wider input
**Loss:** Huber
**Historical years used:** 7
**Sequential days used:** 7

---

## Before vs. After — Hold-out Model (2015–2022 train → 2023–2025 test)

| Metric | v1 (Original) | v2 (Improved) | Change |
|---|---|---|---|
| MAE High | 4.61°F | **4.53°F** | −0.08°F |
| MAE Low | 3.45°F | **2.94°F** | −0.51°F ✓ |
| RMSE High | 6.00°F | **5.81°F** | −0.19°F |
| RMSE Low | 4.38°F | **3.89°F** | −0.49°F ✓ |

Low temperature accuracy improved the most (+0.5°F MAE), driven primarily by the addition of precipitation features — overnight lows correlate strongly with wet vs. dry conditions that the original model had no visibility into.

---

## Overall Accuracy — 2023–2025 Test Set (v2)

| Model | MAE High | MAE Low | RMSE High | RMSE Low |
|---|---|---|---|---|
| **Hold-out model** (trained 2015–2022) | **4.53°F** | **2.94°F** | 5.81°F | 3.89°F |
| Production model (trained on all years) | 2.87°F | 1.96°F | 4.06°F | 2.72°F |
| Baseline — same-day average, no NN | 5.33°F | 3.87°F | 6.81°F | 4.97°F |

The neural network outperforms the plain historical average by **~0.8–0.9°F MAE** on unseen years — an improvement over the v1 gap of ~0.5–0.7°F.

---

## Per-Year Breakdown — Hold-out Model (v2)

| Year | Samples | MAE High | MAE Low | RMSE High | RMSE Low |
|---|---|---|---|---|---|
| 2023 | 365 | 4.44°F | 2.96°F | 5.74°F | 3.85°F |
| 2024 | 366 | 4.77°F | 3.12°F | 6.22°F | 4.03°F |
| 2025 | 365 | 4.37°F | 2.74°F | 5.43°F | 3.78°F |

---

## Monthly MAE — Hold-out Model (v2, Averaged Over 2023–2025)

| Month | Samples | MAE High | MAE Low |
|---|---|---|---|
| Jan | 93 | 3.68°F | 4.24°F |
| Feb | 85 | 4.75°F | 4.49°F |
| Mar | 93 | 5.57°F | 3.19°F |
| Apr | 90 | 4.81°F | 2.77°F |
| May | 93 | 5.22°F | 2.22°F |
| Jun | 90 | 3.79°F | 1.64°F |
| Jul | 93 | 4.85°F | 1.98°F |
| Aug | 93 | 3.68°F | 1.72°F |
| Sep | 90 | 4.48°F | 2.60°F |
| Oct | 93 | 5.77°F | 3.64°F |
| Nov | 90 | 4.30°F | 3.42°F |
| Dec | 93 | 3.42°F | 3.49°F |

**October remains the hardest month** (5.77°F MAE on highs) due to San Jose's
unpredictable fall heat waves driven by Santa Ana wind events — anomalous spikes
that no historical-based model can reliably anticipate. The worst individual errors
are now concentrated almost exclusively in these October heatwave windows rather
than scattered across the calendar, which indicates the improved model has learned
normal patterns well and only struggles on genuine anomalies.

---

## Largest Individual Errors — Hold-out Model (v2)

| Date | Predicted High | Actual High | Predicted Low | Actual Low | Total Error |
|---|---|---|---|---|---|
| Oct 2, 2024 | 88.2°F | 106.0°F | 62.1°F | 76.0°F | 31.7°F |
| Oct 3, 2024 | 81.3°F | 101.0°F | 63.1°F | 73.0°F | 29.6°F |
| Oct 6, 2023 | 82.6°F | 98.0°F | 58.4°F | 72.0°F | 29.0°F |
| Oct 19, 2023 | 79.9°F | 98.0°F | 56.2°F | 67.0°F | 28.9°F |
| Jul 2, 2024 | 82.0°F | 101.0°F | 59.1°F | 65.0°F | 24.9°F |

All top errors now fall on extreme heat events (October heatwaves and unusual July spikes). The v1 model had large errors scattered across January, March, and May as well — the v2 model has cleaned those up, leaving only genuinely unpredictable anomalies in the worst-error list.

---

## How to Reproduce

```bash
# From the backend/ directory
python src/evaluate.py
```
