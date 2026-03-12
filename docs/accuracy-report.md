# Model Accuracy Report

Evaluation methodology: **hold-out validation**.
The model was retrained exclusively on 2015–2022 data, then tested on the
completely unseen 2023–2025 data (1,096 samples across three full years).
A naive baseline (plain historical same-day average, no neural network) is
included to show the value added by the model.

---

## Overall Accuracy — 2023–2025 Test Set

| Model | MAE High | MAE Low | RMSE High | RMSE Low |
|---|---|---|---|---|
| **Hold-out model** (trained 2015–2022) | **4.90°F** | **3.20°F** | 6.23°F | 4.17°F |
| Production model (trained on all years) | 3.22°F | 2.36°F | 4.14°F | 2.96°F |
| Baseline — same-day average, no NN | 5.42°F | 3.87°F | 6.93°F | 4.98°F |

The neural network outperforms the plain historical average by **~0.5–0.7°F MAE**
on unseen years, demonstrating genuine learned patterns beyond seasonal averages.

---

## Per-Year Breakdown (Hold-out Model)

| Year | Samples | MAE High | MAE Low | RMSE High | RMSE Low |
|---|---|---|---|---|---|
| 2023 | 365 | 5.01°F | 3.33°F | 6.32°F | 4.29°F |
| 2024 | 366 | 5.20°F | 3.29°F | 6.54°F | 4.23°F |
| 2025 | 365 | 4.50°F | 2.98°F | 5.81°F | 4.00°F |

---

## Monthly MAE (Averaged Over 2023–2025)

| Month | Samples | MAE High | MAE Low |
|---|---|---|---|
| Jan | 93 | 4.26°F | 3.64°F |
| Feb | 85 | 4.82°F | 4.38°F |
| Mar | 93 | 4.46°F | 3.92°F |
| Apr | 90 | 5.50°F | 3.06°F |
| May | 93 | 5.58°F | 2.35°F |
| Jun | 90 | 5.04°F | 2.07°F |
| Jul | 93 | 4.65°F | 2.49°F |
| Aug | 93 | 4.68°F | 2.12°F |
| Sep | 90 | 5.59°F | 2.74°F |
| Oct | 93 | 6.09°F | 3.84°F |
| Nov | 90 | 4.39°F | 3.71°F |
| Dec | 93 | 3.81°F | 4.15°F |

**October is the hardest month** (6.09°F MAE on highs) due to San Jose's
unpredictable fall heat waves — anomalous events that historical averages
cannot reliably anticipate.

---

## Largest Individual Errors (Hold-out Model)

| Date | Predicted High | Actual High | Predicted Low | Actual Low | Total Error |
|---|---|---|---|---|---|
| Jan 2, 2023 | 71.6°F | 52.0°F | 55.6°F | 42.0°F | 33.2°F |
| Oct 6, 2023 | 84.6°F | 98.0°F | 57.5°F | 72.0°F | 27.9°F |
| Oct 5, 2024 | 85.8°F | 104.0°F | 58.1°F | 67.0°F | 27.1°F |
| Mar 28, 2023 | 74.7°F | 54.0°F | 48.4°F | 42.0°F | 27.1°F |
| Oct 2, 2024 | 88.5°F | 106.0°F | 66.4°F | 76.0°F | 27.1°F |

Large errors occur almost exclusively during **unseasonable heat events** and
**atypical cold snaps** — exactly the scenarios where any historical-based
model will struggle.

---

## How to Reproduce

```bash
python src/evaluate.py
```
