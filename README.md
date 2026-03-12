# San Jose Weather Predictor

A full-stack weather prediction application that uses a PyTorch neural network trained on 11 years of historical San Jose, CA temperature data (2015–2025) to forecast high and low temperatures for the upcoming week and any arbitrary future date.

---

## Features

- **7-day forecast** — predicted daily high/low temperatures starting from today
- **Date lookup** — predict the high/low for any date by typing it in
- **Transparent predictions** — each forecast shows which historical years it was based on
- **iOS-inspired UI** — glassmorphism weather card built in React
- **REST API** — Django backend exposable to any client

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                        │
│         Forecast Tab  │  Predict a Date Tab             │
└──────────────┬──────────────────────────┬───────────────┘
               │ HTTP                     │ HTTP
               ▼                          ▼
┌─────────────────────────────────────────────────────────┐
│               Django REST API (port 8000)                │
│    GET /api/forecast/     GET /api/predict/?date=...     │
└──────────────────────────┬──────────────────────────────┘
                           │
               ┌───────────▼───────────┐
               │     WeatherNet (MLP)  │
               │   23 inputs → 2 out   │
               │   (tmax, tmin in °F)  │
               └───────────┬───────────┘
                           │
               ┌───────────▼───────────┐
               │   data/ (CSV files)   │
               │   2015 – 2025         │
               │   San Jose, CA        │
               └───────────────────────┘
```

---

## Neural Network Model

**Architecture:** 5-layer MLP (23 → 128 → 256 → 128 → 64 → 2)

**Input features (23 total):**
| Feature Group | Size | Description |
|---|---|---|
| Same-day historical temps | 10 | tmax + tmin for the same calendar date in each of the past 5 years |
| Presence flags | 5 | 1 if historical data exists for that year slot, 0 otherwise |
| Sequential prior days | 6 | tmax + tmin from the 3 days immediately preceding the target date |
| Cyclical day-of-year | 2 | sin and cos encoding of day-of-year (captures seasonal patterns) |

**Output:** tmax, tmin (°F) for the target date

**Training:** Adam optimizer, MSE loss, 1,000 epochs, batch size 64

**Hold-out accuracy (tested on 2023–2025, never seen during training):**
- High temp MAE: **4.90°F** vs baseline 5.42°F
- Low temp MAE: **3.20°F** vs baseline 3.87°F

See [`docs/accuracy-report.md`](docs/accuracy-report.md) for the full evaluation.

---

## Project Structure

```
.
├── backend/                    # Django REST API
│   ├── weather/
│   │   ├── views.py            # /api/forecast/ and /api/predict/ endpoints
│   │   ├── urls.py
│   │   └── ml/
│   │       ├── model.py        # WeatherNet definition + feature builder
│   │       ├── train.py        # Training script
│   │       └── model_weights.pth
│   ├── weather_project/
│   │   └── settings.py
│   └── manage.py
├── frontend/                   # React app
│   └── src/
│       └── components/
│           ├── WeatherApp.js   # Forecast + Predict tabs
│           └── WeatherApp.css
├── data/                       # Historical CSV files (2015–2025)
│   └── SanJoseWeather{year}.csv
├── src/
│   ├── evaluate.py             # Hold-out accuracy evaluation
│   ├── noaa_fetcher.py         # Optional: fetch data via NOAA CDO API
│   └── csv-reader.py           # CSV utility functions
├── docs/
│   └── accuracy-report.md
├── requirements.txt
└── README.md
```

---

## Setup & Running

### Prerequisites

- Python 3.9+
- Node.js 18+

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Django backend

```bash
cd backend
python manage.py migrate
python manage.py runserver 8000
```

API will be available at `http://127.0.0.1:8000/api/`

### 3. Start the React frontend

```bash
cd frontend
npm install
npm start
```

App will open at `http://localhost:3000`

---

## API Reference

### `GET /api/forecast/`

Returns today + 7-day predicted forecast.

**Response:**
```json
{
  "location": "San Jose, CA",
  "forecast": [
    {
      "offset": 0,
      "date": "2026-03-12",
      "label": "Today",
      "short_date": "Mar 12",
      "tmax": 65,
      "tmin": 48,
      "based_on_years": [2021, 2022, 2023, 2024, 2025]
    }
  ]
}
```

### `GET /api/predict/?date=YYYY-MM-DD`

Returns a prediction for any specific date.

**Example:** `GET /api/predict/?date=2027-06-15`

**Response:**
```json
{
  "date": "2027-06-15",
  "label": "Tuesday",
  "short_date": "Jun 15, 2027",
  "tmax": 81,
  "tmin": 57,
  "based_on_years": [2021, 2022, 2023, 2024, 2025]
}
```

---

## Retrain the Model

```bash
cd backend/weather/ml
python train.py
# Options:
python train.py --data-dir ../../data --epochs 1000
```

## Evaluate Accuracy

```bash
python src/evaluate.py
```

Trains a fresh hold-out model on 2015–2022, evaluates on 2023–2025, and prints MAE, RMSE, per-year breakdowns, worst predictions, and monthly error breakdown.

## Fetch New Data

Data is sourced from the [Open-Meteo Historical API](https://open-meteo.com/) (free, no token required) for San Jose International Airport (37.3622°N, 121.9289°W).

Optionally, use the NOAA CDO API fetcher (requires a [free token](https://www.ncdc.noaa.gov/cdo-web/token)):

```bash
export NOAA_CDO_TOKEN=your_token
python src/noaa_fetcher.py --year 2026
```

---

## Data Format

Each CSV in `data/` follows this schema:

| Column | Type | Description |
|---|---|---|
| `year` | int | Calendar year |
| `month` | int | Month (1–12) |
| `day` | int | Day of month |
| `day_of_year` | int | Day of year (1–366) |
| `tmax` | int | Daily high temperature (°F) |
| `tmin` | int | Daily low temperature (°F) |
| `precip` | float | Precipitation (inches) |
| `rained` | int | 1 if precip > 0, else 0 |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, CSS (glassmorphism) |
| Backend | Python 3.9, Django 4.2, Django REST Framework |
| ML | PyTorch 2.8 |
| Data | Open-Meteo Historical API, NOAA CDO API |
| Database | SQLite (Django default) |
