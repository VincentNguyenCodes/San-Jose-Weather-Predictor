import csv
from datetime import date, timedelta
from pathlib import Path

import torch
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings

from .ml.model import WeatherNet, build_features, SEQ_DAYS

HIST_YEARS = 7

_model    = None
_all_data = None   # {year: {doy: (tmax, tmin, precip)}}


def _load_data():
    global _all_data
    if _all_data is not None:
        return
    data_dir = Path(settings.DATA_DIR)
    _all_data = {}
    for f in sorted(data_dir.glob('SanJoseWeather*.csv')):
        with open(f, newline='') as fp:
            for row in csv.DictReader(fp):
                yr  = int(row['year'])
                doy = int(row['day_of_year'])
                tx, tn = row['tmax'], row['tmin']
                if tx == '' or tn == '':
                    continue
                precip = float(row['precip']) if row.get('precip', '') != '' else 0.0
                _all_data.setdefault(yr, {})[doy] = (float(tx), float(tn), precip)


def _get_model():
    global _model
    if _model is None:
        m = WeatherNet()
        weights = Path(settings.MODEL_PATH)
        if not weights.exists():
            raise FileNotFoundError(f"Model weights not found at {weights}. Run training first.")
        m.load_state_dict(torch.load(weights, map_location='cpu'))
        m.eval()
        _model = m
    return _model


WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MONTHS   = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def _predict_day(model, target_date, predicted_cache):
    """
    Predict tmax/tmin for target_date.
    predicted_cache: {date: (tmax, tmin, precip)} — filled in as we forecast forward.
    """
    _load_data()
    doy = target_date.timetuple().tm_yday
    yr  = target_date.year

    # Same-day historical: look in past HIST_YEARS years from CSV data
    same_day = []
    for past_yr in sorted(_all_data.keys()):
        if past_yr >= yr:
            continue
        if doy in _all_data.get(past_yr, {}):
            entry = _all_data[past_yr][doy]
            same_day.append((entry[0], entry[1]))
    same_day = same_day[-HIST_YEARS:]

    # Sequential: last SEQ_DAYS days before target_date
    recent = []
    precip_seq = []
    for offset in range(1, SEQ_DAYS + 1):
        prev = target_date - timedelta(days=offset)
        prev_doy = prev.timetuple().tm_yday
        prev_yr  = prev.year
        if prev in predicted_cache:
            entry = predicted_cache[prev]
            recent.append((entry[0], entry[1]))
            precip_seq.append(entry[2] if len(entry) > 2 else 0.0)
        elif prev_yr in _all_data and prev_doy in _all_data[prev_yr]:
            entry = _all_data[prev_yr][prev_doy]
            recent.append((entry[0], entry[1]))
            precip_seq.append(entry[2])

    feats = build_features(same_day, recent, doy, precip_seq).unsqueeze(0)
    with torch.no_grad():
        pred = _get_model()(feats).squeeze(0)
    return round(pred[0].item()), round(pred[1].item())


@api_view(['GET'])
def predict(request):
    """
    GET /api/predict/?date=YYYY-MM-DD
    Returns a single-day prediction for any arbitrary date.
    """
    date_str = request.query_params.get('date', '').strip()
    if not date_str:
        return Response({'error': 'Provide a ?date=YYYY-MM-DD query parameter.'}, status=400)
    try:
        target = date.fromisoformat(date_str)
    except ValueError:
        return Response({'error': f'Invalid date format: {date_str!r}. Use YYYY-MM-DD.'}, status=400)

    _load_data()

    # Build a sequential cache up to the day before target using CSV data
    cache = {}
    yr = target.year
    if yr in _all_data:
        base = date(yr, 1, 1)
        for doy, entry in _all_data[yr].items():
            d = base + timedelta(days=doy - 1)
            if d < target:
                cache[d] = entry

    tmax, tmin = _predict_day(_get_model(), target, cache)
    doy = target.timetuple().tm_yday
    hist_years = [
        y for y in sorted(_all_data.keys())
        if y < target.year and doy in _all_data.get(y, {})
    ][-HIST_YEARS:]

    return Response({
        'date':            target.isoformat(),
        'label':           WEEKDAYS[target.weekday()],
        'short_date':      f"{MONTHS[target.month - 1]} {target.day}, {target.year}",
        'tmax':            tmax,
        'tmin':            tmin,
        'based_on_years':  hist_years,
    })


@api_view(['GET'])
def forecast(request):
    _load_data()
    today = date.today()
    predicted_cache = {}

    # Pre-load any available data for the current year from CSVs
    yr = today.year
    if yr in _all_data:
        base = date(yr, 1, 1)
        for doy, entry in _all_data[yr].items():
            d = base + timedelta(days=doy - 1)
            predicted_cache[d] = entry

    results = []
    for offset in range(8):  # today + 7 days
        d = today + timedelta(days=offset)
        tmax, tmin = _predict_day(_get_model(), d, predicted_cache)
        predicted_cache[d] = (tmax, tmin, 0.0)

        # Determine which past years' same-day data was used
        doy = d.timetuple().tm_yday
        hist_years_used = [
            y for y in sorted(_all_data.keys())
            if y < d.year and doy in _all_data.get(y, {})
        ][-HIST_YEARS:]

        results.append({
            'offset':      offset,
            'date':        d.isoformat(),
            'label':       'Today' if offset == 0 else WEEKDAYS[d.weekday()],
            'short_date':  f"{MONTHS[d.month - 1]} {d.day}",
            'tmax':        tmax,
            'tmin':        tmin,
            'based_on_years': hist_years_used,
        })

    return Response({
        'location': 'San Jose, CA',
        'forecast': results,
    })
