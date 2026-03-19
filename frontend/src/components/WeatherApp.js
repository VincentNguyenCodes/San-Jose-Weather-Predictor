import { useState, useEffect } from 'react';
import './WeatherApp.css';

const SUN_ICON = (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="5" />
    <line x1="12" y1="1" x2="12" y2="3" />
    <line x1="12" y1="21" x2="12" y2="23" />
    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
    <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
    <line x1="1" y1="12" x2="3" y2="12" />
    <line x1="21" y1="12" x2="23" y2="12" />
    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
    <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
  </svg>
);

const CLOUD_SUN_ICON = (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 2v2M4.93 4.93l1.41 1.41M20 12h2M19.07 4.93l-1.41 1.41" />
    <circle cx="12" cy="9" r="3" />
    <path d="M6 18a4 4 0 0 1 0-8 5 5 0 0 1 9.9-1A4.5 4.5 0 0 1 18 18Z" />
  </svg>
);

function getIcon(tmax) {
  return tmax >= 75 ? SUN_ICON : CLOUD_SUN_ICON;
}

function getCondition(tmax) {
  if (tmax >= 85) return 'Sunny & Hot';
  if (tmax >= 75) return 'Sunny';
  if (tmax >= 65) return 'Partly Cloudy';
  if (tmax >= 55) return 'Cloudy';
  return 'Cool & Overcast';
}

function TempBar({ lo, hi, allDays }) {
  const allLo   = Math.min(...allDays.map(d => d.tmin));
  const allHi   = Math.max(...allDays.map(d => d.tmax));
  const range   = allHi - allLo || 1;
  const leftPct = ((lo - allLo) / range) * 100;
  const widthPct = ((hi - lo)   / range) * 100;
  return (
    <div className="temp-bar-track">
      <div className="temp-bar-fill" style={{ left: `${leftPct}%`, width: `${widthPct}%` }} />
    </div>
  );
}

// ── Forecast tab ──────────────────────────────────────────────────────────────
function ForecastTab() {
  const [data, setData]     = useState(null);
  const [error, setError]   = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('http://127.0.0.1:8001/api/forecast/')
      .then(r => r.json())
      .then(json => { setData(json); setLoading(false); })
      .catch(() => { setError('Could not connect to the weather server.'); setLoading(false); });
  }, []);

  if (loading) return <div className="tab-loading">Loading forecast…</div>;
  if (error)   return <div className="tab-error">{error}</div>;

  const today = data.forecast[0];
  const week  = data.forecast.slice(1);

  return (
    <>
      <div className="weather-header">
        <div className="weather-location">{data.location}</div>
        <div className="weather-condition">{getCondition(today.tmax)}</div>
        <div className="weather-hero-temp">{today.tmax}<span className="deg">°</span></div>
        <div className="weather-hi-lo">
          <span>H:{today.tmax}°</span>
          <span>L:{today.tmin}°</span>
        </div>
      </div>

      <div className="weather-divider" />
      <div className="weather-forecast-label">7-DAY FORECAST</div>

      <div className="weather-forecast-list">
        {week.map(day => (
          <div className="weather-day-row" key={day.date}>
            <span className="day-label">{day.label}</span>
            <span className="day-icon">{getIcon(day.tmax)}</span>
            <div className="day-temps">
              <span className="day-lo">{day.tmin}°</span>
              <div className="day-bar-wrap">
                <TempBar lo={day.tmin} hi={day.tmax} allDays={data.forecast} />
              </div>
              <span className="day-hi">{day.tmax}°</span>
            </div>
          </div>
        ))}
      </div>

      <div className="weather-footer">
        Neural network trained on 2015–2025 · predicted from same date in{' '}
        {data.forecast[0].based_on_years.join(', ')} + recent days
      </div>
    </>
  );
}

// ── Predict tab ───────────────────────────────────────────────────────────────
function PredictTab() {
  const today = new Date().toISOString().split('T')[0];
  const [inputDate, setInputDate] = useState(today);
  const [result, setResult]       = useState(null);
  const [error, setError]         = useState(null);
  const [loading, setLoading]     = useState(false);

  function handleSubmit(e) {
    e.preventDefault();
    if (!inputDate) return;
    setLoading(true);
    setResult(null);
    setError(null);

    fetch(`http://127.0.0.1:8001/api/predict/?date=${inputDate}`)
      .then(r => r.json())
      .then(json => {
        if (json.error) { setError(json.error); }
        else            { setResult(json); }
        setLoading(false);
      })
      .catch(() => { setError('Could not connect to the weather server.'); setLoading(false); });
  }

  return (
    <div className="predict-tab">
      <div className="predict-header">
        <div className="weather-location">San Jose, CA</div>
        <div className="weather-condition">Date Prediction</div>
      </div>

      <form className="predict-form" onSubmit={handleSubmit}>
        <label className="predict-label">ENTER A DATE</label>
        <input
          className="predict-input"
          type="date"
          value={inputDate}
          onChange={e => setInputDate(e.target.value)}
          required
        />
        <button className="predict-btn" type="submit" disabled={loading}>
          {loading ? 'Predicting…' : 'Predict'}
        </button>
      </form>

      {error && <div className="predict-error">{error}</div>}

      {result && (
        <div className="predict-result">
          <div className="predict-result-date">{result.short_date}</div>
          <div className="predict-result-day">{result.label}</div>

          <div className="predict-temps">
            <div className="predict-temp-block">
              <div className="predict-temp-val">{result.tmax}°</div>
              <div className="predict-temp-lbl">High</div>
            </div>
            <div className="predict-temp-divider" />
            <div className="predict-temp-block">
              <div className="predict-temp-val predict-temp-lo">{result.tmin}°</div>
              <div className="predict-temp-lbl">Low</div>
            </div>
          </div>

          <div className="predict-condition">{getCondition(result.tmax)}</div>

          {result.based_on_years.length > 0 && (
            <div className="predict-sources">
              Based on {result.based_on_years.join(', ')}
            </div>
          )}
        </div>
      )}

      <div className="weather-footer" style={{ marginTop: 'auto' }}>
        Neural network trained on 2015–2025 San Jose historical data
      </div>
    </div>
  );
}

// ── Root ──────────────────────────────────────────────────────────────────────
export default function WeatherApp() {
  const [tab, setTab] = useState('forecast');

  return (
    <div className="weather-root">
      <div className="weather-card">
        <div className="tab-bar">
          <button
            className={`tab-btn ${tab === 'forecast' ? 'tab-active' : ''}`}
            onClick={() => setTab('forecast')}
          >
            Forecast
          </button>
          <button
            className={`tab-btn ${tab === 'predict' ? 'tab-active' : ''}`}
            onClick={() => setTab('predict')}
          >
            Predict a Date
          </button>
        </div>

        {tab === 'forecast' ? <ForecastTab /> : <PredictTab />}
      </div>
    </div>
  );
}
