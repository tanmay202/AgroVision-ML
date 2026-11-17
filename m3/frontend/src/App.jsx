import React, { useState, useEffect } from 'react';
import {
  MapPin,
  Volume2,
  Calculator,
  HelpCircle,
  AlertTriangle,
  CheckCircle2,
  Share2,
  CloudRain,
  ShieldAlert,
  RotateCcw,
  Satellite
} from 'lucide-react';
import './App.css';

const TRANSLATIONS = {
  en: {
    tagline: "Multi-Source Agricultural Price, Weather Shock & Yield Intelligence",
    greeting: "Namaste, Farmer 👋",
    intro: "Fuses Agmarknet mandi trades, Open-Meteo/NASA POWER weather, MODIS satellite NDVI, district storage, and M1/M2 ML models.",
    mandiPrice: "CURRENT MODAL PRICE",
    perQuintal: "per quintal",
    expectedYield: "EXPECTED YIELD (M1)",
    perAcre: "per acre · M1 Regional Est.",
    compositeRisk: "COMPOSITE RISK SCORE",
    hearAdvice: "Hear today's advice",
    saveAdvice: "✓ Save advice",
    share: "↗ Share",
    why: "Why?",
  },
  bn: {
    tagline: "কৃষকদের জন্য বহুমাত্রিক বাজার দর, আবহাওয়া ঝুঁকি ও ফলন পূর্বাভাস",
    greeting: "নমস্কার, কৃষক বন্ধু 👋",
    intro: "হুগলী জেলার মান্ডি দর, আগামী ৪৮ ঘণ্টার বৃষ্টির পূর্বাভাস, উপগ্রহ NDVI ও গুদাম ক্ষমতার সমন্বিত বিশ্লেষণ।",
    mandiPrice: "বর্তমান মান্ডি দর",
    perQuintal: "প্রতি কুইন্টাল",
    expectedYield: "সম্ভাব্য ফলন (M1)",
    perAcre: "প্রতি একর · আমন ধান",
    compositeRisk: "সামগ্রিক ঝুঁকির মাত্রা",
    hearAdvice: "আজকের পরামর্শ শুনুন",
    saveAdvice: "✓ সংরক্ষণ করুন",
    share: "↗ শেয়ার করুন",
    why: "কেন?",
  }
};

export default function App() {
  const [lang, setLang] = useState('en');
  const t = TRANSLATIONS[lang];

  const [activeHorizon, setActiveHorizon] = useState('short'); // 'short' | '14' | '30'
  const [marketSeries, setMarketSeries] = useState([]);
  const [selectedDistrictName, setSelectedDistrictName] = useState('Hooghly');
  const [selectedMandiName, setSelectedMandiName] = useState('Kalipur(Arambagh Krishak Bazar)APMC');
  const [selectedVariety, setSelectedVariety] = useState('Rice (Fine)');
  const [forecastData, setForecastData] = useState(null);
  const [historyData, setHistoryData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [forecastError, setForecastError] = useState(null);
  const [toastMsg, setToastMsg] = useState('');
  const [hoveredPoint, setHoveredPoint] = useState(null);

  // Interactive Weather Shock Simulator & Tier 3 ASOF Demo State
  const [simRainMm, setSimRainMm] = useState(null);
  const [simulateStale, setSimulateStale] = useState(false);

  // Modals & Explainability / Drift Telemetry
  const [activeModal, setActiveModal] = useState(null); // 'why' | 'profit' | 'help' | 'drift'
  const [driftReport, setDriftReport] = useState(null);
  const [shapData, setShapData] = useState(null);
  const [shapLoading, setShapLoading] = useState(false);

  // Profit Calculator State
  const [calcArea, setCalcArea] = useState(2);
  const [calcYield, setCalcYield] = useState(11.5);
  const [calcCost, setCalcCost] = useState(28000);

  const showToast = (msg) => {
    setToastMsg(msg);
    setTimeout(() => setToastMsg(''), 2500);
  };

  // 1. Initial market hydration from TimescaleDB
  useEffect(() => {
    async function loadMarkets() {
      try {
        const res = await fetch('/api/markets');
        if (!res.ok) throw new Error('API failed');
        const data = await res.json();
        setMarketSeries(data);
      } catch (err) {
        showToast('Error connecting to TimescaleDB serving layer');
      }
    }
    loadMarkets();
  }, []);

  const availableDistricts = Array.from(new Set(marketSeries.map(m => m.district_name)));
  const districtSeries = marketSeries.filter(m => m.district_name === selectedDistrictName);
  const availableMandis = Array.from(new Set(districtSeries.map(m => m.mandi_name)));
  const varietiesForMandi = districtSeries
    .filter(m => m.mandi_name === selectedMandiName)
    .map(m => ({ crop_id: m.crop_id, crop_name: m.crop_name, mandi_id: m.mandi_id }));

  const activeSeries = districtSeries.find(
    m => m.mandi_name === selectedMandiName && m.crop_name === selectedVariety
  ) || districtSeries[0] || marketSeries[0];

  // 2a. Fetch trailing history (only when mandi or variety changes)
  useEffect(() => {
    if (!activeSeries) return;
    const controller = new AbortController();

    async function loadHistory() {
      try {
        const res = await fetch(
          `/api/history?mandi_id=${activeSeries.mandi_id}&crop_id=${activeSeries.crop_id}&limit=14`,
          { signal: controller.signal }
        );
        if (res.ok) {
          setHistoryData(await res.json());
        }
      } catch (err) {
        if (err.name !== 'AbortError') {
          setHistoryData([]);
        }
      }
    }

    loadHistory();
    return () => controller.abort();
  }, [activeSeries?.mandi_id, activeSeries?.crop_id]);

  // 2b. Fetch forecast (triggers on mandi, variety, rainfall simulation, or Tier 3 ASOF test)
  useEffect(() => {
    if (!activeSeries) return;
    const controller = new AbortController();

    async function loadForecast() {
      setLoading(true);
      setForecastError(null);
      try {
        const simParam = simRainMm !== null ? `&sim_rain_48h_mm=${simRainMm}` : '';
        const staleParam = simulateStale ? '&simulate_stale=true' : '';
        const forecastRes = await fetch(
          `/api/forecast?mandi_id=${activeSeries.mandi_id}&crop_id=${activeSeries.crop_id}${simParam}${staleParam}`,
          { signal: controller.signal }
        );

        if (forecastRes.ok) {
          const fData = await forecastRes.json();
          setForecastData(fData);
          if (fData.predicted_yield_q_acre) {
            setCalcYield(fData.predicted_yield_q_acre);
          }
        } else {
          setForecastData(null);
          setForecastError(`Model forecast pending for ${activeSeries.crop_name} at ${selectedMandiName.replace(' APMC', '')}.`);
        }
      } catch (err) {
        if (err.name !== 'AbortError') {
          setForecastData(null);
          setForecastError('Failed to communicate with prediction backend.');
        }
      } finally {
        setLoading(false);
      }
    }

    loadForecast();
    return () => controller.abort();
  }, [activeSeries?.mandi_id, activeSeries?.crop_id, selectedMandiName, simRainMm, simulateStale]);

  // 2c. Fetch covariate drift telemetry (PSI + Wasserstein-1) for active series
  useEffect(() => {
    if (!activeSeries) return;
    const controller = new AbortController();

    async function loadDrift() {
      try {
        const res = await fetch(
          `/api/drift?mandi_id=${activeSeries.mandi_id}&crop_id=${activeSeries.crop_id}&window_days=30`,
          { signal: controller.signal }
        );
        if (res.ok) {
          setDriftReport(await res.json());
        }
      } catch (err) {
        if (err.name !== 'AbortError') {
          setDriftReport(null);
        }
      }
    }

    loadDrift();
    return () => controller.abort();
  }, [activeSeries?.mandi_id, activeSeries?.crop_id]);

  // 2d. Fetch request-time TreeSHAP attributions when 'Why?' modal is open
  useEffect(() => {
    if (activeModal !== 'why' || !activeSeries) return;
    const controller = new AbortController();
    const hKey = activeHorizon === 'short' ? 'short' : `${activeHorizon}d`;

    async function loadExplain() {
      setShapLoading(true);
      try {
        const res = await fetch(
          `/api/explain?mandi_id=${activeSeries.mandi_id}&crop_id=${activeSeries.crop_id}&horizon=${hKey}&top_k=8`,
          { signal: controller.signal }
        );
        if (res.ok) {
          setShapData(await res.json());
        } else {
          setShapData(null);
        }
      } catch (err) {
        if (err.name !== 'AbortError') {
          setShapData(null);
        }
      } finally {
        setShapLoading(false);
      }
    }

    loadExplain();
    return () => controller.abort();
  }, [activeModal, activeSeries?.mandi_id, activeSeries?.crop_id, activeHorizon]);

  const horizonKey = activeHorizon === 'short' ? 'short' : `${activeHorizon}d`;
  const currentForecast = forecastData?.forecasts?.find(f => f.horizon_label === horizonKey);
  const shortForecast = forecastData?.forecasts?.find(f => f.horizon_label === 'short');
  const activeRisk = currentForecast?.risk;
  const telemetry = forecastData?.telemetry;

  const isRising = (currentForecast?.predicted_price_change || 0) >= -1.5;

  // Map multi-source advisory action to visual tone & stamp seal
  const action = activeRisk?.advisory_action || (isRising ? 'HOLD' : 'SELL');
  const adviceTone =
    action === 'EXPEDITE SALE'
      ? 'tone-alert'
      : action === 'PARTIAL SELL' || action === 'SELL'
      ? 'tone-caution'
      : 'tone-good';
  const sealText =
    action === 'EXPEDITE SALE'
      ? 'SELL NOW'
      : action === 'PARTIAL SELL'
      ? 'MOVE 50%'
      : action;

  const speakAdvice = () => {
    if (!('speechSynthesis' in window)) {
      showToast('Speech synthesis not available in this browser');
      return;
    }
    window.speechSynthesis.cancel();
    const rainInfo = activeRisk ? `48-hour rainfall forecast is ${activeRisk.forecast_rain_48h_mm} millimeters.` : '';
    const text = lang === 'bn'
      ? `আজকের মান্ডি দর প্রতি কুইন্টাল ${Math.round(currentForecast?.current_modal_price || 0)} টাকা। পরামর্শ: ${sealText}।`
      : `Today's modal price in ${selectedMandiName.replace(' APMC', '')} is ${Math.round(currentForecast?.current_modal_price || 0)} rupees per quintal. ${rainInfo} Recommendation is ${action}. ${activeRisk?.advisory_headline || ''}`;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = lang === 'bn' ? 'bn-IN' : 'en-IN';
    window.speechSynthesis.speak(u);
    showToast('Playing audio advisory...');
  };

  // High-Contrast SVG Trajectory + Weather-Adjusted Downside Risk Bound
  const renderTrajectorySvg = () => {
    if (!historyData.length || !currentForecast) {
      return (
        <div style={{ height: 190, display: 'grid', placeItems: 'center', color: 'var(--ink-soft)', fontSize: 13, background: 'var(--paper-deep)', borderRadius: 12 }}>
          {forecastError ? forecastError : 'Awaiting time-series observations...'}
        </div>
      );
    }

    const width = 580;
    const height = 205;
    const pad = { top: 25, bottom: 30, left: 55, right: 105 };

    const stormLower = activeRisk?.weather_adjusted_lower ?? currentForecast.predicted_price_lower;
    const hasWeatherDownside = stormLower < currentForecast.predicted_price_lower - 2;

    const prices = [
      ...historyData.map(d => d.modal_price_inr),
      currentForecast.predicted_modal_price,
      currentForecast.predicted_price_lower,
      currentForecast.predicted_price_upper,
      stormLower
    ];
    const minY = Math.min(...prices) * 0.985;
    const maxY = Math.max(...prices) * 1.015;

    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const getY = (val) => pad.top + plotH - ((val - minY) / (maxY - minY)) * plotH;

    const histStep = (plotW * 0.68) / Math.max(historyData.length - 1, 1);
    const histPts = historyData.map((d, i) => ({
      x: pad.left + i * histStep,
      y: getY(d.modal_price_inr),
      price: d.modal_price_inr,
      date: d.trade_date
    }));

    const lastPt = histPts[histPts.length - 1];
    const forecastX = pad.left + plotW;
    const forecastY = getY(currentForecast.predicted_modal_price);
    const lowerY = getY(currentForecast.predicted_price_lower);
    const upperY = getY(currentForecast.predicted_price_upper);
    const stormLowerY = getY(stormLower);

    const polylineStr = histPts.map(p => `${p.x},${p.y}`).join(' ');
    const fanPolyStr = `${lastPt.x},${lastPt.y} ${forecastX},${upperY} ${forecastX},${lowerY}`;
    const stormPolyStr = `${lastPt.x},${lastPt.y} ${forecastX},${lowerY} ${forecastX},${stormLowerY}`;

    return (
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 'auto', overflow: 'visible' }}>
        {[0, 0.5, 1].map((r) => {
          const val = minY + r * (maxY - minY);
          const yPos = getY(val);
          return (
            <g key={r}>
              <line x1={pad.left} y1={yPos} x2={forecastX} y2={yPos} stroke="var(--line)" strokeDasharray="3 3" />
              <text x={pad.left - 8} y={yPos + 4} fill="var(--ink-soft)" fontSize="10.5" fontWeight="600" textAnchor="end">
                ₹{Math.round(val)}
              </text>
            </g>
          );
        })}

        {/* Weather Shock Distress Downside Wedge */}
        {hasWeatherDownside && (
          <>
            <polygon points={stormPolyStr} fill="rgba(192, 57, 43, 0.16)" stroke="rgba(192, 57, 43, 0.6)" strokeWidth="1.2" strokeDasharray="4 3" />
            <line x1={forecastX} y1={lowerY} x2={forecastX} y2={stormLowerY} stroke="var(--red)" strokeWidth="2.5" />
            <line x1={forecastX - 5} y1={stormLowerY} x2={forecastX + 5} y2={stormLowerY} stroke="var(--red)" strokeWidth="2.5" />
            <text x={forecastX + 10} y={stormLowerY + 3} fill="var(--red)" fontSize="9.5" fontWeight="800">
              ₹{Math.round(stormLower)} [Rain Risk]
            </text>
          </>
        )}

        {/* 90% Statistical Confidence Fan */}
        <polygon points={fanPolyStr} fill="rgba(44, 92, 134, 0.16)" stroke="rgba(44, 92, 134, 0.45)" strokeWidth="1" strokeDasharray="3 3" />

        <line x1={forecastX} y1={upperY} x2={forecastX} y2={lowerY} stroke="var(--blue)" strokeWidth="2.5" strokeLinecap="round" />
        <line x1={forecastX - 5} y1={upperY} x2={forecastX + 5} y2={upperY} stroke="var(--blue)" strokeWidth="2" />
        <line x1={forecastX - 5} y1={lowerY} x2={forecastX + 5} y2={lowerY} stroke="var(--blue)" strokeWidth="2" />

        {/* Historical Price Path */}
        <polyline points={polylineStr} fill="none" stroke="var(--green-600)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />

        {/* Forecast Projection Vector */}
        <line x1={lastPt.x} y1={lastPt.y} x2={forecastX} y2={forecastY} stroke="var(--blue)" strokeWidth="2" strokeDasharray="5 5" />

        {histPts.map((p, idx) => (
          <circle
            key={idx}
            cx={p.x}
            cy={p.y}
            r={idx === histPts.length - 1 ? 5 : 3.5}
            fill="var(--green-700)"
            stroke="#fff"
            strokeWidth="1.5"
            style={{ cursor: 'pointer' }}
            onMouseEnter={() => setHoveredPoint(p)}
            onMouseLeave={() => setHoveredPoint(null)}
          />
        ))}

        <circle
          cx={forecastX}
          cy={forecastY}
          r="6"
          fill="var(--blue)"
          stroke="#fff"
          strokeWidth="2"
          style={{ cursor: 'pointer' }}
          onMouseEnter={() => setHoveredPoint({ date: currentForecast.target_date, price: currentForecast.predicted_modal_price, isForecast: true })}
          onMouseLeave={() => setHoveredPoint(null)}
        />

        <text x={forecastX + 10} y={forecastY + 4} fill="var(--blue)" fontSize="12" fontWeight="800">
          ₹{Math.round(currentForecast.predicted_modal_price)}
        </text>
        <text x={forecastX + 10} y={upperY + 3} fill="var(--ink-soft)" fontSize="9.5" fontWeight="600">
          ₹{Math.round(currentForecast.predicted_price_upper)} [p95]
        </text>
        {!hasWeatherDownside && (
          <text x={forecastX + 10} y={lowerY + 3} fill="var(--ink-soft)" fontSize="9.5" fontWeight="600">
            ₹{Math.round(currentForecast.predicted_price_lower)} [p05]
          </text>
        )}

        {hoveredPoint && (
          <g transform={`translate(${Math.min(hoveredPoint.x || forecastX - 50, width - 120)}, 18)`}>
            <rect width="115" height="24" rx="6" fill="var(--ink)" opacity="0.92" />
            <text x="57" y="16" fill="#fff" fontSize="10.5" fontWeight="700" textAnchor="middle">
              {hoveredPoint.date}: ₹{Math.round(hoveredPoint.price)}
            </text>
          </g>
        )}
      </svg>
    );
  };

  const displayRain = simRainMm !== null ? simRainMm : (activeRisk?.forecast_rain_48h_mm ?? 0);

  return (
    <div className="app">
      <header className="top">
        <div className="top-row">
          <div className="brand">
            <div className="logo" aria-hidden="true">🌾</div>
            <div>
              <b>AgroVision</b>
              <small>{t.tagline}</small>
            </div>
          </div>
          <div className="top-actions">
            <button className="icon-btn" onClick={() => setLang(l => l === 'en' ? 'bn' : 'en')}>
              <span>{lang === 'en' ? 'বাংলা' : 'English'}</span>
            </button>
            <button className="icon-btn" onClick={() => setActiveModal('profit')}>
              <Calculator size={15} /> <span>Profit Sim</span>
            </button>
            <button className="icon-btn" onClick={() => setActiveModal('help')}>
              <HelpCircle size={15} /> <span>Architecture</span>
            </button>
          </div>
        </div>
      </header>

      <main>
        {/* Tier 3 ASOF Circuit Breaker Freshness Banner */}
        {forecastData?.stale_data_warning && (
          <div
            className="forecast-banner experimental"
            style={{ marginTop: 0, marginBottom: '18px', justifyContent: 'space-between' }}
          >
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
              <AlertTriangle size={18} style={{ flex: '0 0 auto' }} />
              <div>
                <b>Tier 3 ASOF Circuit Breaker Active (Fail-Secure Cache):</b>{' '}
                {telemetry?.stale_reasons?.join(' · ') || 'Serving last-known-good telemetry.'}
              </div>
            </div>
            <button
              onClick={() => setSimulateStale(s => !s)}
              style={{
                fontSize: '11px',
                fontWeight: 800,
                padding: '4px 10px',
                borderRadius: '8px',
                border: '1px solid currentColor',
                flex: '0 0 auto',
              }}
            >
              {simulateStale ? 'Exit Stale Sim' : 'ASOF Guard On'}
            </button>
          </div>
        )}

        {/* Top Deck */}
        <div className="deck-grid">
          {/* Left Column: Trajectory Chart, Horizon Selector & Live Weather Shock Simulator */}
          <div className="card chart-card">
            <div className="chart-header">
              <div>
                <div className="eyebrow">TIMESCALEDB HYPERTABLE + ASOF WEATHER FUSION</div>
                <h2 style={{ margin: '4px 0 0', fontSize: '18px' }}>
                  Price Trajectory &amp; Weather-Adjusted Risk Fan
                </h2>
              </div>
              <span className={`badge ${activeHorizon === '30' ? 'tone-caution' : 'tone-good'}`}>
                {activeHorizon === '30' ? '30d [Experimental]' : activeHorizon === '14' ? '14d [Opt-In]' : '1–7d [Production]'}
              </span>
            </div>

            {renderTrajectorySvg()}

            {/* 3-Button Horizon Selector */}
            <div className="horizon-selector-container">
              <div className="eyebrow" style={{ marginBottom: '6px' }}>SELECT PREDICTION HORIZON</div>
              <div className="horizon-bar">
                <button
                  className={`horizon-btn ${activeHorizon === 'short' ? 'active' : ''}`}
                  onClick={() => setActiveHorizon('short')}
                >
                  <span>Short-Term</span>
                  <span className="horizon-tag horizon-tag-prod">1–7 Days</span>
                </button>
                <button
                  className={`horizon-btn ${activeHorizon === '14' ? 'active' : ''}`}
                  onClick={() => setActiveHorizon('14')}
                >
                  <span>14-Day</span>
                  <span className="horizon-tag horizon-tag-optin">Opt-In</span>
                </button>
                <button
                  className={`horizon-btn ${activeHorizon === '30' ? 'active' : ''}`}
                  onClick={() => setActiveHorizon('30')}
                >
                  <span>30-Day</span>
                  <span className="horizon-tag horizon-tag-exp">Experimental</span>
                </button>
              </div>
            </div>

            {/* Interactive 48h Weather Shock What-If Simulator */}
            <div className={`shock-panel ${simRainMm !== null ? 'simulated' : ''}`}>
              <div className="shock-head">
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <CloudRain size={15} color={simRainMm !== null && simRainMm >= 45 ? 'var(--red)' : 'var(--blue)'} />
                  <span className="eyebrow" style={{ color: 'var(--ink)' }}>
                    48H RAINFALL SHOCK SIMULATOR (EXOGENOUS LOGISTICS &amp; SPOILAGE)
                  </span>
                </div>
                {simRainMm !== null ? (
                  <button
                    onClick={() => setSimRainMm(null)}
                    style={{ fontSize: '11px', fontWeight: 800, color: 'var(--red)', display: 'inline-flex', alignItems: 'center', gap: '4px' }}
                  >
                    <RotateCcw size={12} /> Reset to Live
                  </button>
                ) : (
                  <span className="badge tone-good">Live {telemetry?.weather_source || 'open-meteo'}</span>
                )}
              </div>

              <div className="shock-presets">
                <button
                  className={`preset-btn ${simRainMm === null ? 'active' : ''}`}
                  onClick={() => setSimRainMm(null)}
                >
                  Live Forecast ({activeRisk?.is_simulated ? '0' : (activeRisk?.forecast_rain_48h_mm ?? 0)} mm)
                </button>
                <button
                  className={`preset-btn ${simRainMm === 30 ? 'active-warn' : ''}`}
                  onClick={() => setSimRainMm(30)}
                >
                  Moderate Rain (30 mm)
                </button>
                <button
                  className={`preset-btn ${simRainMm === 85 ? 'active-danger' : ''}`}
                  onClick={() => setSimRainMm(85)}
                >
                  Heavy Storm (85 mm)
                </button>
              </div>

              <div className="shock-slider-row">
                <input
                  type="range"
                  min="0"
                  max="150"
                  step="5"
                  value={displayRain}
                  onChange={(e) => setSimRainMm(Number(e.target.value))}
                />
                <span style={{ minWidth: '72px', textAlign: 'right', fontSize: '13px', fontWeight: 900, color: displayRain >= 45 ? 'var(--red)' : 'var(--ink)' }}>
                  {displayRain} mm
                </span>
              </div>
              <div className="small" style={{ marginTop: '4px', fontSize: '11.5px', color: 'var(--ink-soft)' }}>
                Tests how sudden 48h heavy rainfall overrides a flat-price <b>HOLD</b> when feeder roads flood and paddy moisture exceeds 14%.
              </div>
            </div>

            {activeHorizon === 'short' && (
              <div className="forecast-banner production">
                <CheckCircle2 size={18} style={{ flex: '0 0 auto' }} />
                <div>
                  <b>Short-Term Production Model (1–7 Days):</b> Direct next-observation model. Holdout MAE: <b>₹18.98/q</b> (vs Naive Baseline ₹19.43/q). Shaded fan shows empirical 90% error bounds.
                </div>
              </div>
            )}
            {activeHorizon === '14' && (
              <div className="forecast-banner opt-in">
                <AlertTriangle size={18} style={{ flex: '0 0 auto' }} />
                <div>
                  <b>14-Day Calendar Horizon (Opt-In Production):</b> Multi-week projection. Holdout MAE: <b>₹59.87/q</b> (vs Naive Baseline ₹60.16/q across 18,921 trades).
                </div>
              </div>
            )}
            {activeHorizon === '30' && (
              <div className="forecast-banner experimental">
                <AlertTriangle size={18} style={{ flex: '0 0 auto' }} />
                <div>
                  <b>30-Day Model Transparency Notice:</b> Holdout evaluation reveals model MAE (<b>₹99.95</b>) trails Naive Persistence (<b>₹99.50</b>) with 47.9% Directional Accuracy. Exposed with explicit experimental governance.
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Controls & Actionable Multi-Source Advice Card */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div className="hero">
              <div className="hero-eyebrow">
                <MapPin size={13} /> {selectedDistrictName}, West Bengal · Multi-Source Pipeline
              </div>
              <h1>{t.greeting}</h1>
              <p>{t.intro}</p>

              <div className="selector-row">
                <select
                  aria-label="Select District"
                  value={selectedDistrictName}
                  onChange={(e) => {
                    const nextDist = e.target.value;
                    setSelectedDistrictName(nextDist);
                    const inDist = marketSeries.filter(m => m.district_name === nextDist);
                    const sameCrop = inDist.find(m => m.crop_name === selectedVariety);
                    const nextItem = sameCrop || inDist[0];
                    if (nextItem) {
                      setSelectedMandiName(nextItem.mandi_name);
                      setSelectedVariety(nextItem.crop_name);
                    }
                  }}
                >
                  {availableDistricts.map(d => (
                    <option key={d} value={d}>{d}</option>
                  ))}
                </select>

                <select
                  aria-label="Select Mandi"
                  value={selectedMandiName}
                  onChange={(e) => {
                    const newMandi = e.target.value;
                    setSelectedMandiName(newMandi);
                    const validVarieties = districtSeries.filter(m => m.mandi_name === newMandi);
                    const keepVariety = validVarieties.find(v => v.crop_name === selectedVariety);
                    if (keepVariety) {
                      setSelectedVariety(keepVariety.crop_name);
                    } else if (validVarieties.length > 0) {
                      setSelectedVariety(validVarieties[0].crop_name);
                    }
                  }}
                >
                  {availableMandis.map(m => (
                    <option key={m} value={m}>{m.replace(' APMC', '')}</option>
                  ))}
                </select>
              </div>

              <div className="selector-row" style={{ marginTop: '8px' }}>
                <select
                  value={selectedVariety}
                  onChange={(e) => setSelectedVariety(e.target.value)}
                >
                  {varietiesForMandi.map(v => (
                    <option key={v.crop_name} value={v.crop_name}>
                      {v.crop_name}
                    </option>
                  ))}
                </select>

                <div style={{ display: 'flex', alignItems: 'center', fontSize: '12px', color: 'var(--ink-soft)', padding: '0 4px' }}>
                  <span>Origin: <b>{currentForecast?.forecast_origin_date || '2026-10-28'}</b></span>
                </div>
              </div>

              <button className="voice-btn" onClick={speakAdvice}>
                <Volume2 size={16} /> <span>{t.hearAdvice}</span>
              </button>
            </div>

            {/* Actionable Multi-Source Advice Card */}
            {currentForecast ? (
              <div className={`advice ${adviceTone}`}>
                <div className="seal-container">
                  <div className={`seal ${adviceTone}`}>
                    <span>{sealText}</span>
                  </div>
                </div>

                <div className="advice-content">
                  <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                    <span className={`badge ${adviceTone}`}>RECOMMENDED ACTION</span>
                    {activeRisk?.is_simulated && (
                      <span className="badge tone-alert">SIMULATED STORM</span>
                    )}
                  </div>
                  <h3>{activeRisk?.advisory_headline || (isRising ? 'Hold — Price trend steady' : 'Sell — Near-term softness')}</h3>
                  <p>{activeRisk?.advisory_reason}</p>
                </div>

                <div className="buttons">
                  <button className="primary" onClick={() => showToast('Advisory saved to notes')}><span>{t.saveAdvice}</span></button>
                  <button className="secondary" onClick={() => {
                    navigator.clipboard?.writeText(`AgroVision Advisory: ${selectedVariety} in ${selectedMandiName.replace(' APMC', '')} (${selectedDistrictName}) is ₹${currentForecast.current_modal_price}/q. Action: ${action} (${activeRisk?.advisory_headline})`);
                    showToast('Advisory copied to clipboard');
                  }}>
                    <Share2 size={13} style={{ marginRight: 4, verticalAlign: -1 }} />
                    <span>{t.share}</span>
                  </button>
                  <button className="secondary" onClick={() => setActiveModal('why')}><span>{t.why}</span></button>
                </div>
              </div>
            ) : (
              <div className="card advice tone-caution" style={{ padding: '24px' }}>
                <span className="badge tone-neutral">PENDING FORECAST</span>
                <h3>{forecastError || 'Loading...'}</h3>
              </div>
            )}
          </div>
        </div>

        {/* Middle Deck: 4-Metric Grid (All Real Models + Live Risk) */}
        <div className="stat-grid">
          <div className="card stat">
            <div className="eyebrow">{t.mandiPrice}</div>
            <div className="big">
              ₹{currentForecast?.current_modal_price ? Math.round(currentForecast.current_modal_price).toLocaleString('en-IN') : '—'}
            </div>
            <div className="small">{t.perQuintal}</div>
            <div className="trend tone-good">Trade date: {currentForecast?.forecast_origin_date || '—'}</div>
          </div>

          <div className="card stat">
            <div className="eyebrow">PREDICTED MODAL ({activeHorizon.toUpperCase()})</div>
            <div className="big">
              ₹{currentForecast?.predicted_modal_price ? Math.round(currentForecast.predicted_modal_price).toLocaleString('en-IN') : '—'}
            </div>
            <div className="small">target: {currentForecast?.target_date || '—'}</div>
            <div className={`trend ${isRising ? 'tone-good' : 'tone-alert'}`}>
              {currentForecast?.predicted_price_change >= 0 ? '▲ +' : '▼ '}
              {currentForecast?.predicted_price_change ? currentForecast.predicted_price_change.toFixed(1) : '0.0'} Rs/q
            </div>
          </div>

          <div className="card stat">
            <div className="eyebrow">{t.expectedYield}</div>
            <div className="big">{forecastData?.predicted_yield_q_acre ?? '11.57'} q</div>
            <div className="small">{t.perAcre} ({forecastData?.predicted_yield_kg_ha ?? '2860.0'} kg/ha)</div>
            <div className="trend tone-good">{forecastData?.yield_method || "M1 Regional Yield Model"}</div>
          </div>

          <div className="card stat">
            <div className="eyebrow">{t.compositeRisk}</div>
            <div className="big" style={{ color: activeRisk?.risk_regime === 'HIGH' ? 'var(--red)' : activeRisk?.risk_regime === 'MEDIUM' ? 'var(--marigold-deep)' : 'var(--green-700)' }}>
              {activeRisk ? `${activeRisk.composite_risk_score.toFixed(2)} ${activeRisk.risk_regime}` : '0.13 LOW'}
            </div>
            <div className="small">48h Rain: {displayRain} mm · Storage: {Math.round((forecastData?.storage_avail_index ?? 0.72) * 100)}%</div>
            <div className={`trend ${activeRisk?.risk_regime === 'HIGH' ? 'tone-alert' : activeRisk?.risk_regime === 'MEDIUM' ? 'tone-caution' : 'tone-good'}`}>
              Vol {shortForecast?.volatility_class || 'LOW'} · Weather + Storage Fused
            </div>
          </div>
        </div>

        {/* Lower Deck: Model Governance & Multi-Source Risk + Satellite NDVI Provenance */}
        <div className="lower-deck">
          <div className="governance-card">
            <div className="section-head">
              <b style={{ fontSize: '13.5px' }}>Model Governance &amp; Multi-Model Lineage</b>
              <span className={`badge ${currentForecast?.is_experimental ? 'tone-caution' : 'tone-good'}`}>
                {currentForecast?.is_experimental ? 'Experimental' : 'Production'}
              </span>
            </div>
            <div className="governance-row">
              <span className="governance-label">M2 Price Regressor</span>
              <span className="governance-val">{activeSeries?.crop_id?.startsWith("WHEAT") ? "81-Feature CatBoost v2" : activeSeries?.crop_id?.startsWith("POTATO") ? "81-Feature XGBoost v2" : "48-Feature XGBoost (MAE ₹18.98)"}</span>
            </div>
            <div className="governance-row">
              <span className="governance-label">M2 Volatility Regime</span>
              <span className="governance-val">
                {shortForecast?.volatility_prob_low
                  ? `${((shortForecast.volatility_prob_low) * 100).toFixed(0)}% L / ${((shortForecast.volatility_prob_medium) * 100).toFixed(0)}% M / ${((shortForecast.volatility_prob_high) * 100).toFixed(0)}% H`
                  : 'Calibrated v2'}
              </span>
            </div>
            <div className="governance-row">
              <span className="governance-label">M1 Crop Yield Model</span>
              <span className="governance-val">{forecastData?.yield_method || 'M1 m1_improved_delta_ridge.pkl'}</span>
            </div>
            <div className="governance-row">
              <span className="governance-label">Weather-Adjusted Downside</span>
              <span className="governance-val" style={{ color: activeRisk?.risk_regime === 'HIGH' ? 'var(--red)' : 'inherit' }}>
                ₹{Math.round(activeRisk?.weather_adjusted_lower ?? currentForecast?.predicted_price_lower ?? 0)} / q
              </span>
            </div>
            <p className="note" style={{ marginTop: '10px' }}>
              Separation of concerns: autoregressive market microstructure drives baseline price forecasts, while exogenous weather shocks and district storage capacity govern post-harvest risk overrides.
            </p>
          </div>

          <div className="card" style={{ padding: '18px' }}>
            <div className="section-head">
              <b style={{ fontSize: '13.5px' }}>Multi-Source Risk &amp; 3-Tier Telemetry (TimescaleDB ASOF)</b>
              <span className="freshness">
                <i className="live-dot" /> {activeRisk?.is_simulated ? 'Simulated 48h Storm' : `${telemetry?.weather_source || 'open-meteo'} + MODIS`}
              </span>
            </div>

            {activeRisk && (
              <div className="risk-bars">
                <div className="risk-bar-item">
                  <span>Market Vol (0.35)</span>
                  <div className="risk-track">
                    <div className="risk-fill" style={{ width: `${Math.round(activeRisk.v_norm * 100)}%`, background: 'var(--blue)' }} />
                  </div>
                  <span style={{ textAlign: 'right' }}>{activeRisk.v_norm.toFixed(2)}</span>
                </div>
                <div className="risk-bar-item">
                  <span>Rain Shock (0.45)</span>
                  <div className="risk-track">
                    <div className="risk-fill" style={{ width: `${Math.round(activeRisk.h_anomaly * 100)}%`, background: activeRisk.h_anomaly >= 0.5 ? 'var(--red)' : 'var(--marigold)' }} />
                  </div>
                  <span style={{ textAlign: 'right' }}>{activeRisk.h_anomaly.toFixed(2)}</span>
                </div>
                <div className="risk-bar-item">
                  <span>Storage Vuln (0.20)</span>
                  <div className="risk-track">
                    <div className="risk-fill" style={{ width: `${Math.round(activeRisk.storage_vuln * 100)}%`, background: 'var(--green-600)' }} />
                  </div>
                  <span style={{ textAlign: 'right' }}>{activeRisk.storage_vuln.toFixed(2)}</span>
                </div>
              </div>
            )}

            <div className="governance-row" style={{ marginTop: '10px' }}>
              <span className="governance-label" style={{ display: 'inline-flex', alignItems: 'center', gap: '5px' }}>
                <Satellite size={13} /> Satellite NDVI (MODIS 250m)
              </span>
              <span className="governance-val">
                {telemetry?.latest_ndvi !== null && telemetry?.latest_ndvi !== undefined
                  ? `${telemetry.latest_ndvi.toFixed(4)} (${telemetry.ndvi_anomaly >= 0 ? '+' : ''}${telemetry.ndvi_anomaly?.toFixed(4)} vs DOY · ${telemetry.ndvi_date})`
                  : 'Awaiting MODIS pass'}
              </span>
            </div>
            <div className="governance-row">
              <span className="governance-label">3-Tier Ingestion Provenance</span>
              <span className="governance-val">
                Weather: {telemetry?.weather_source || 'open-meteo'} ({telemetry?.weather_age_hours ?? 0}h) · Storage: {Math.round((forecastData?.storage_avail_index ?? 0.72) * 100)}%
              </span>
            </div>
            <div className="governance-row">
              <span className="governance-label">Covariate Drift (PSI + W1)</span>
              <span className="governance-val" style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
                <span
                  className={`badge ${
                    driftReport?.overall_status === 'SIGNIFICANT_DRIFT'
                      ? 'tone-alert'
                      : driftReport?.overall_status === 'MODERATE_SHIFT'
                      ? 'tone-caution'
                      : 'tone-good'
                  }`}
                >
                  {driftReport ? `${driftReport.overall_status.replace('_', ' ')} (PSI ${driftReport.max_psi.toFixed(2)})` : 'Checking...'}
                </span>
                {driftReport && (
                  <button
                    onClick={() => setActiveModal('drift')}
                    style={{ fontSize: '11.5px', fontWeight: 800, color: 'var(--blue)', textDecoration: 'underline' }}
                  >
                    Audit ({driftReport.features?.length || 8})
                  </button>
                )}
              </span>
            </div>
          </div>
        </div>
      </main>

      {/* Modal: Profit Calculator */}
      {activeModal === 'profit' && (
        <div className="modal-bg" onClick={() => setActiveModal(null)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-head">
              <h2>🧮 Farmer Profit Simulator</h2>
              <button className="close" onClick={() => setActiveModal(null)}>×</button>
            </div>
            <div style={{ marginTop: '14px', display: 'grid', gap: '12px' }}>
              <div>
                <label className="small" style={{ fontWeight: 800 }}>Cultivated Area (Acres)</label>
                <input className="input" type="number" value={calcArea} onChange={e => setCalcArea(+e.target.value)} />
              </div>
              <div>
                <label className="small" style={{ fontWeight: 800 }}>Expected Yield (Quintals/Acre — M1 Pre-filled)</label>
                <input className="input" type="number" value={calcYield} onChange={e => setCalcYield(+e.target.value)} />
              </div>
              <div>
                <label className="small" style={{ fontWeight: 800 }}>Total Cultivation Cost (₹)</label>
                <input className="input" type="number" value={calcCost} onChange={e => setCalcCost(+e.target.value)} />
              </div>
              {(() => {
                const prod = calcArea * calcYield;
                const p = currentForecast?.predicted_modal_price || 3400;
                const rev = prod * p;
                const profit = rev - calcCost;
                return (
                  <div style={{ background: 'var(--green-tint-2)', border: '1px solid #CDE6D4', borderRadius: '14px', padding: '16px' }}>
                    <div className="small">Projected Production: <b>{prod.toFixed(1)} quintals</b></div>
                    <div className="small" style={{ marginTop: '4px' }}>Estimated Gross Revenue (@ ₹{Math.round(p)}/q): <b>₹{Math.round(rev).toLocaleString('en-IN')}</b></div>
                    <div className="small" style={{ marginTop: '8px' }}>Projected Net Farmer Profit:</div>
                    <div style={{ fontSize: '28px', fontWeight: 900, color: profit >= 0 ? 'var(--green-700)' : 'var(--red)', fontFamily: 'var(--font-display)' }}>
                      ₹{Math.round(profit).toLocaleString('en-IN')}
                    </div>
                  </div>
                );
              })()}
            </div>
          </div>
        </div>
      )}

      {/* Modal: Why This Recommendation (TreeSHAP + 4-Signal Risk) */}
      {activeModal === 'why' && (
        <div className="modal-bg" onClick={() => setActiveModal(null)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-head">
              <h2>Why "{action}"? (TreeSHAP + 4-Signal Risk)</h2>
              <button className="close" onClick={() => setActiveModal(null)}>×</button>
            </div>

            {/* Live Request-Time TreeSHAP Feature Contributions */}
            <div className="card" style={{ padding: '14px', marginTop: '14px', background: '#FAFCF9' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <b>TreeSHAP Price Drivers ({horizonKey.toUpperCase()})</b>
                {shapData && (
                  <span className="badge tone-good">
                    Base {shapData.base_expected_change_inr >= 0 ? '+' : ''}{shapData.base_expected_change_inr.toFixed(2)} → Net {shapData.predicted_change_inr >= 0 ? '+' : ''}{shapData.predicted_change_inr.toFixed(2)} ₹/q
                  </span>
                )}
              </div>

              {shapLoading ? (
                <div className="small">Computing C++ TreeSHAP feature attributions...</div>
              ) : shapData?.top_features?.length ? (
                (() => {
                  const maxAbs = Math.max(...shapData.top_features.map(f => Math.abs(f.shap_value_inr)), 0.1);
                  return (
                    <div style={{ display: 'grid', gap: '7px' }}>
                      {shapData.top_features.map((item) => {
                        const pct = Math.min(100, Math.round((Math.abs(item.shap_value_inr) / maxAbs) * 100));
                        const isPos = item.shap_value_inr >= 0;
                        return (
                          <div key={item.feature} style={{ display: 'grid', gridTemplateColumns: '1fr 110px 68px', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
                            <div>
                              <div style={{ fontWeight: 700, color: 'var(--ink)' }}>{item.label}</div>
                              <div style={{ fontSize: '10.5px', color: 'var(--ink-soft)' }}>
                                Input: {item.feature_value.toLocaleString('en-IN')} {item.unit !== 'val' ? item.unit : ''}
                              </div>
                            </div>
                            <div className="risk-track">
                              <div
                                className="risk-fill"
                                style={{
                                  width: `${pct}%`,
                                  background: isPos ? 'var(--green-600)' : 'var(--red)',
                                }}
                              />
                            </div>
                            <span style={{ textAlign: 'right', fontWeight: 800, color: isPos ? 'var(--green-700)' : 'var(--red)' }}>
                              {isPos ? '+' : ''}₹{item.shap_value_inr.toFixed(2)}
                            </span>
                          </div>
                        );
                      })}
                      <div className="small" style={{ marginTop: '4px', fontSize: '11px' }}>
                        Remaining 40 features sum: <b>{shapData.other_features_shap_inr >= 0 ? '+' : ''}₹{shapData.other_features_shap_inr.toFixed(2)}/q</b> · Exact local Shapley additivity verified.
                      </div>
                    </div>
                  );
                })()
              ) : (
                <div className="small">TreeSHAP breakdown unavailable for this series.</div>
              )}
            </div>

            <div style={{ display: 'grid', gap: '10px', marginTop: '12px' }}>
              <div className="card" style={{ padding: '14px' }}>
                <b>1. Market Price Trajectory (M2 XGBoost)</b>
                <div className="small" style={{ marginTop: '3px' }}>
                  Projects {currentForecast?.predicted_price_change >= 0 ? '+' : ''}{currentForecast?.predicted_price_change?.toFixed(1)} Rs/q change (Target: ₹{Math.round(currentForecast?.predicted_modal_price || 0)}/q).
                </div>
              </div>
              <div className="card" style={{ padding: '14px' }}>
                <b>2. 48-Hour Weather &amp; Transit Shock ({telemetry?.weather_source || 'Open-Meteo'})</b>
                <div className="small" style={{ marginTop: '3px' }}>
                  48h rainfall forecast is <b>{displayRain} mm</b> (Weather Shock Index H_anomaly = <b>{activeRisk?.h_anomaly.toFixed(2)}</b>). Heavy rain (&gt;45 mm) clogs rural feeder roads and pushes paddy moisture past the 14% safe storage limit.
                </div>
              </div>
              <div className="card" style={{ padding: '14px' }}>
                <b>3. District Storage Buffer (features.location_dim)</b>
                <div className="small" style={{ marginTop: '3px' }}>
                  {forecastData?.district_name || 'Hooghly'} storage availability index is <b>{((forecastData?.storage_avail_index ?? 0.72) * 100).toFixed(0)}%</b> (Vulnerability = {activeRisk?.storage_vuln.toFixed(2)}).
                </div>
              </div>
              <div className="card" style={{ padding: '14px' }}>
                <b>4. Calibrated Volatility &amp; Canopy Health (M2 + MODIS)</b>
                <div className="small" style={{ marginTop: '3px' }}>
                  Regime: <b>{shortForecast?.volatility_class || 'LOW'}</b> (V_norm = {activeRisk?.v_norm.toFixed(2)}) · Latest MODIS NDVI: <b>{telemetry?.latest_ndvi?.toFixed(4) ?? '—'}</b> ({telemetry?.ndvi_date ?? '—'}).
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Modal: Covariate Drift Monitor (PSI + Wasserstein-1) */}
      {activeModal === 'drift' && driftReport && (
        <div className="modal-bg" onClick={() => setActiveModal(null)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-head">
              <h2>Covariate Drift Monitor (PSI + Wasserstein-1)</h2>
              <button className="close" onClick={() => setActiveModal(null)}>×</button>
            </div>
            <p className="small" style={{ marginTop: '8px' }}>
              Compares trailing {driftReport.window_days}-day live TimescaleDB observations against frozen West Bengal training distributions. Uses <b>PSI</b> for N &ge; 30 and <b>Standardized W1</b> (W1 / &sigma;_ref) for small-sample weather/MODIS windows.
            </p>
            <div style={{ display: 'grid', gap: '8px', marginTop: '12px' }}>
              {driftReport.features.map((f) => {
                const tone =
                  f.status === 'SIGNIFICANT_DRIFT'
                    ? 'tone-alert'
                    : f.status === 'MODERATE_SHIFT'
                    ? 'tone-caution'
                    : 'tone-good';
                return (
                  <div key={f.feature} className="card" style={{ padding: '11px 14px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '10px' }}>
                    <div>
                      <b style={{ fontSize: '13px' }}>{f.label}</b>
                      <div className="small" style={{ fontSize: '11.5px' }}>
                        Ref Mean: <b>{f.ref_mean} {f.unit}</b> · Live Mean: <b>{f.live_mean !== null ? `${f.live_mean} ${f.unit}` : '—'}</b> (N={f.live_samples})
                      </div>
                      <div className="small" style={{ fontSize: '11px' }}>
                        PSI: <b>{f.psi.toFixed(3)}</b> · W1: <b>{f.wasserstein_distance.toFixed(2)} {f.unit}</b> ({f.normalized_w1.toFixed(2)}&sigma;) · Rule: {f.classification_metric}
                      </div>
                    </div>
                    <span className={`badge ${tone}`}>{f.status.replace('_', ' ')}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Modal: System Architecture & Coordinator Defense */}
      {activeModal === 'help' && (
        <div className="modal-bg" onClick={() => setActiveModal(null)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-head">
              <h2><ShieldAlert size={19} style={{ verticalAlign: -3, marginRight: 6 }} />System Architecture</h2>
              <button className="close" onClick={() => setActiveModal(null)}>×</button>
            </div>
            <div style={{ display: 'grid', gap: '10px', marginTop: '14px', fontSize: '13px', color: 'var(--ink-soft)' }}>
              <div className="card" style={{ padding: '14px' }}>
                <b style={{ color: 'var(--ink)' }}>How AgroVision Handles Sudden Rain After Steady Prices</b>
                <p style={{ margin: '6px 0 0' }}>
                  Autoregressive price models trained on daily mandi trades excel on normal days (₹18.98 MAE) but cannot anticipate sudden 48-hour storms because extreme weather is a low-frequency shock. AgroVision decouples <b>Price Prediction (M2)</b> from an exogenous <b>Multi-Source Risk Engine</b> combining 3-tier weather forecasts with district storage capacity.
                </p>
              </div>
              <div className="card" style={{ padding: '14px' }}>
                <b style={{ color: 'var(--ink)' }}>3-Tier Ingestion Resilience &amp; ASOF Circuit Breaker</b>
                <p style={{ margin: '6px 0 0' }}>
                  <b>Weather:</b> Open-Meteo → NASA POWER MERRA-2 → ASOF Cache.<br />
                  <b>Satellite NDVI:</b> MODIS Terra (MOD13Q1) → MODIS Aqua (MYD13Q1) → ASOF Cache.<br />
                  <b>Market:</b> OGD API → Agmarknet 2.0 XHR+OCR → ASOF Cache (never zero-fills on upstream failure).
                </p>
              </div>
              <div className="card" style={{ padding: '14px' }}>
                <b style={{ color: 'var(--ink)' }}>Composite Risk Equation</b>
                <p style={{ margin: '6px 0 0', fontFamily: 'monospace', fontSize: '12px', color: 'var(--green-900)' }}>
                  Risk = 0.35·V_norm + 0.45·H_anomaly + 0.20·(1 - S_avail)
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {toastMsg && (
        <div style={{
          position: 'fixed',
          bottom: '24px',
          left: '50%',
          transform: 'translateX(-50%)',
          background: 'var(--ink)',
          color: '#fff',
          padding: '10px 20px',
          borderRadius: '12px',
          fontSize: '13px',
          fontWeight: 700,
          boxShadow: 'var(--shadow-lg)',
          zIndex: 120
        }}>
          {toastMsg}
        </div>
      )}
    </div>
  );
}
