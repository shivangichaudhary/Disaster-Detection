"""
dashboard/app.py
─────────────────
FastAPI + Jinja2 dashboard serving a Leaflet.js GIS map
with real-time disaster alert overlays.
"""

import os
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

from loguru import logger

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/disaster_db")

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Disaster Detection Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; }
  #app { display: flex; height: 100vh; }
  #sidebar { width: 320px; background: #1e293b; display: flex; flex-direction: column; overflow: hidden; border-right: 1px solid #334155; }
  #map { flex: 1; }
  .sidebar-header { padding: 16px; background: #0f172a; border-bottom: 1px solid #334155; }
  .sidebar-header h1 { font-size: 16px; font-weight: 600; color: #f1f5f9; }
  .sidebar-header p  { font-size: 12px; color: #64748b; margin-top: 2px; }
  .live-badge { display: inline-flex; align-items: center; gap: 4px; font-size: 11px; color: #22c55e; margin-top: 6px; }
  .live-dot { width: 6px; height: 6px; background: #22c55e; border-radius: 50%; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
  .filter-bar { padding: 12px 16px; border-bottom: 1px solid #334155; display: flex; gap: 8px; flex-wrap: wrap; }
  .filter-btn { padding: 4px 10px; border-radius: 999px; border: 1px solid #334155; background: transparent;
                color: #94a3b8; font-size: 12px; cursor: pointer; transition: all .15s; }
  .filter-btn.active { background: #3b82f6; border-color: #3b82f6; color: white; }
  .filter-btn:hover { border-color: #94a3b8; color: #e2e8f0; }
  .alerts-list { flex: 1; overflow-y: auto; }
  .alert-card { padding: 12px 16px; border-bottom: 1px solid #1e293b; cursor: pointer; transition: background .1s; }
  .alert-card:hover { background: #0f172a; }
  .alert-card .type-badge { display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px;
                            border-radius: 4px; font-size: 11px; font-weight: 500; margin-bottom: 4px; }
  .badge-flood      { background: #1d4ed8; color: #bfdbfe; }
  .badge-earthquake { background: #b45309; color: #fde68a; }
  .badge-wildfire   { background: #b91c1c; color: #fecaca; }
  .badge-cyclone    { background: #6d28d9; color: #ddd6fe; }
  .badge-landslide  { background: #047857; color: #a7f3d0; }
  .badge-none       { background: #374151; color: #d1d5db; }
  .alert-loc   { font-size: 12px; color: #94a3b8; }
  .alert-conf  { font-size: 11px; color: #64748b; margin-top: 2px; }
  .sev-bar { height: 3px; border-radius: 2px; margin-top: 6px; }
  .sev-low  { background: #22c55e; width: 33%; }
  .sev-med  { background: #f59e0b; width: 66%; }
  .sev-high { background: #ef4444; width: 100%; }
  .stats-bar { padding: 12px 16px; border-top: 1px solid #334155; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
  .stat { text-align: center; }
  .stat .val  { font-size: 18px; font-weight: 600; color: #f1f5f9; }
  .stat .lbl  { font-size: 10px; color: #64748b; margin-top: 2px; }
  .leaflet-popup-content-wrapper { background: #1e293b; color: #e2e8f0; border: 1px solid #334155; border-radius: 8px; }
  .leaflet-popup-tip { background: #1e293b; }
  .popup-title { font-weight: 600; font-size: 14px; color: #f1f5f9; margin-bottom: 6px; }
  .popup-row   { font-size: 12px; color: #94a3b8; margin: 2px 0; }
  .popup-row span { color: #e2e8f0; }
  #toast { position: fixed; bottom: 20px; right: 20px; background: #1e293b; border: 1px solid #334155;
           border-left: 3px solid #ef4444; padding: 10px 14px; border-radius: 6px; font-size: 13px;
           display: none; z-index: 9999; max-width: 280px; }
  #toast.show { display: block; animation: slidein .3s ease; }
  @keyframes slidein { from { transform: translateX(30px); opacity: 0; } }
</style>
</head>
<body>
<div id="app">
  <div id="sidebar">
    <div class="sidebar-header">
      <h1>Disaster Detection</h1>
      <p>SAR + Social Media Fusion</p>
      <div class="live-badge"><span class="live-dot"></span> Live monitoring</div>
    </div>
    <div class="filter-bar">
      <button class="filter-btn active" onclick="setFilter('all')">All</button>
      <button class="filter-btn" onclick="setFilter('flood')">Flood</button>
      <button class="filter-btn" onclick="setFilter('earthquake')">EQ</button>
      <button class="filter-btn" onclick="setFilter('wildfire')">Fire</button>
      <button class="filter-btn" onclick="setFilter('cyclone')">Cyclone</button>
    </div>
    <div class="alerts-list" id="alertsList"></div>
    <div class="stats-bar">
      <div class="stat"><div class="val" id="statTotal">0</div><div class="lbl">Total</div></div>
      <div class="stat"><div class="val" id="statHigh">0</div><div class="lbl">High sev.</div></div>
      <div class="stat"><div class="val" id="statToday">0</div><div class="lbl">Today</div></div>
    </div>
  </div>
  <div id="map"></div>
</div>
<div id="toast"></div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const DISASTER_COLORS = {
  flood: '#3b82f6', earthquake: '#f59e0b',
  wildfire: '#ef4444', cyclone: '#8b5cf6',
  landslide: '#22c55e', none: '#64748b',
};
const DISASTER_ICONS = {
  flood: '💧', earthquake: '🌍', wildfire: '🔥',
  cyclone: '🌀', landslide: '⛰️', none: '📍',
};

let map, markers = {}, alerts = [], activeFilter = 'all';

function initMap() {
  map = L.map('map', { center: [20, 0], zoom: 2, zoomControl: true });
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '© CartoDB', maxZoom: 19,
  }).addTo(map);
}

function makeCircleIcon(type, severity) {
  const color = DISASTER_COLORS[type] || '#64748b';
  const size  = severity === 'high' ? 22 : severity === 'medium' ? 16 : 12;
  return L.divIcon({
    className: '',
    iconSize: [size, size],
    html: `<div style="width:${size}px;height:${size}px;border-radius:50%;
           background:${color};opacity:0.9;border:2px solid white;
           box-shadow:0 0 8px ${color}"></div>`,
  });
}

function addMarker(alert) {
  if (markers[alert.id]) return;
  const m = L.marker([alert.lat, alert.lon], { icon: makeCircleIcon(alert.disaster_type, alert.severity) })
    .addTo(map)
    .bindPopup(`
      <div class="popup-title">${DISASTER_ICONS[alert.disaster_type] || '📍'} ${alert.disaster_type.toUpperCase()}</div>
      <div class="popup-row">Severity: <span>${alert.severity}</span></div>
      <div class="popup-row">Confidence: <span>${(alert.confidence * 100).toFixed(1)}%</span></div>
      <div class="popup-row">Location: <span>${alert.lat.toFixed(3)}, ${alert.lon.toFixed(3)}</span></div>
      <div class="popup-row">Time: <span>${new Date(alert.timestamp).toLocaleString()}</span></div>
    `);
  markers[alert.id] = m;
}

function renderCard(alert) {
  const sev_class = `sev-${alert.severity}`;
  return `
    <div class="alert-card" onclick="flyTo(${alert.lat},${alert.lon},${alert.id})">
      <span class="type-badge badge-${alert.disaster_type}">
        ${DISASTER_ICONS[alert.disaster_type]} ${alert.disaster_type}
      </span>
      <div class="alert-loc">${alert.lat.toFixed(3)}, ${alert.lon.toFixed(3)}</div>
      <div class="alert-conf">${(alert.confidence * 100).toFixed(1)}% confidence · ${alert.severity} severity</div>
      <div class="sev-bar ${sev_class}"></div>
    </div>`;
}

function flyTo(lat, lon, id) {
  map.flyTo([lat, lon], 9, { duration: 1.2 });
  if (markers[id]) markers[id].openPopup();
}

function setFilter(f) {
  activeFilter = f;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  renderAlerts();
}

function renderAlerts() {
  const filtered = activeFilter === 'all' ? alerts : alerts.filter(a => a.disaster_type === activeFilter);
  document.getElementById('alertsList').innerHTML = filtered.map(renderCard).join('');
  filtered.forEach(addMarker);
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 4000);
}

function updateStats() {
  const today = new Date(); today.setHours(0,0,0,0);
  document.getElementById('statTotal').textContent = alerts.length;
  document.getElementById('statHigh').textContent  = alerts.filter(a => a.severity === 'high').length;
  document.getElementById('statToday').textContent = alerts.filter(a => new Date(a.timestamp) >= today).length;
}

async function fetchAlerts() {
  try {
    const r = await fetch('/api/alerts/recent?hours=48&limit=200');
    const d = await r.json();
    const newAlerts = (d.alerts || []).filter(a => a.is_disaster !== false);
    const prevLen   = alerts.length;
    alerts = newAlerts;
    renderAlerts();
    updateStats();
    if (alerts.length > prevLen && prevLen > 0) {
      const a = alerts[0];
      showToast(`⚠️ New ${a.disaster_type} alert at ${a.lat.toFixed(2)}, ${a.lon.toFixed(2)}`);
    }
  } catch(e) {
    // API not available — load demo data
    loadDemoData();
  }
}

function loadDemoData() {
  alerts = [
    { id:1, lat:19.08, lon:72.88, disaster_type:'flood',      severity:'high',   confidence:0.92, timestamp: new Date().toISOString(), is_disaster: true },
    { id:2, lat:28.61, lon:77.21, disaster_type:'earthquake', severity:'medium', confidence:0.78, timestamp: new Date(Date.now()-3600000).toISOString(), is_disaster: true },
    { id:3, lat:13.08, lon:80.27, disaster_type:'cyclone',    severity:'high',   confidence:0.88, timestamp: new Date(Date.now()-7200000).toISOString(), is_disaster: true },
    { id:4, lat:22.57, lon:88.36, disaster_type:'flood',      severity:'low',    confidence:0.61, timestamp: new Date(Date.now()-10800000).toISOString(), is_disaster: true },
    { id:5, lat:17.38, lon:78.49, disaster_type:'wildfire',   severity:'medium', confidence:0.74, timestamp: new Date(Date.now()-14400000).toISOString(), is_disaster: true },
  ];
  renderAlerts();
  updateStats();
}

initMap();
fetchAlerts();
setInterval(fetchAlerts, 30000);  // refresh every 30s
</script>
</body>
</html>
"""


if FASTAPI_AVAILABLE:
    dashboard_app = FastAPI(title="Disaster Dashboard")

    @dashboard_app.get("/", response_class=HTMLResponse)
    def index():
        return DASHBOARD_HTML

    # Global state for mock live alerts
    MOCK_ALERTS = []

    @dashboard_app.get("/api/alerts/recent")
    def api_alerts(hours: int = 48, limit: int = 200):
        import random

        # ── Try live DB first ─────────────────────────────────────────────
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            try:
                from sqlalchemy import create_engine, text
                engine = create_engine(db_url, pool_pre_ping=True)
                since  = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
                with engine.connect() as conn:
                    rows = conn.execute(text("""
                        SELECT id, timestamp::text, lat, lon,
                               disaster_type, severity, confidence,
                               true as is_disaster
                        FROM disaster_alerts
                        WHERE timestamp > :since AND disaster_type != 'none'
                        ORDER BY timestamp DESC LIMIT :lim
                    """), {"since": since, "lim": limit})
                    db_alerts = [dict(r._mapping) for r in rows]
                if db_alerts:
                    return {"alerts": db_alerts, "source": "database"}
            except Exception as e:
                logger.warning(f"DB query failed, falling back to mock: {e}")

        # ── Mock data fallback (no DB or DB empty) ────────────────────────
        # Known land coordinate pairs for reliable mock data
        _LAND_COORDS = [
            (19.076, 72.877), (28.613, 77.209), (13.082, 80.270),
            (22.572, 88.363), (17.385, 78.486), (51.507, -0.127),
            (40.714, -74.006), (35.689, 139.692), (-33.868, 151.209),
            (48.852, 2.350),  (55.755, 37.617),  (39.916, 116.397),
            (1.352, 103.820), (-23.550, -46.633), (6.524, 3.379),
            (30.044, 31.235), (-1.286, 36.817),   (9.050, 7.499),
            (33.886, 9.537),  (14.693, -17.448),
        ]

        # Add a new mock alert with 40% probability on each request
        if random.random() < 0.4 or not MOCK_ALERTS:
            lat, lon = random.choice(_LAND_COORDS)
            # Add small jitter so points don't overlap, kept small to stay near the city center
            lat += random.uniform(-0.02, 0.02)
            lon += random.uniform(-0.02, 0.02)
            new_alert = {
                "id": len(MOCK_ALERTS) + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "disaster_type": random.choice(['flood', 'earthquake', 'wildfire', 'cyclone', 'landslide']),
                "severity": random.choice(['low', 'medium', 'high']),
                "confidence": round(random.uniform(0.6, 0.99), 4),
                "is_disaster": True
            }
            MOCK_ALERTS.append(new_alert)

        return {"alerts": MOCK_ALERTS[-limit:], "source": "mock"}


if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        uvicorn.run(dashboard_app, host="0.0.0.0", port=8081, log_level="info")
    else:
        logger.error("FastAPI not installed. pip install fastapi uvicorn")
