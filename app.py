#!/usr/bin/env python3
"""
IV Day Range - NSE F&O ATM IV day high/low tracker.
Server-side, always running. Public API + simple page.
"""

import os
import json
import csv
import time
import datetime as dt
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import concurrent.futures

from fastapi import FastAPI, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Optional: nse library
try:
    from nse import NSE
    HAS_NSE = True
except ImportError:
    HAS_NSE = False

from scipy.stats import norm
from scipy.optimize import brentq
import numpy as np

IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Holidays 2026 (weekday trading holidays)
NSE_HOLIDAYS_2026 = {
    "2026-01-15", "2026-01-26", "2026-03-03", "2026-03-26", "2026-03-31",
    "2026-04-03", "2026-04-14", "2026-05-01", "2026-05-28", "2026-06-26",
    "2026-09-14", "2026-10-02", "2026-10-20", "2026-11-10", "2026-11-24",
    "2026-12-25",
}

INDICES = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"]

app = FastAPI(title="IV Day Range")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ivday")

# In-memory state for current day and health
current_day_data: Dict = {}  # key: (symbol, book) -> dict
health = {
    "last_cycle_time": None,
    "names_updated": 0,
    "errors": 0,
    "status": "idle",
    "last_error": None,
}

# Previous minute IV for jump filter
prev_iv: Dict[Tuple[str, str], float] = {}
pending_jump: Dict[Tuple[str, str], float] = {}  # for 2-min confirmation

def is_trading_day(d: dt.date) -> bool:
    if d.weekday() >= 5:  # Sat/Sun
        return False
    return d.isoformat() not in NSE_HOLIDAYS_2026

def now_ist() -> datetime:
    return datetime.now(IST)

def is_market_hours() -> bool:
    n = now_ist()
    if not is_trading_day(n.date()):
        return False
    t = n.time()
    return dt.time(9, 16) <= t <= dt.time(15, 30)

def load_universe() -> List[str]:
    """Load current F&O symbols from NSE lot file or fallback."""
    symbols = INDICES.copy()
    try:
        import requests
        url = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if r.ok:
            lines = r.text.splitlines()
            for line in lines[8:]:  # skip headers
                parts = [p.strip() for p in line.split(",")]
                if len(parts) > 1 and parts[1] and parts[1] not in ("SYMBOL", "Symbol"):
                    sym = parts[1]
                    if sym and sym not in symbols:
                        symbols.append(sym)
        log.info(f"Loaded {len(symbols)} symbols from lot file")
    except Exception as e:
        log.warning(f"Could not load lot file: {e}. Using indices only.")
    return symbols

UNIVERSE = load_universe()

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == "call" else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_vol(price, S, K, T, r, option_type="call"):
    if price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    try:
        iv = brentq(
            lambda sigma: black_scholes_price(S, K, T, r, sigma, option_type) - price,
            0.01, 5.0, xtol=1e-6, maxiter=100
        )
        return iv * 100  # percent
    except Exception:
        return None

def get_expiries(symbol: str, nse_client) -> Tuple[Optional[str], Optional[str]]:
    """Return (front_expiry, next_expiry) as YYYY-MM-DD or None."""
    try:
        if symbol in INDICES:
            exps = nse_client.get_futures_expiry(index=symbol.lower())
        else:
            # For stocks, use option chain to get expiries or approximate
            chain = nse_client.option_chain(symbol)
            # Parse expiries from chain if available
            exps = []
            if isinstance(chain, dict) and "records" in chain:
                raw = chain["records"].get("expiryDates", [])
                exps = raw[:3]
        if not exps:
            return None, None
        # Convert to date
        front = None
        next_ = None
        today = now_ist().date()
        parsed = []
        for e in exps:
            try:
                if isinstance(e, str):
                    d = datetime.strptime(e, "%d-%b-%Y").date()
                else:
                    d = e
                if d >= today:
                    parsed.append(d)
            except Exception:
                continue
        parsed.sort()
        if parsed:
            front = parsed[0].isoformat()
            if len(parsed) > 1:
                next_ = parsed[1].isoformat()
        return front, next_
    except Exception as e:
        log.debug(f"Expiry fetch failed for {symbol}: {e}")
        return None, None

def compute_atm_iv(symbol: str, expiry: str, nse_client, r=0.065) -> Optional[float]:
    """Compute average ATM CE/PE IV off futures price. Returns IV % or None."""
    try:
        exp_dt = datetime.strptime(expiry, "%Y-%m-%d").date()
        chain = nse_client.compile_option_chain(symbol, expiry_date=exp_dt)
        if not chain or "atm" not in chain or "underlying" not in chain:
            return None
        atm_strike = chain["atm"]
        # Prefer futures price; fall back to underlying
        S = chain.get("underlying") or 0
        # Try to get futures quote if possible
        try:
            # Approximate: many chains have futures in records
            pass
        except Exception:
            pass
        if S <= 0:
            return None
        T = max(0.0, (exp_dt - now_ist().date()).days / 365.0)
        if T <= 0:
            T = 1 / 365.0
        strike_data = chain.get("chain", {}).get(str(int(atm_strike))) or chain.get("chain", {}).get(str(atm_strike))
        if not strike_data:
            return None
        ce = strike_data.get("ce", {})
        pe = strike_data.get("pe", {})
        ce_price = ce.get("last") or 0
        pe_price = pe.get("last") or 0
        # Filter zero/blank
        if ce_price <= 0 or pe_price <= 0:
            return None
        # Simple volume/last trade proxy: if last is present and >0, assume recent enough for now
        iv_ce = implied_vol(ce_price, S, atm_strike, T, r, "call")
        iv_pe = implied_vol(pe_price, S, atm_strike, T, r, "put")
        if iv_ce is None or iv_pe is None:
            return None
        avg_iv = (iv_ce + iv_pe) / 2.0
        if avg_iv < 3 or avg_iv > 200:
            return None
        return round(avg_iv, 2)
    except Exception as e:
        log.debug(f"ATM IV failed {symbol} {expiry}: {e}")
        return None

def update_day_record(symbol: str, book: str, iv: float, ts: str):
    key = (symbol, book)
    rec = current_day_data.get(key, {
        "date": now_ist().date().isoformat(),
        "book": book,
        "symbol": symbol,
        "day_high_iv": iv,
        "day_high_time": ts,
        "day_low_iv": iv,
        "day_low_time": ts,
        "last_iv": iv,
        "last_time": ts,
        "samples": 0,
    })
    rec["samples"] += 1
    rec["last_iv"] = iv
    rec["last_time"] = ts
    if iv > rec["day_high_iv"]:
        rec["day_high_iv"] = iv
        rec["day_high_time"] = ts
    if iv < rec["day_low_iv"]:
        rec["day_low_iv"] = iv
        rec["day_low_time"] = ts
    current_day_data[key] = rec

def save_day(date_str: str):
    path = DATA_DIR / f"{date_str}.json"
    records = list(current_day_data.values())
    # Only save if this is current day data
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    log.info(f"Saved {len(records)} records for {date_str}")

def load_day(date_str: str) -> List[dict]:
    path = DATA_DIR / f"{date_str}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []

def cycle():
    """One polling cycle."""
    if not is_market_hours():
        health["status"] = "outside hours"
        return
    health["status"] = "running"
    health["errors"] = 0
    updated = 0
    start = time.time()
    n = now_ist()
    ts = n.strftime("%H:%M:%S")
    date_str = n.date().isoformat()

    if not HAS_NSE:
        health["last_error"] = "nse library not available"
        health["errors"] += 1
        return

    try:
        with NSE(download_folder=str(DATA_DIR / "nse_cache")) as nse:
            for symbol in UNIVERSE:
                front_exp, next_exp = get_expiries(symbol, nse)
                for book, exp in [("front", front_exp), ("next", next_exp)]:
                    if not exp:
                        continue
                    try:
                        iv = compute_atm_iv(symbol, exp, nse)
                        if iv is None:
                            continue
                        key = (symbol, book)
                        # Jump filter: >40% change that doesn't hold
                        prev = prev_iv.get(key)
                        if prev is not None:
                            change = abs(iv - prev) / max(prev, 0.1)
                            if change > 0.40:
                                # Require 2 consecutive
                                if pending_jump.get(key) == iv:
                                    # Confirmed
                                    pass
                                else:
                                    pending_jump[key] = iv
                                    continue  # skip this reading
                            else:
                                pending_jump.pop(key, None)
                        prev_iv[key] = iv
                        update_day_record(symbol, book, iv, ts)
                        updated += 1
                    except Exception as e:
                        health["errors"] += 1
                        log.debug(f"Error {symbol} {book}: {e}")
                        if health["errors"] > 50:
                            break
                if health["errors"] > 50:
                    break
        save_day(date_str)
        health["names_updated"] = updated
        health["last_cycle_time"] = n.isoformat()
        health["status"] = "ok"
        elapsed = time.time() - start
        log.info(f"Cycle done: {updated} names in {elapsed:.1f}s, errors={health['errors']}")
    except Exception as e:
        health["last_error"] = str(e)
        health["errors"] += 1
        health["status"] = "error"
        log.error(f"Cycle failed: {e}")

def scheduler_loop():
    """Background loop: run every ~60s during market hours."""
    while True:
        try:
            if is_market_hours():
                cycle()
            else:
                health["status"] = "closed"
            # Sleep to next minute
            now = time.time()
            sleep_for = 60 - (now % 60) + 1
            time.sleep(max(5, sleep_for))
        except Exception as e:
            log.error(f"Scheduler error: {e}")
            time.sleep(30)

@app.on_event("startup")
def start_scheduler():
    t = threading.Thread(target=scheduler_loop, daemon=True)
    t.start()
    log.info("IV Day Range scheduler started")

@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>IV Day Range</title>
<style>
body { font-family: system-ui, sans-serif; margin: 1.5rem; background: #0f1419; color: #e7e9ea; }
h1 { margin: 0 0 0.5rem; }
.controls { display: flex; gap: 1rem; align-items: center; margin: 1rem 0; flex-wrap: wrap; }
input, select, button { padding: 0.4rem 0.6rem; border-radius: 4px; border: 1px solid #38444d; background: #1e2732; color: inherit; }
table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
th, td { padding: 0.4rem 0.5rem; text-align: left; border-bottom: 1px solid #38444d; }
th { cursor: pointer; user-select: none; }
tr:hover { background: #1e2732; }
.health { font-size: 0.85rem; color: #8b98a5; margin-top: 0.5rem; }
.tabs button.active { background: #1d9bf0; color: white; border-color: #1d9bf0; }
#search { min-width: 200px; }
</style>
</head>
<body>
<h1>IV Day Range</h1>
<div class="controls">
  <label>Date <input type="date" id="date"></label>
  <div class="tabs">
    <button id="tab-front" class="active" data-book="front">Front (Today)</button>
    <button id="tab-next" data-book="next">Next month</button>
  </div>
  <input type="search" id="search" placeholder="Search symbol...">
  <button id="refresh">Refresh</button>
</div>
<div class="health" id="health">Loading...</div>
<table id="table">
  <thead>
    <tr>
      <th data-col="symbol">Symbol</th>
      <th data-col="day_high_iv">Day High IV</th>
      <th data-col="day_high_time">High Time</th>
      <th data-col="day_low_iv">Day Low IV</th>
      <th data-col="day_low_time">Low Time</th>
      <th data-col="last_iv">Last IV</th>
      <th data-col="last_time">Last Time</th>
      <th data-col="samples">Samples</th>
    </tr>
  </thead>
  <tbody></tbody>
</table>
<script>
const dateEl = document.getElementById('date');
const searchEl = document.getElementById('search');
const tbody = document.querySelector('#table tbody');
const healthEl = document.getElementById('health');
let currentBook = 'front';
let allData = [];

dateEl.valueAsDate = new Date();

async function load() {
  const date = dateEl.value;
  const res = await fetch(`/api/day-range?date=${date}`);
  const data = await res.json();
  allData = data.records || [];
  render();
  const h = data.health || {};
  healthEl.textContent = `Last cycle: ${h.last_cycle_time || '—'} · Names updated: ${h.names_updated || 0} · Errors: ${h.errors || 0} · Status: ${h.status || '—'}`;
}

function render() {
  const q = searchEl.value.toLowerCase();
  const filtered = allData.filter(r => r.book === currentBook && (!q || r.symbol.toLowerCase().includes(q)));
  tbody.innerHTML = filtered.map(r => `
    <tr>
      <td>${r.symbol}</td>
      <td>${r.day_high_iv ?? ''}</td>
      <td>${r.day_high_time ?? ''}</td>
      <td>${r.day_low_iv ?? ''}</td>
      <td>${r.day_low_time ?? ''}</td>
      <td>${r.last_iv ?? ''}</td>
      <td>${r.last_time ?? ''}</td>
      <td>${r.samples ?? 0}</td>
    </tr>`).join('');
}

document.querySelectorAll('.tabs button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentBook = btn.dataset.book;
    render();
  });
});
searchEl.addEventListener('input', render);
dateEl.addEventListener('change', load);
document.getElementById('refresh').addEventListener('click', load);
load();
setInterval(load, 30000);
</script>
</body>
</html>
"""

@app.get("/api/day-range")
def api_day_range(date: str = Query(None), format: str = Query("json")):
    if not date:
        date = now_ist().date().isoformat()
    records = load_day(date)
    # Merge with live current day if matching
    if date == now_ist().date().isoformat():
        live = list(current_day_data.values())
        # Prefer live for current
        records = live or records
    if format.lower() == "csv":
        def gen():
            if not records:
                yield "date,book,symbol,day_high_iv,day_high_time,day_low_iv,day_low_time,last_iv,last_time,samples\n"
                return
            yield "date,book,symbol,day_high_iv,day_high_time,day_low_iv,day_low_time,last_iv,last_time,samples\n"
            for r in records:
                yield f"{r.get('date','')},{r.get('book','')},{r.get('symbol','')},{r.get('day_high_iv','')},{r.get('day_high_time','')},{r.get('day_low_iv','')},{r.get('day_low_time','')},{r.get('last_iv','')},{r.get('last_time','')},{r.get('samples',0)}\n"
        return StreamingResponse(gen(), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=iv-day-range-{date}.csv"})
    return {
        "date": date,
        "records": records,
        "health": health,
        "updated": now_ist().isoformat(),
    }

@app.get("/api/health")
def api_health():
    return health

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
