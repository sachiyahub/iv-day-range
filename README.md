# IV Day Range

Server-side tracker for NSE F&O ATM Implied Volatility day high / day low.

Public API and simple page for fetching by allnseivs.grok.me.

## Local run

```bash
pip install -r requirements.txt
python app.py
```

## Netlify

Repo is ready. Connect this repo to a new Netlify site for public access.

API: `/api/day-range?date=YYYY-MM-DD` (JSON) or `?format=csv`

CORS enabled. Data stored in data/ on the runtime.
