# src/config.py
from __future__ import annotations

TICKERS: dict[str, str] = {
    "^GSPC": "S&P 500",
    "^DJI": "Dow Jones",
    "^NDX": "Nasdaq 100",
    "IOO": "S&P Global 100 (proxy: IOO)",
}

# How much history to pull for computing metrics
DEFAULT_PERIOD = "2y"      # enough for rolling stats + stability
DEFAULT_INTERVAL = "1d"    # daily bars
CACHE_DIR = "data/raw"     # cached downloads (gitignored)