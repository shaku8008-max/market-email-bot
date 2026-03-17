from __future__ import annotations

import numpy as np
import pandas as pd


def _get_price_series(df: pd.DataFrame) -> pd.Series:
    """
    Extract a single price Series from a yfinance OHLCV DataFrame.
    Robust to MultiIndex columns and 1-col DataFrame selections.
    """
    if df is None or df.empty:
        raise ValueError("Input OHLCV dataframe is empty.")

    # Flatten MultiIndex columns: ('Close','^GSPC') -> 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    # Prefer Adj Close if available, else Close
    if "Adj Close" in df.columns:
        price_col = "Adj Close"
    elif "Close" in df.columns:
        price_col = "Close"
    else:
        # fallback: try to find any close-like column
        candidates = [c for c in df.columns if str(c).strip().lower() in {"adj close", "close"}]
        if not candidates:
            raise ValueError(f"Close/Adj Close not found. Columns: {list(df.columns)}")
        price_col = candidates[0]

    s = df[price_col]

    # If selection yields a DataFrame (duplicate columns), take first column
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]

    s = s.dropna()

    if s.empty:
        raise ValueError("Price series is empty after dropna().")

    # Ensure datetime index (nice to have, not required)
    try:
        s.index = pd.to_datetime(s.index)
    except Exception:
        pass

    return s


def compute_metrics(df_ohlcv: pd.DataFrame) -> dict[str, float | str]:
    """
    Compute snapshot metrics from OHLCV data.

    Returns keys:
      last_close
      change_1d_pct, change_1d_raw
      change_5d_pct, change_5d_raw
      trend
      vol_14d_pct
    """
    s = _get_price_series(df_ohlcv)

    if len(s) < 30:
        raise ValueError("Not enough rows for metrics (need ~30+ trading days).")

    last_close = float(s.iloc[-1])

    # Percent changes
    one_day_pct = float(s.pct_change(1).iloc[-1]) * 100.0
    five_day_pct = float(s.pct_change(5).iloc[-1]) * 100.0

    # Raw changes (points/$)
    one_day_raw = float(s.iloc[-1] - s.iloc[-2])

    # Need at least 6 data points to compute raw 5D change (today minus 5 trading days ago)
    if len(s) >= 6:
        five_day_raw = float(s.iloc[-1] - s.iloc[-6])
    else:
        five_day_raw = float("nan")

    # Trend: price vs 20-day MA + slope over last 20 points
    ma20 = s.rolling(20).mean()
    above_ma = bool(s.iloc[-1] > ma20.iloc[-1])

    y = s.tail(20).values
    x = np.arange(len(y))
    slope = float(np.polyfit(x, y, 1)[0])

    if above_ma and slope > 0:
        trend = "Up"
    elif (not above_ma) and slope < 0:
        trend = "Down"
    else:
        trend = "Sideways"

    # 14D volatility of daily returns (in %)
    r1 = s.pct_change(1)
    vol_14d_pct = float(r1.rolling(14).std().iloc[-1]) * 100.0

    return {
        "last_close": last_close,
        "change_1d_pct": one_day_pct,
        "change_1d_raw": one_day_raw,
        "change_5d_pct": five_day_pct,
        "change_5d_raw": five_day_raw,
        "trend": trend,
        "vol_14d_pct": vol_14d_pct,
    }


def build_summary_table(
    results: dict[str, dict[str, float | str]],
    name_map: dict[str, str],
) -> pd.DataFrame:
    """
    Convert per-ticker metrics dict into a DataFrame suitable for printing/email.
    """
    rows: list[dict[str, object]] = []

    for ticker, m in results.items():
        rows.append(
            {
                "Ticker": ticker,
                "Name": name_map.get(ticker, ticker),
                "Last Close": m.get("last_close"),
                "1D %": m.get("change_1d_pct"),
                "1D Raw": m.get("change_1d_raw"),
                "5D %": m.get("change_5d_pct"),
                "5D Raw": m.get("change_5d_raw"),
                "Trend": m.get("trend"),
                "Vol(14D)%": m.get("vol_14d_pct"),
            }
        )

    return pd.DataFrame(rows)