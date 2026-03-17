from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf
import time


@dataclass(frozen=True)
class FetchConfig:
    period: str = "2y"
    interval: str = "1d"
    cache_dir: str = "data/raw"
    max_retries: int = 3
    retry_sleep_seconds: float = 2.0


def _cache_path(cache_dir: str, ticker: str, period: str, interval: str) -> Path:
    safe = ticker.replace("^", "")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return Path(cache_dir) / f"{safe}__{period}__{interval}.csv"


def _read_cache(cache_file: Path) -> pd.DataFrame:
    df = pd.read_csv(cache_file, parse_dates=["Date"]).set_index("Date")
    df.sort_index(inplace=True)
    return df


def _write_cache(cache_file: Path, df: pd.DataFrame) -> None:
    out = df.reset_index()
    out.rename(columns={out.columns[0]: "Date"}, inplace=True)
    out.to_csv(cache_file, index=False)


def fetch_one(
    ticker: str,
    cfg: FetchConfig,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, bool]:
    """
    Returns: (df, used_cache)
    """
    cache_file = _cache_path(cfg.cache_dir, ticker, cfg.period, cfg.interval)

    # Use cache if allowed and not forcing refresh
    if use_cache and cache_file.exists() and not force_refresh:
        return _read_cache(cache_file), True

    last_error: Exception | None = None

    for attempt in range(1, cfg.max_retries + 1):
        try:
            df = yf.download(
                tickers=ticker,
                period=cfg.period,
                interval=cfg.interval,
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=True,
            )

            if df is None or df.empty:
                raise RuntimeError("Empty data returned")

            df = df.copy()
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            # Flatten MultiIndex columns like ('Close','^GSPC') -> 'Close'
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            _write_cache(cache_file, df)
            return df, False

        except Exception as e:
            last_error = e
            if attempt < cfg.max_retries:
                time.sleep(cfg.retry_sleep_seconds * attempt)
            continue

    # Download failed; fall back to cache if possible
    if use_cache and cache_file.exists():
        return _read_cache(cache_file), True

    raise RuntimeError(f"Failed to fetch {ticker} after {cfg.max_retries} retries: {last_error}")


def fetch_many(
    tickers: Iterable[str],
    cfg: FetchConfig,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, bool]]:
    data: dict[str, pd.DataFrame] = {}
    used_cache: dict[str, bool] = {}

    for t in tickers:
        df, was_cache = fetch_one(t, cfg, use_cache=use_cache, force_refresh=force_refresh)
        data[t] = df
        used_cache[t] = was_cache

    return data, used_cache