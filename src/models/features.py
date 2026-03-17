from __future__ import annotations

import numpy as np
import pandas as pd


def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _pick_close(df: pd.DataFrame) -> pd.Series:

    df = _flatten_cols(df)

    if "Adj Close" in df.columns:
        s = df["Adj Close"]
    else:
        s = df["Close"]

    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]

    return s.astype(float).dropna()


def split_train_test_timewise(feat: pd.DataFrame, test_size: float = 0.3):

    n = len(feat)
    split = int(n * (1 - test_size))

    train = feat.iloc[:split].copy()
    test = feat.iloc[split:].copy()

    return train, test


def _rsi(series: pd.Series, window=14):

    delta = series.diff()

    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.ewm(alpha=1/window).mean()
    roll_down = down.ewm(alpha=1/window).mean()

    rs = roll_up / roll_down
    rsi = 100 - (100/(1+rs))

    return rsi


def build_features_and_target(df_ohlcv: pd.DataFrame) -> pd.DataFrame:

    df = _flatten_cols(df_ohlcv).copy()

    close = _pick_close(df)

    base = pd.DataFrame(index=close.index)
    base["close"] = close

    # =========================
    # RETURNS
    # =========================

    base["ret_1"] = close.pct_change(1)
    base["ret_5"] = close.pct_change(5)
    base["ret_10"] = close.pct_change(10)
    base["ret_20"] = close.pct_change(20)

    # =========================
    # VOLATILITY
    # =========================

    base["vol_14"] = base["ret_1"].rolling(14).std()
    base["vol_20"] = base["ret_1"].rolling(20).std()
    base["vol_60"] = base["ret_1"].rolling(60).std()

    base["vol_ratio_20_60"] = base["vol_20"] / base["vol_60"]

    # =========================
    # MOVING AVERAGES
    # =========================

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    base["dist_ma20"] = (close - ma20) / ma20
    base["dist_ma50"] = (close - ma50) / ma50

    base["z20"] = (close - ma20) / close.rolling(20).std()
    base["z50"] = (close - ma50) / close.rolling(50).std()

    base["ma_gap_20"] = ma20.pct_change(5)

    # slope
    base["slope_20"] = close.rolling(20).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0],
        raw=False
    )

    # =========================
    # RSI
    # =========================

    base["rsi_14"] = _rsi(close)

    # =========================
    # BOLLINGER
    # =========================

    ma = close.rolling(20).mean()
    sd = close.rolling(20).std()

    upper = ma + 2*sd
    lower = ma - 2*sd

    base["bb_pos_20"] = (close-lower)/(upper-lower)
    base["bb_width_20"] = (upper-lower)/ma

    # =========================
    # VOLUME SHOCK
    # =========================

    if "Volume" in df.columns:
        vol = df["Volume"]
        if isinstance(vol, pd.DataFrame):
            vol = vol.iloc[:,0]

        base["vol_spike_20"] = vol / vol.rolling(20).mean()
    else:
        base["vol_spike_20"] = np.nan

    # =========================
    # ADX
    # =========================

    if "High" in df.columns and "Low" in df.columns:

        high = df["High"]
        low = df["Low"]

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = pd.concat([
            high-low,
            (high-close.shift()).abs(),
            (low-close.shift()).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(14).mean()

        plus_di = 100*(plus_dm.rolling(14).mean()/atr)
        minus_di = 100*(minus_dm.rolling(14).mean()/atr)

        dx = abs(plus_di-minus_di)/(plus_di+minus_di)*100

        base["adx_14"] = dx.rolling(14).mean()

    else:
        base["adx_14"] = np.nan

    # =========================
    # VOL REGIME
    # =========================

    base["vol20_pctl_126"] = base["vol_20"].rolling(126).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )

    base["high_vol_flag"] = (base["vol20_pctl_126"] > 0.8).astype(float)
    base["low_vol_flag"] = (base["vol20_pctl_126"] < 0.2).astype(float)

    base["vol_regime"] = base["high_vol_flag"] - base["low_vol_flag"]

    # =========================
    # MARKET REGIME
    # =========================

    base["bull_trend_flag"] = (ma50 > ma200).astype(float)
    base["bear_trend_flag"] = (ma50 < ma200).astype(float)

    peak = close.rolling(126).max()

    base["drawdown_126"] = (close/peak)-1

    base["crash_flag"] = (base["drawdown_126"] < -0.15).astype(float)

    panic = (base["crash_flag"]==1) & (base["high_vol_flag"]==1)

    trend_up = (ma50>ma200) & (base["adx_14"]>25)
    trend_down = (ma50<ma200) & (base["adx_14"]>25)

    base["regime_label"] = np.select(
        [panic, trend_down, trend_up],
        ["PANIC","TREND_DOWN","TREND_UP"],
        default="MEAN_REVERT"
    )

    regime_map = {
        "MEAN_REVERT":0,
        "TREND_UP":1,
        "TREND_DOWN":2,
        "PANIC":3
    }

    base["regime_id"] = base["regime_label"].map(regime_map)

    # =========================
    # TARGETS
    # =========================

    base["target_1d"] = close.pct_change(1).shift(-1)

    base["target_3d"] = close.pct_change(3).shift(-3)

    base["target_5d"] = close.pct_change(5).shift(-5)

    # =========================
    # CLEAN
    # =========================

    base = base.replace([np.inf,-np.inf],np.nan)
    base = base.dropna()

    return base