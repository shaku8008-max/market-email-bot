from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Forecast:
    last_close: float

    pred_1d_return_pct: float
    pred_1d_move_raw: float
    pred_1d_target_close: float

    pred_5d_return_pct: float
    pred_5d_move_raw: float
    pred_5d_target_close: float


def _get_price_series(df: pd.DataFrame) -> pd.Series:
    """
    Extract a single price Series from a yfinance OHLCV DataFrame.
    Mirrors the robustness logic used in metrics.py.
    """
    if df is None or df.empty:
        raise ValueError("Input OHLCV dataframe is empty.")

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    if "Adj Close" in df.columns:
        price_col = "Adj Close"
    elif "Close" in df.columns:
        price_col = "Close"
    else:
        candidates = [c for c in df.columns if str(c).strip().lower() in {"adj close", "close"}]
        if not candidates:
            raise ValueError(f"Close/Adj Close not found. Columns: {list(df.columns)}")
        price_col = candidates[0]

    s = df[price_col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]

    s = s.dropna()
    if s.empty:
        raise ValueError("Price series is empty after dropna().")

    try:
        s.index = pd.to_datetime(s.index)
    except Exception:
        pass

    return s


def _make_feature_frame(price: pd.Series) -> pd.DataFrame:
    """
    Build features + targets for 1D and 5D Close→Close returns.
    """
    s = price.dropna().copy()

    r1 = s.pct_change(1)

    df = pd.DataFrame(index=s.index)
    df["ret_1"] = r1
    df["ret_2"] = s.pct_change(2)
    df["ret_5"] = s.pct_change(5)
    df["ma_5"] = s.rolling(5).mean()
    df["ma_20"] = s.rolling(20).mean()
    df["ma_ratio_5_20"] = (df["ma_5"] / df["ma_20"]) - 1.0
    df["vol_14"] = r1.rolling(14).std()

    # Targets: future returns (Close→Close)
    df["y_1d"] = s.pct_change(1).shift(-1)
    df["y_5d"] = s.pct_change(5).shift(-5)

    df = df.dropna()
    return df


def _ridge_fit_predict(X: np.ndarray, y: np.ndarray, x_last: np.ndarray, alpha: float = 10.0) -> float:
    """
    Ridge regression with standardization, implemented in NumPy:
      beta = (X^T X + alpha I)^-1 X^T y
    Returns prediction for x_last.
    """
    # Standardize features
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0

    Xs = (X - mu) / sigma
    xls = (x_last - mu) / sigma

    # Add intercept
    X_design = np.column_stack([np.ones(len(Xs)), Xs])
    xl_design = np.concatenate([[1.0], xls])

    # Ridge solve
    n_features = X_design.shape[1]
    I = np.eye(n_features)
    I[0, 0] = 0.0  # don't penalize intercept

    A = X_design.T @ X_design + alpha * I
    b = X_design.T @ y
    beta = np.linalg.solve(A, b)

    pred = float(xl_design @ beta)
    return pred


def predict_forecast(df_ohlcv: pd.DataFrame, alpha: float = 10.0) -> Forecast:
    """
    Train on the available history and produce next 1D and 5D forecasts.
    Outputs both percent and raw moves from the latest close.
    """
    price = _get_price_series(df_ohlcv)
    feat = _make_feature_frame(price)

    if len(feat) < 60:
        raise ValueError("Not enough rows after feature engineering (need ~60+).")

    feature_cols = ["ret_1", "ret_2", "ret_5", "ma_ratio_5_20", "vol_14"]

    X = feat[feature_cols].to_numpy(dtype=float)

    y1 = feat["y_1d"].to_numpy(dtype=float)
    y5 = feat["y_5d"].to_numpy(dtype=float)

    x_last = X[-1, :].copy()
    last_close = float(price.iloc[-1])

    pred_1d = _ridge_fit_predict(X, y1, x_last, alpha=alpha)  # as decimal return
    pred_5d = _ridge_fit_predict(X, y5, x_last, alpha=alpha)

    pred_1d_pct = pred_1d * 100.0
    pred_5d_pct = pred_5d * 100.0

    pred_1d_move = last_close * pred_1d
    pred_5d_move = last_close * pred_5d

    return Forecast(
        last_close=last_close,
        pred_1d_return_pct=float(pred_1d_pct),
        pred_1d_move_raw=float(pred_1d_move),
        pred_1d_target_close=float(last_close + pred_1d_move),
        pred_5d_return_pct=float(pred_5d_pct),
        pred_5d_move_raw=float(pred_5d_move),
        pred_5d_target_close=float(last_close + pred_5d_move),
    )


def predict_many(
    data: dict[str, pd.DataFrame],
    alpha: float = 10.0,
) -> dict[str, Forecast]:
    """
    Run predictions for many tickers.
    """
    out: dict[str, Forecast] = {}
    for ticker, df in data.items():
        out[ticker] = predict_forecast(df, alpha=alpha)
    return out