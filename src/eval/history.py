from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd


Horizon = Literal["1D", "5D"]


@dataclass(frozen=True)
class PredRow:
    prediction_date: str  # YYYY-MM-DD (date the prediction was MADE, at close)
    ticker: str
    horizon: Horizon
    last_close: float
    pred_return_pct: float

    @property
    def pred_move_raw(self) -> float:
        return self.last_close * (self.pred_return_pct / 100.0)

    @property
    def pred_target_close(self) -> float:
        return self.last_close + self.pred_move_raw


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "prediction_date",
                "ticker",
                "horizon",
                "last_close",
                "pred_return_pct",
                "pred_move_raw",
                "pred_target_close",
                "actual_close",
                "actual_return_pct",
                "actual_move_raw",
                "direction_correct",
                "abs_error_pct",
                "abs_error_raw",
                "evaluated_date",
            ]
        )

    df = pd.read_csv(path)
    # normalize types
    df["prediction_date"] = df["prediction_date"].astype(str)
    df["ticker"] = df["ticker"].astype(str)
    df["horizon"] = df["horizon"].astype(str)
    return df


def _unique_key(df: pd.DataFrame) -> pd.Series:
    return df["prediction_date"].astype(str) + "|" + df["ticker"].astype(str) + "|" + df["horizon"].astype(str)


def append_predictions_dedup(history_path: str, rows: list[PredRow]) -> pd.DataFrame:
    """
    Append today's predictions, but never duplicate the same (date, ticker, horizon).
    Safe to run multiple times per day.
    """
    path = Path(history_path)
    _ensure_parent(path)

    hist = _load_history(path)

    new_df = pd.DataFrame(
        [
            {
                "prediction_date": r.prediction_date,
                "ticker": r.ticker,
                "horizon": r.horizon,
                "last_close": r.last_close,
                "pred_return_pct": r.pred_return_pct,
                "pred_move_raw": r.pred_move_raw,
                "pred_target_close": r.pred_target_close,
                # evaluation fields (filled later)
                "actual_close": pd.NA,
                "actual_return_pct": pd.NA,
                "actual_move_raw": pd.NA,
                "direction_correct": pd.NA,
                "abs_error_pct": pd.NA,
                "abs_error_raw": pd.NA,
                "evaluated_date": pd.NA,
            }
            for r in rows
        ]
    )

    # Combine + de-duplicate by key, keeping the newest row
    combined = pd.concat([hist, new_df], ignore_index=True)

    key = _unique_key(combined)
    combined = combined.loc[~key.duplicated(keep="last")].copy()

    # stable ordering
    combined.sort_values(["prediction_date", "ticker", "horizon"], inplace=True)

    combined.to_csv(path, index=False)
    return combined


def evaluate_pending_close_to_close(
    history_path: str,
    close_prices_today: dict[str, float],
    today_date: str,
) -> pd.DataFrame:
    """
    For any rows where actual_close is missing AND the horizon is 1D,
    fill actual outcomes using today's close (Close→Close).

    Close→Close evaluation rule for 1D:
      prediction_date = day we made prediction at close
      actual_close = today's close (day after prediction_date close)
    """
    path = Path(history_path)
    _ensure_parent(path)

    hist = _load_history(path)
    if hist.empty:
        return hist

    # Evaluate only 1D rows that haven't been evaluated yet
    mask = (hist["horizon"] == "1D") & (hist["actual_close"].isna())

    # If none pending, still save/return
    if not mask.any():
        hist.to_csv(path, index=False)
        return hist

    # For each pending row, compute actuals if we have today's close for that ticker
    for idx in hist[mask].index:
        ticker = str(hist.at[idx, "ticker"])
        if ticker not in close_prices_today:
            continue

        last_close = float(hist.at[idx, "last_close"])
        pred_pct = float(hist.at[idx, "pred_return_pct"])

        actual_close = float(close_prices_today[ticker])
        actual_move = actual_close - last_close
        actual_pct = (actual_move / last_close) * 100.0 if last_close != 0 else 0.0

        pred_move = last_close * (pred_pct / 100.0)
        direction_correct = (pred_move == 0 and actual_move == 0) or (pred_move * actual_move > 0)

        hist.at[idx, "actual_close"] = actual_close
        hist.at[idx, "actual_return_pct"] = actual_pct
        hist.at[idx, "actual_move_raw"] = actual_move
        hist.at[idx, "direction_correct"] = bool(direction_correct)
        hist.at[idx, "abs_error_pct"] = abs(actual_pct - pred_pct)
        hist.at[idx, "abs_error_raw"] = abs(actual_move - pred_move)
        hist.at[idx, "evaluated_date"] = today_date

    hist.to_csv(path, index=False)
    return hist