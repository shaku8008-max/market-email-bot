from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.models.features import build_features_and_target, split_train_test_timewise
from src.models.feature_importance import xgb_feature_importance_df


@dataclass(frozen=True)
class WalkForwardConfig:
    min_train_size: int = 126
    test_window: int = 10
    step: int = 10
    rolling_train_size: int = 252   # NEW: use only the latest ~1 year per fold


@dataclass(frozen=True)
class HorizonPrediction:
    pred_return: float
    pred_move_raw: float
    pred_target_close: float


@dataclass(frozen=True)
class RegressionResult:
    model_type: str

    mae: float
    rmse: float
    directional_accuracy: float

    wf_mae: float
    wf_rmse: float
    wf_directional_accuracy: float
    wf_folds: int

    pred_1d: HorizonPrediction
    pred_3d: HorizonPrediction
    pred_5d: HorizonPrediction

    top_features: list[str]
    top_features_text: str

    regime: str
    ensemble_weights: str


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _directional_accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def _make_model(model_type: str):
    model_type = model_type.lower().strip()

    if model_type == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)

    if model_type == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )

    if model_type == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model type: {model_type}")


def _ensemble_weights(regime: str) -> tuple[float, float, float]:
    regime = (regime or "").upper()

    if regime == "TREND_UP":
        return 0.15, 0.20, 0.65

    if regime == "TREND_DOWN":
        return 0.20, 0.30, 0.50

    if regime == "PANIC":
        return 0.10, 0.35, 0.55

    if regime == "MEAN_REVERT":
        return 0.45, 0.25, 0.30

    return 0.33, 0.33, 0.34


def _predict_ensemble(models: tuple[object, object, object], weights: tuple[float, float, float], X: np.ndarray) -> np.ndarray:
    model_ridge, model_rf, model_xgb = models
    w_ridge, w_rf, w_xgb = weights

    pred = (
        w_ridge * model_ridge.predict(X)
        + w_rf * model_rf.predict(X)
        + w_xgb * model_xgb.predict(X)
    )
    return pred


def walk_forward_validate(
    feat: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    regime: str,
    cfg: WalkForwardConfig,
) -> tuple[float, float, float, int]:
    """
    Rolling walk-forward with a rolling training window.
    For each fold, train on only the most recent `rolling_train_size` rows.
    """
    n = len(feat)
    min_needed = cfg.min_train_size + cfg.test_window
    if n <= min_needed:
        return np.nan, np.nan, np.nan, 0

    y_all = feat[target_col].values
    X_all = feat[feature_cols].values

    weights = _ensemble_weights(regime)

    maes = []
    rmses = []
    dirs = []

    folds = 0
    train_end = cfg.min_train_size

    while True:
        test_start = train_end
        test_end = test_start + cfg.test_window

        if test_end > n:
            break

        train_start = max(0, train_end - cfg.rolling_train_size)

        X_train = X_all[train_start:train_end]
        y_train = y_all[train_start:train_end]

        X_test = X_all[test_start:test_end]
        y_test = y_all[test_start:test_end]

        model_ridge = _make_model("ridge")
        model_rf = _make_model("rf")
        model_xgb = _make_model("xgb")

        model_ridge.fit(X_train, y_train)
        model_rf.fit(X_train, y_train)
        model_xgb.fit(X_train, y_train)

        y_pred = _predict_ensemble(
            (model_ridge, model_rf, model_xgb),
            weights,
            X_test,
        )

        maes.append(float(mean_absolute_error(y_test, y_pred)))
        rmses.append(_rmse(y_test, y_pred))
        dirs.append(_directional_accuracy(y_test, y_pred))

        folds += 1
        train_end += cfg.step

    if folds == 0:
        return np.nan, np.nan, np.nan, 0

    return float(np.mean(maes)), float(np.mean(rmses)), float(np.mean(dirs)), folds


def _fit_horizon_models(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[object, object, object]:
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values

    model_ridge = _make_model("ridge")
    model_rf = _make_model("rf")
    model_xgb = _make_model("xgb")

    model_ridge.fit(X_train, y_train)
    model_rf.fit(X_train, y_train)
    model_xgb.fit(X_train, y_train)

    return model_ridge, model_rf, model_xgb


def _make_horizon_prediction(
    models: tuple[object, object, object],
    weights: tuple[float, float, float],
    last_feat: np.ndarray,
    last_close: float,
) -> HorizonPrediction:
    pred = float(_predict_ensemble(models, weights, last_feat.reshape(1, -1))[0])

    # Noise reduction: clamp extreme daily/short-horizon forecasts
    pred = float(np.clip(pred, -0.05, 0.05))

    move_raw = pred * last_close
    target_close = last_close * (1.0 + pred)

    return HorizonPrediction(
        pred_return=pred,
        pred_move_raw=float(move_raw),
        pred_target_close=float(target_close),
    )


def train_and_forecast_next_day(
    df_ohlcv,
    test_size: float = 0.3,
    wf_cfg: WalkForwardConfig | None = None,
) -> RegressionResult:
    if wf_cfg is None:
        wf_cfg = WalkForwardConfig()

    feat = build_features_and_target(df_ohlcv)

    feature_cols = [
        c for c in feat.columns
        if c not in {
            "target_1d",
            "target_3d",
            "target_5d",
            "close",
            "regime_label",
        }
    ]

    regime = str(feat.iloc[-1]["regime_label"])
    weights = _ensemble_weights(regime)
    w_ridge, w_rf, w_xgb = weights

    # Time split
    train_df, test_df = split_train_test_timewise(feat, test_size=test_size)

    # NEW: rolling train window for the final fitted models too
    train_df = train_df.tail(wf_cfg.rolling_train_size).copy()

    X_test = test_df[feature_cols].values
    y_test_1d = test_df["target_1d"].values

    # -------------------------
    # Fit 1D models
    # -------------------------
    models_1d = _fit_horizon_models(train_df, feature_cols, "target_1d")
    y_pred_1d = _predict_ensemble(models_1d, weights, X_test)

    mae = float(mean_absolute_error(y_test_1d, y_pred_1d))
    rmse = _rmse(y_test_1d, y_pred_1d)
    dir_acc = _directional_accuracy(y_test_1d, y_pred_1d)

    # -------------------------
    # Walk-forward on 1D
    # -------------------------
    wf_mae, wf_rmse, wf_dir, wf_folds = walk_forward_validate(
        feat=feat,
        feature_cols=feature_cols,
        target_col="target_1d",
        regime=regime,
        cfg=wf_cfg,
    )

    # -------------------------
    # Fit 3D and 5D models
    # -------------------------
    models_3d = _fit_horizon_models(train_df, feature_cols, "target_3d")
    models_5d = _fit_horizon_models(train_df, feature_cols, "target_5d")

    # -------------------------
    # Forecast from latest row
    # -------------------------
    last_feat = feat.iloc[-1][feature_cols].values.astype(float)
    last_close = float(feat.iloc[-1]["close"])

    pred_1d = _make_horizon_prediction(models_1d, weights, last_feat, last_close)
    pred_3d = _make_horizon_prediction(models_3d, weights, last_feat, last_close)
    pred_5d = _make_horizon_prediction(models_5d, weights, last_feat, last_close)

    # -------------------------
    # Feature importance from XGB (1D model)
    # -------------------------
    model_xgb_1d = models_1d[2]
    imp_df = xgb_feature_importance_df(
        model_xgb_1d,
        feature_cols,
        importance_type="gain",
        top_n=12,
    )

    top_features = []
    top_text = ""

    if not imp_df.empty:
        top_features = imp_df["feature"].tolist()
        top_text = ", ".join(
            [f"{r.feature} ({r.pct:.1f}%)" for r in imp_df.itertuples(index=False)]
        )

    return RegressionResult(
        model_type="ensemble_rolling",
        mae=mae,
        rmse=rmse,
        directional_accuracy=dir_acc,
        wf_mae=wf_mae,
        wf_rmse=wf_rmse,
        wf_directional_accuracy=wf_dir,
        wf_folds=int(wf_folds),
        pred_1d=pred_1d,
        pred_3d=pred_3d,
        pred_5d=pred_5d,
        top_features=top_features,
        top_features_text=top_text,
        regime=regime,
        ensemble_weights=f"ridge={w_ridge:.2f}, rf={w_rf:.2f}, xgb={w_xgb:.2f}",
    )


def train_many_indexes(data, test_size=0.3, wf_cfg=None):
    if wf_cfg is None:
        wf_cfg = WalkForwardConfig()

    results = {}

    for ticker, df in data.items():
        results[ticker] = train_and_forecast_next_day(
            df,
            test_size=test_size,
            wf_cfg=wf_cfg,
        )

    return results