# src/models/feature_importance.py
from __future__ import annotations

import pandas as pd


def xgb_feature_importance_df(
    model,
    feature_cols: list[str],
    importance_type: str = "gain",
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Returns a tidy DataFrame of XGBoost feature importance.

    importance_type:
      - "gain" (recommended)
      - "weight"
      - "cover"
      - "total_gain"
      - "total_cover"
    """
    booster = getattr(model, "get_booster", None)
    if booster is None:
        raise ValueError("Model does not look like an XGBRegressor (missing get_booster).")

    booster = model.get_booster()

    # XGBoost uses feature names like f0, f1... unless you pass feature_names explicitly.
    score = booster.get_score(importance_type=importance_type)  # dict like {"f0": 1.23, ...}
    if not score:
        return pd.DataFrame(columns=["feature", "importance", "pct"])

    rows = []
    for k, v in score.items():
        # k is like "f12"
        if k.startswith("f"):
            idx = int(k[1:])
            name = feature_cols[idx] if idx < len(feature_cols) else k
        else:
            name = k
        rows.append((name, float(v)))

    df = pd.DataFrame(rows, columns=["feature", "importance"]).sort_values("importance", ascending=False)
    total = df["importance"].sum()
    df["pct"] = (df["importance"] / total) * 100.0 if total != 0 else 0.0

    return df.head(top_n).reset_index(drop=True)