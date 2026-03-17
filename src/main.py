from __future__ import annotations

import os
import pandas as pd
from dotenv import load_dotenv

from src.config import TICKERS
from src.data.fetch_data import FetchConfig, fetch_many
from src.analytics.metrics import compute_metrics, build_summary_table
from src.models.train_regression import train_many_indexes, WalkForwardConfig
from src.models.features import build_features_and_target
from src.reporting.save_report import save_daily_report
from src.reporting.send_email import send_market_email


def main() -> None:
    load_dotenv()

    default_period = os.getenv("DEFAULT_PERIOD", "2y")
    default_interval = os.getenv("DEFAULT_INTERVAL", "1d")
    cache_dir = os.getenv("CACHE_DIR", "data/cache")
    output_dir = os.getenv("OUTPUT_DIR", "outputs")

    test_split = float(os.getenv("TEST_SPLIT", "0.3"))
    rolling_train_days = int(os.getenv("ROLLING_TRAIN_DAYS", "252"))

    print("Fetching market data...\n")

    cfg = FetchConfig(
        period=default_period,
        interval=default_interval,
        cache_dir=cache_dir,
        max_retries=3,
        retry_sleep_seconds=2.0,
    )

    data, used_cache = fetch_many(
        TICKERS.keys(),
        cfg,
        use_cache=True,
        force_refresh=True,
    )

    results: dict[str, dict[str, float | str]] = {}

    for ticker, df in data.items():
        try:
            results[ticker] = compute_metrics(df)
        except Exception as e:
            results[ticker] = {
                "last_close": float("nan"),
                "change_1d_pct": float("nan"),
                "change_1d_raw": float("nan"),
                "change_5d_pct": float("nan"),
                "change_5d_raw": float("nan"),
                "trend": f"ERROR: {e}",
                "vol_14d_pct": float("nan"),
            }

    snapshot = build_summary_table(results, TICKERS)

    regime_map: dict[str, str] = {}
    for ticker, df in data.items():
        try:
            feat = build_features_and_target(df)
            regime_map[ticker] = str(feat.iloc[-1]["regime_label"])
        except Exception:
            regime_map[ticker] = "UNKNOWN"

    wf_cfg = WalkForwardConfig(
        min_train_size=126,
        test_window=10,
        step=10,
        rolling_train_size=rolling_train_days,
    )

    reg_results = train_many_indexes(
        data,
        test_size=test_split,
        wf_cfg=wf_cfg,
    )

    reg_rows = []

    for ticker in data.keys():
        r = reg_results[ticker]

        reg_rows.append(
            {
                "Ticker": ticker,
                "Model": r.model_type,
                "Regime(Model)": r.regime,
                "Weights": r.ensemble_weights,
                "Model MAE%": r.mae * 100,
                "Model RMSE%": r.rmse * 100,
                "Dir Acc%": r.directional_accuracy * 100,
                "WF MAE%": r.wf_mae * 100,
                "WF RMSE%": r.wf_rmse * 100,
                "WF Dir Acc%": r.wf_directional_accuracy * 100,
                "Pred 1D %": r.pred_1d.pred_return * 100,
                "Pred 1D Raw": r.pred_1d.pred_move_raw,
                "Pred 1D Target": r.pred_1d.pred_target_close,
                "Pred 3D %": r.pred_3d.pred_return * 100,
                "Pred 3D Raw": r.pred_3d.pred_move_raw,
                "Pred 3D Target": r.pred_3d.pred_target_close,
                "Pred 5D %": r.pred_5d.pred_return * 100,
                "Pred 5D Raw": r.pred_5d.pred_move_raw,
                "Pred 5D Target": r.pred_5d.pred_target_close,
                "Top Signals": r.top_features_text,
            }
        )

    reg_df = pd.DataFrame(reg_rows)
    combined = snapshot.merge(reg_df, on="Ticker", how="left")
    combined["Regime"] = combined["Ticker"].map(regime_map)

    preferred_order = [
        "Ticker",
        "Name",
        "Regime",
        "Last Close",
        "Pred 1D %",
        "Pred 5D %",
    ]

    existing_cols = [c for c in preferred_order if c in combined.columns]
    remaining_cols = [c for c in combined.columns if c not in existing_cols]
    combined = combined[existing_cols + remaining_cols]

    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", None)

    print("\n=== Market Snapshot + Model Forecast ===\n")
    print(combined.to_string(index=False))

    print("\n=== Data Source ===")
    for t, was_cache in used_cache.items():
        print(f"{t:>6}  ->  {'CACHE' if was_cache else 'FRESH'}")

    out_path = save_daily_report(combined, used_cache, output_dir=output_dir)
    print(f"\nSaved daily report to: {out_path}")

    try:
        send_market_email(
            combined=combined,
            used_cache=used_cache,
            raw_data=data,
            reg_results=reg_results,
            subject="Daily Market Report",
        )
        print("Email sent successfully.")
    except Exception as e:
        print("Email sending failed:", e)


if __name__ == "__main__":
    main()