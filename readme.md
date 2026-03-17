# Market Email Bot

A Python-based market research and reporting tool that:

- fetches daily index data
- engineers quantitative features
- detects market regimes
- trains ensemble ML models
- generates 1D / 3D / 5D forecasts
- saves a daily report
- emails a formatted summary with charts

This project is designed as a **quant-style research pipeline**, not a live trading system.

---

## What this project does

The bot tracks major indexes such as:

- **S&P 500** (`^GSPC`)
- **Dow Jones** (`^DJI`)
- **Nasdaq 100** (`^NDX`)
- **S&P Global 100 proxy** (`IOO`)

Each run:

1. downloads recent market data
2. computes market snapshot metrics
3. builds technical and regime features
4. trains machine learning models
5. predicts future returns
6. saves a report
7. emails a compact dashboard summary

---

## Core outputs

The system produces:

- **1-day forecast**
- **3-day forecast**
- **5-day forecast**

For each horizon it estimates:

- predicted return %
- predicted raw move
- predicted target level

It also shows:

- current regime
- recent volatility
- model error
- directional accuracy
- top signals driving the model

---

## Project structure

```text
market-email-bot/
├── .env
├── requirements.txt
├── outputs/
├── src/
│   ├── main.py
│   ├── config.py
│   ├── data/
│   │   └── fetch_data.py
│   ├── analytics/
│   │   └── metrics.py
│   ├── models/
│   │   ├── features.py
│   │   ├── train_regression.py
│   │   └── feature_importance.py
│   └── reporting/
│       ├── save_report.py
│       └── send_email.py


How the pipeline works
1. Data collection

The bot downloads historical index data using Yahoo Finance.

It uses:

daily interval

local caching

retry logic

This gives the raw OHLCV data used by the rest of the system.

2. Market snapshot metrics

The analytics module calculates:

Last Close

1D %

1D Raw

5D %

5D Raw

Trend

Vol(14D)%

These are descriptive metrics.

They tell you what the market has already done.

3. Feature engineering

The model builds predictive features such as:

Momentum features

ret_1

ret_5

ret_10

ret_20

Mean reversion features

dist_ma20

dist_ma50

z20

z50

rsi_14

bb_pos_20

Volatility features

vol_14

vol_20

vol_60

vol_ratio_20_60

bb_width_20

Volume and trend strength

vol_spike_20

adx_14

Market regime features

drawdown_126

high_vol_flag

crash_flag

trend_up_flag

trend_down_flag

mean_revert_flag

regime_id

4. Regime detection

The bot classifies the market into one of these regimes:

TREND_UP

TREND_DOWN

PANIC

MEAN_REVERT

This is based on a combination of:

moving averages

ADX

drawdown

volatility percentile

Regime meanings

TREND_UP
The market is in a strong upward trend.

TREND_DOWN
The market is in a strong downward trend.

PANIC
The market is under crash-like stress with high volatility and drawdown.

MEAN_REVERT
The market is not strongly trending and tends to revert after short-term moves.

5. Model training

The bot uses an ensemble of three models:

Ridge Regression

Random Forest

XGBoost

These are combined using regime-based weights.

Why use an ensemble?

Each model captures different structure:

Ridge → stable linear relationships

Random Forest → nonlinear interactions

XGBoost → stronger boosted nonlinear patterns

Averaging them reduces noise and makes the forecast more stable.

6. Regime-based weighting

The ensemble weights change depending on regime.

Example:

TREND_UP / TREND_DOWN → more weight to XGBoost

MEAN_REVERT → more weight to Ridge

PANIC → more weight to Random Forest + XGBoost

This makes the system more adaptive.

7. Rolling training window

The model does not train on all historical data equally.

Instead, it uses a rolling training window.

Example:

train on the most recent ~252 trading days

re-train every run

This helps the model adapt to the current market regime instead of learning stale patterns from older conditions.

8. Walk-forward validation

The system measures robustness using walk-forward validation.

Instead of testing once, it repeatedly:

trains on a rolling historical window

tests on the next block of data

repeats across time

This gives more realistic performance estimates.

How to read the report
Summary columns
Regime

The current detected market regime.

Last Close

Most recent closing level.

1D %

Actual one-day return.

5D %

Actual five-day return.

Vol(14D)%

Recent daily volatility estimate.

Model quality columns
Model MAE%

Average absolute prediction error on the main test split.

Lower is better.

Model RMSE%

Error measure that penalizes large misses more heavily.

Lower is better.

Dir Acc%

Directional accuracy on the main test split.

This measures whether the model got up vs down correct.

WF MAE%

Walk-forward mean absolute error.

WF RMSE%

Walk-forward root mean squared error.

WF Dir Acc%

Walk-forward directional accuracy.

This is the most trustworthy directional metric.

Forecast columns
Pred 1D %

Forecast return over the next 1 trading day.

Pred 3D %

Forecast return over the next 3 trading days.

Pred 5D %

Forecast return over the next 5 trading days.

Pred 1D Raw, Pred 3D Raw, Pred 5D Raw

Predicted raw point or price move.

Pred 1D Target, Pred 3D Target, Pred 5D Target

Predicted target level for that horizon.

Top Signals

These are the most important model features from XGBoost.

Examples:

ret_5 → short-term momentum

rsi_14 → overbought/oversold condition

drawdown_126 → stress from recent peak

adx_14 → trend strength

vol_spike_20 → unusual volume activity

These tell you what the model is relying on.

How to interpret the forecasts
1D forecast

Best for very short-term bias.

Most noisy.

3D forecast

Usually more stable than 1D.

Good for short swing interpretation.

5D forecast

Usually the cleanest signal.

Best for a weekly directional view.

How to interpret regime + forecast together
TREND_UP + positive forecast

Bullish continuation signal.

TREND_DOWN + negative forecast

Bearish continuation signal.

MEAN_REVERT + negative after a bounce

Possible pullback after short-term strength.

MEAN_REVERT + positive after a selloff

Possible rebound setup.

PANIC + strongly negative forecast

High-risk stress environment.

Charts in the email

Each chart shows:

solid line = actual recent market price

dotted line = predicted path

green dotted line = bullish forecast

red dotted line = bearish forecast

The charts help you visually compare:
where the index has been
where the model expects it to go next
Email summary fields
The email intentionally shows only the most important columns:
Ticker
Regime
Last Close
Pred 1D %
Pred 5D %
This keeps the email clean and readable.
The full report remains saved locally in outputs/.

Running the bot
Run manually:
python -m src.main