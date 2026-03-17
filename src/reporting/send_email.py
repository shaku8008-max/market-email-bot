from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


EMAIL_COLUMNS = [
    "Ticker",
    "Regime",
    "Last Close",
    "Pred 1D %",
    "Pred 5D %",
]


def _prepare_email_table(combined: pd.DataFrame) -> pd.DataFrame:
    df = combined.copy()
    keep_cols = [c for c in EMAIL_COLUMNS if c in df.columns]
    df = df[keep_cols].copy()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(2)

    return df


def _color_for_value(v: Any) -> str:
    try:
        x = float(v)
    except Exception:
        return "#222222"
    if x > 0:
        return "#188038"
    if x < 0:
        return "#d93025"
    return "#5f6368"


def _styled_summary_table_html(df: pd.DataFrame) -> str:
    headers = "".join(
        f"<th style='background:#f5f7fa;border:1px solid #dcdcdc;padding:10px 8px;text-align:center;'>{col}</th>"
        for col in df.columns
    )

    rows_html = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            style = "border:1px solid #dcdcdc;padding:10px 8px;text-align:center;vertical-align:middle;"

            if col in {"Pred 1D %", "Pred 5D %"}:
                style += f"color:{_color_for_value(val)};font-weight:600;"
            elif col == "Regime":
                regime = str(val).upper()
                if "UP" in regime or regime == "BULL":
                    style += "color:#188038;font-weight:600;"
                elif "DOWN" in regime or regime in {"BEAR", "PANIC"}:
                    style += "color:#d93025;font-weight:600;"
                else:
                    style += "color:#b06000;font-weight:600;"

            cells.append(f"<td style='{style}'>{val}</td>")

        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    return f"""
    <table style="border-collapse:collapse;width:100%;table-layout:fixed;font-size:14px;margin-top:10px;">
      <thead><tr>{headers}</tr></thead>
      <tbody>
        {''.join(rows_html)}
      </tbody>
    </table>
    """


def _extract_close_series(df: pd.DataFrame) -> pd.Series:
    data = df.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Adj Close" in data.columns:
        s = data["Adj Close"]
    elif "Close" in data.columns:
        s = data["Close"]
    else:
        raise ValueError("No Close/Adj Close column found.")

    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]

    return s.dropna().astype(float)


def _make_chart_png_bytes(
    ticker: str,
    df: pd.DataFrame,
    regime: str,
    pred_1d_pct: float | None,
    pred_3d_pct: float | None,
    pred_5d_pct: float | None,
) -> bytes:
    s = _extract_close_series(df)
    hist = s.tail(30).copy()

    last_close = float(hist.iloc[-1])

    # Forecast points anchored from today's close
    x_actual = list(range(len(hist)))
    x0 = x_actual[-1]

    x_forecast = [x0, x0 + 1, x0 + 3, x0 + 5]
    y_forecast = [last_close]

    y1 = last_close if pred_1d_pct is None else last_close * (1 + pred_1d_pct / 100.0)
    y3 = last_close if pred_3d_pct is None else last_close * (1 + pred_3d_pct / 100.0)
    y5 = last_close if pred_5d_pct is None else last_close * (1 + pred_5d_pct / 100.0)

    y_forecast.extend([y1, y3, y5])

    direction_color = "#188038"
    if pred_5d_pct is not None and pred_5d_pct < 0:
        direction_color = "#d93025"
    elif pred_5d_pct is None and pred_1d_pct is not None and pred_1d_pct < 0:
        direction_color = "#d93025"

    actual_color = "#1a3c6e"
    fill_color = "#dbe7f5"
    divider_color = "#b0b7c3"

    fig, ax = plt.subplots(figsize=(7.4, 3.8), facecolor="white")
    ax.set_facecolor("white")

    # Actual series
    ax.plot(
        x_actual,
        hist.values,
        color=actual_color,
        linewidth=2.4,
        solid_capstyle="round",
        label="Actual",
        zorder=3,
    )
    ax.fill_between(
        x_actual,
        hist.values,
        min(hist.values) * 0.995,
        color=fill_color,
        alpha=0.45,
        zorder=1,
    )

    # Divider
    ax.axvline(
        x=x0,
        color=divider_color,
        linestyle=(0, (3, 3)),
        linewidth=1.0,
        alpha=0.9,
        zorder=2,
    )

    # Forecast dotted line
    ax.plot(
        x_forecast,
        y_forecast,
        color=direction_color,
        linewidth=2.6,
        linestyle=(0, (1.2, 2.4)),
        solid_capstyle="round",
        label="Forecast",
        zorder=4,
    )

    # Forecast markers
    ax.scatter(
        x_forecast[1:],
        y_forecast[1:],
        s=30,
        color=direction_color,
        edgecolors="white",
        linewidths=0.9,
        zorder=5,
    )

    # Title + subtitle
    title = f"{ticker} | Last 30 days + forecast"
    sub = f"Regime: {regime} | 1D: {pred_1d_pct:.2f}% | 5D: {pred_5d_pct:.2f}%"
    ax.set_title(title, fontsize=11.5, loc="left", pad=16, color="#111111")
    ax.text(
        0.0,
        1.02,
        sub,
        transform=ax.transAxes,
        fontsize=9.3,
        color="#5f6368",
        ha="left",
        va="bottom",
    )

    # Axis styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d0d7de")
    ax.spines["bottom"].set_color("#d0d7de")

    ax.tick_params(axis="x", colors="#5f6368", labelsize=8)
    ax.tick_params(axis="y", colors="#5f6368", labelsize=8)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.22)
    ax.grid(axis="x", visible=False)

    ax.set_xlabel("Trading Days", fontsize=8.5, color="#5f6368")
    ax.set_ylabel("Price", fontsize=8.5, color="#5f6368")

    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    # Tight y range with padding
    y_all = list(hist.values) + y_forecast
    y_min = min(y_all)
    y_max = max(y_all)
    padding = max((y_max - y_min) * 0.12, abs(last_close) * 0.003)
    ax.set_ylim(y_min - padding, y_max + padding)

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _build_html_email(
    title: str,
    combined: pd.DataFrame,
    used_cache: dict[str, bool],
    chart_cids: list[str],
    tickers_for_charts: list[str],
) -> str:
    summary_df = _prepare_email_table(combined)
    summary_table_html = _styled_summary_table_html(summary_df)

    chart_blocks = []
    for ticker, cid in zip(tickers_for_charts, chart_cids):
        chart_blocks.append(
            f"""
            <div style="margin-top:20px;">
              <img src="cid:{cid}" style="width:100%;max-width:720px;border:1px solid #e6e6e6;border-radius:10px;display:block;" />
            </div>
            """
        )

    data_source_rows = "".join(
        f"<tr>"
        f"<td style='padding:8px 12px;border:1px solid #dcdcdc;'>{ticker}</td>"
        f"<td style='padding:8px 12px;border:1px solid #dcdcdc;'>{'CACHE' if was_cache else 'FRESH'}</td>"
        f"</tr>"
        for ticker, was_cache in used_cache.items()
    )

    return f"""
    <html>
    <body style="font-family:Arial,sans-serif;color:#222;line-height:1.45;margin:0;padding:24px;background:#ffffff;">
      <div style="max-width:760px;margin:0 auto;">
        <h1 style="font-size:22px;margin:0 0 8px 0;">{title}</h1>

        <div style="background:#f8f9fb;border:1px solid #e3e7ee;padding:12px 14px;border-radius:8px;margin-top:14px;font-size:13px;">
          <b>How to read:</b><br>
          <span style="color:#188038;font-weight:600;">Green dotted line</span> = upward forecast.<br>
          <span style="color:#d93025;font-weight:600;">Red dotted line</span> = downward forecast.<br>
          <span style="color:#1a3c6e;font-weight:600;">Solid line</span> = actual recent market price.<br>
          Summary table shows only the 5 most important fields.
        </div>

        <h2 style="font-size:17px;margin:28px 0 10px 0;">Daily Summary</h2>
        {summary_table_html}

        <h2 style="font-size:17px;margin:28px 0 10px 0;">Market Charts</h2>
        {''.join(chart_blocks)}

        <h2 style="font-size:17px;margin:28px 0 10px 0;">Data Source</h2>
        <table style="border-collapse:collapse;width:100%;font-size:14px;">
          <thead>
            <tr>
              <th style="background:#f5f7fa;border:1px solid #dcdcdc;padding:10px 8px;text-align:center;">Ticker</th>
              <th style="background:#f5f7fa;border:1px solid #dcdcdc;padding:10px 8px;text-align:center;">Source</th>
            </tr>
          </thead>
          <tbody>
            {data_source_rows}
          </tbody>
        </table>
      </div>
    </body>
    </html>
    """


def send_market_email(
    combined: pd.DataFrame,
    used_cache: dict[str, bool],
    raw_data: dict[str, pd.DataFrame],
    reg_results: dict[str, Any],
    subject: str | None = None,
    sender_email: str | None = None,
    sender_app_password: str | None = None,
    recipient_email: str | None = None,
) -> None:
    sender_email = sender_email or os.getenv("EMAIL_SENDER")
    sender_app_password = sender_app_password or os.getenv("EMAIL_APP_PASSWORD")
    recipient_email = recipient_email or os.getenv("EMAIL_RECIPIENT")

    if not sender_email:
        raise ValueError("Missing EMAIL_SENDER")
    if not sender_app_password:
        raise ValueError("Missing EMAIL_APP_PASSWORD")
    if not recipient_email:
        raise ValueError("Missing EMAIL_RECIPIENT")

    if subject is None:
        subject = "Daily Market Report"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg.set_content("Your email client does not support HTML.")

    tickers_for_charts = list(raw_data.keys())
    chart_cids = []

    for i, ticker in enumerate(tickers_for_charts):
        r = reg_results[ticker]
        cid = f"chart_{i}_{ticker.replace('^', '')}"

        png_bytes = _make_chart_png_bytes(
            ticker=ticker,
            df=raw_data[ticker],
            regime=getattr(r, "regime", "UNKNOWN"),
            pred_1d_pct=r.pred_1d.pred_return * 100.0,
            pred_3d_pct=r.pred_3d.pred_return * 100.0,
            pred_5d_pct=r.pred_5d.pred_return * 100.0,
        )

        chart_cids.append(cid)

        msg.add_related(
            png_bytes,
            maintype="image",
            subtype="png",
            cid=f"<{cid}>",
        )

    html_body = _build_html_email(
        title="Market Email Bot Report",
        combined=combined,
        used_cache=used_cache,
        chart_cids=chart_cids,
        tickers_for_charts=tickers_for_charts,
    )

    msg.add_alternative(html_body, subtype="html")

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, sender_app_password)
        server.send_message(msg)