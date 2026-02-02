import numpy as np
import pandas as pd
from typing import Tuple


def generate_bias_and_explanation(row: pd.Series) -> Tuple[str, str]:
    close = row["Close"]
    ma20 = row.get("MA_20", np.nan)
    ma50 = row.get("MA_50", np.nan)
    z = row.get("ZScore_Close", np.nan)
    rvol = row.get("RVOL", np.nan)
    roc10 = row.get("ROC_10", np.nan)
    mom10 = row.get("Momentum_10", np.nan)
    vwap_d = row.get("VWAP_Daily", np.nan)
    vwap_w = row.get("VWAP_Weekly", np.nan)

    reasons = []

    if close > ma20 and close > ma50:
        trend = "bullish"
        reasons.append("Price is above both the 20-day and 50-day moving averages (uptrend).")
    elif close < ma20 and close < ma50:
        trend = "bearish"
        reasons.append("Price is below both the 20-day and 50-day moving averages (downtrend).")
    else:
        trend = "mixed"
        reasons.append("Price is between key moving averages, suggesting a mixed or transitioning trend.")

    if z > 1.5:
        reasons.append(f"Z-score is {z:.2f}, showing price is extended above its recent mean.")
    elif z < -1.5:
        reasons.append(f"Z-score is {z:.2f}, showing price is extended below its recent mean.")
    else:
        reasons.append(f"Z-score is {z:.2f}, near its recent mean (no major extension).")

    if roc10 > 0 and mom10 > 0:
        momentum = "up"
        reasons.append("10-day ROC and momentum are positive, confirming upward pressure.")
    elif roc10 < 0 and mom10 < 0:
        momentum = "down"
        reasons.append("10-day ROC and momentum are negative, confirming downward pressure.")
    else:
        momentum = "flat"
        reasons.append("Momentum signals are mixed, not clearly favoring buyers or sellers.")

    if rvol > 1.3:
        reasons.append(f"RVOL is {rvol:.2f}, indicating elevated participation and conviction.")
    elif rvol < 0.7:
        reasons.append(f"RVOL is {rvol:.2f}, indicating lighter-than-normal volume.")
    else:
        reasons.append(f"RVOL is {rvol:.2f}, suggesting typical participation.")

    if not np.isnan(vwap_d):
        if close > vwap_d:
            reasons.append("Price is above daily VWAP, showing buyers in control intraday.")
        else:
            reasons.append("Price is below daily VWAP, showing sellers in control intraday.")
    if not np.isnan(vwap_w):
        if close > vwap_w:
            reasons.append("Price is above weekly VWAP, aligned with higher-timeframe buying.")
        else:
            reasons.append("Price is below weekly VWAP, aligned with higher-timeframe selling.")

    # compute a simple intensity/speed score to convert into adjectives
    score = 0
    if abs(z) >= 1.5:
        score += 2
    elif abs(z) >= 0.75:
        score += 1
    if abs(roc10) >= 0.02:
        score += 1
    if rvol > 1.3:
        score += 1
    vol = row.get("Volatility", np.nan)
    if not np.isnan(vol) and vol > 0.02:
        score += 1

    # map score (0-5) to adjective
    if score >= 4:
        adj = "Very Fast"
    elif score == 3:
        adj = "Fast"
    elif score == 2:
        adj = "Moderate"
    elif score == 1:
        adj = "Slow"
    else:
        adj = "Very Slow"

    # determine direction
    if trend == "bullish" and momentum == "up":
        direction = "Bullish"
    elif trend == "bearish" and momentum == "down":
        direction = "Bearish"
    else:
        # fallback to z-score sign
        direction = "Bullish" if z >= 0 else "Bearish"

    bias = f"{adj} {direction}"

    # trade suggestion logic
    suggestion = "No suggestion"
    if not np.isnan(z) and abs(z) >= 2.0:
        suggestion = "Avoid trading — price is overextended (mean-reversion risk)."
    elif rvol < 0.7:
        suggestion = "Avoid trading — low participation/low conviction."
    elif (rvol > 1.3 and not np.isnan(vol) and vol > 0.02) or score >= 3:
        suggestion = "Consider trading with reduced size; favor momentum or trend-follow entries."
    else:
        suggestion = "Wait for clear opening prints or VWAP confirmation before trading."

    explanation = (
        f"Daily Bias: {bias}. Suggestion: {suggestion} "
        + " ".join(reasons)
    )
    return bias, explanation


def classify_volatility(value: float, vol_history: pd.Series | None = None) -> str:
    """Classify a volatility value as Low/Normal/High.

    If `vol_history` is provided, use empirical percentiles (25/75). Otherwise
    use simple absolute thresholds.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Unknown"
    if vol_history is not None and len(vol_history.dropna()) >= 10:
        p25 = vol_history.quantile(0.25)
        p75 = vol_history.quantile(0.75)
        if value >= p75:
            return "High"
        elif value <= p25:
            return "Low"
        else:
            return "Normal"
    # fallback thresholds (these are coarse and depend on data scale)
    if value > 0.02:
        return "High"
    elif value < 0.01:
        return "Low"
    return "Normal"


def predict_next_day_bias(
    row: pd.Series,
    vol_history: pd.Series | None = None,
    asia_stats: dict | None = None,
    london_stats: dict | None = None,
    forexfactory: dict | None = None,
) -> Tuple[str, str, str]:
    """Predict a bias for the next trading day based on the provided day's stats.

    Returns (bias_label, explanation, volatility_label).
    """
    # Leverage the existing generator for baseline reasons
    base_bias, base_expl = generate_bias_and_explanation(row)

    # Determine open-specific cues: previous close vs VWAPs and moving averages
    close = row.get("Close", np.nan)
    ma20 = row.get("MA_20", np.nan)
    ma50 = row.get("MA_50", np.nan)
    vwap_d = row.get("VWAP_Daily", np.nan)
    vwap_w = row.get("VWAP_Weekly", np.nan)
    z = row.get("ZScore_Close", np.nan)
    rvol = row.get("RVOL", np.nan)

    cues = []
    if not np.isnan(close) and not np.isnan(ma20):
        cues.append(f"Close {close:.2f} vs MA20 {ma20:.2f}")
    if not np.isnan(close) and not np.isnan(ma50):
        cues.append(f"Close vs MA50 {ma50:.2f}")
    if not np.isnan(vwap_d):
        cues.append(f"Close {'above' if close>vwap_d else 'below'} daily VWAP")
    if not np.isnan(vwap_w):
        cues.append(f"Close {'above' if close>vwap_w else 'below'} weekly VWAP")

    # incorporate session stats into cues and influence
    if asia_stats:
        a_ret = asia_stats.get("return")
        a_vol = asia_stats.get("volume")
        if a_ret is not None:
            cues.append(f"Asia session return {a_ret:.3f}")
            if a_ret > 0.005:
                cues.append("Asia session showed bullish bias.")
            elif a_ret < -0.005:
                cues.append("Asia session showed bearish bias.")
        if a_vol is not None:
            cues.append(f"Asia session volume {int(a_vol)}")
    if london_stats:
        l_ret = london_stats.get("return")
        l_vol = london_stats.get("volume")
        if l_ret is not None:
            cues.append(f"London session return {l_ret:.3f}")
            if l_ret > 0.005:
                cues.append("London session showed bullish bias.")
            elif l_ret < -0.005:
                cues.append("London session showed bearish bias.")
        if l_vol is not None:
            cues.append(f"London session volume {int(l_vol)}")

    # Volatility classification
    vol_label = classify_volatility(row.get("Volatility", np.nan), vol_history)

    # incorporate ForexFactory signals if provided (or available on `row`)
    ff_selected = None
    ff_forecast = None
    try:
        if forexfactory and isinstance(forexfactory, dict):
            ff_selected = forexfactory.get("selected")
            ff_forecast = forexfactory.get("forecast")
    except Exception:
        ff_selected = None
        ff_forecast = None

    # fallback: check for FF columns in row
    if ff_selected is None:
        if row.get("FF_EventScore") is not None:
            ff_selected = {
                "score": row.get("FF_EventScore"),
                "high": row.get("FF_HighImpactCount"),
                "mid": row.get("FF_MidImpactCount"),
            }

    # Add FF cues
    if ff_selected:
        try:
            sc = ff_selected.get("score")
            hi = ff_selected.get("high", 0) or 0
            mi = ff_selected.get("mid", 0) or 0
            cues.append(f"ForexFactory (selected-date) score {sc} (high:{int(hi)}, mid:{int(mi)})")
            if int(hi) > 0:
                cues.append("High-impact economic events on selected date — expect higher volatility.")
        except Exception:
            pass
    if ff_forecast:
        try:
            sc = ff_forecast.get("score")
            hi = ff_forecast.get("high", 0) or 0
            mi = ff_forecast.get("mid", 0) or 0
            cues.append(f"ForexFactory (forecast-date) score {sc} (high:{int(hi)}, mid:{int(mi)})")
            if int(hi) > 0:
                cues.append("High-impact economic events scheduled for forecast date — increase caution at open.")
                # bump volatility label if not already high
                if vol_label != "High":
                    vol_label = "High"
        except Exception:
            pass

    # Adjust bias phrasing for open: make it directional + speed
    if base_bias in ("Bullish", "Slow Bullish"):
        open_bias = "Bullish at Open" if base_bias == "Bullish" else "Mildly Bullish at Open"
    elif base_bias in ("Bearish", "Slow Bearish"):
        open_bias = "Bearish at Open" if base_bias == "Bearish" else "Mildly Bearish at Open"
    else:
        open_bias = base_bias

    explanation = (
        f"Forecast for next-day open: {open_bias}. Volatility expected: {vol_label}. "
        + base_expl
        + " Notes: "
        + "; ".join(cues)
    )

    return open_bias, explanation, vol_label
 

def add_bias_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    biases = []
    explanations = []
    for _, row in df.iterrows():
        b, e = generate_bias_and_explanation(row)
        biases.append(b)
        explanations.append(e)
    df["Daily_Bias"] = biases
    df["Bias_Explanation"] = explanations
    return df


def get_overextension_status(row: pd.Series, threshold: float = 2.0) -> str:
    z = row.get("ZScore_Close", np.nan)
    if np.isnan(z):
        return "Unknown"
    if z >= threshold:
        return f"Overextended to the upside (Z-score {z:.2f} ≥ {threshold})."
    elif z <= -threshold:
        return f"Overextended to the downside (Z-score {z:.2f} ≤ -{threshold})."
    else:
        return f"Within normal range (Z-score {z:.2f})."