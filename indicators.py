import pandas as pd
import numpy as np


def add_moving_averages(df: pd.DataFrame, windows=(10, 20, 50, 100, 200)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"MA_{w}"] = df["Close"].rolling(window=w).mean()
    return df


def add_returns_and_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    df["Ret"] = df["Close"].pct_change()
    df["Volatility"] = df["Ret"].rolling(window=window).std()
    df["StdDev_Close"] = df["Close"].rolling(window=window).std()
    return df


def add_z_scores(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    mean = df["Close"].rolling(window=window).mean()
    std = df["Close"].rolling(window=window).std()
    df["ZScore_Close"] = (df["Close"] - mean) / std
    return df


def add_rvol(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    avg_vol = df["Volume"].rolling(window=window).mean()
    df["RVOL"] = df["Volume"] / avg_vol
    return df


def add_roc(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    df = df.copy()
    df[f"ROC_{period}"] = df["Close"].pct_change(periods=period)
    return df


def add_momentum(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    df = df.copy()
    df[f"Momentum_{period}"] = df["Close"] - df["Close"].shift(period)
    return df


def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = add_moving_averages(df)
    df = add_returns_and_volatility(df)
    df = add_z_scores(df)
    df = add_rvol(df)
    df = add_roc(df, period=10)
    df = add_momentum(df, period=10)
    return df


def summarize_indicators(row: pd.Series) -> dict:
    """Return short, human-readable summaries for common indicators for a given row."""
    out = {}
    z = row.get("ZScore_Close", np.nan)
    vol = row.get("Volatility", np.nan)
    roc = row.get("ROC_10", np.nan)
    mom = row.get("Momentum_10", np.nan)
    rvol = row.get("RVOL", np.nan)

    # Z-score
    if not np.isnan(z):
        if z > 1.5:
            out["ZScore_Close"] = "Extended above mean — mean-reversion risk; price may pull back."
        elif z < -1.5:
            out["ZScore_Close"] = "Extended below mean — mean-reversion upside potential."
        else:
            out["ZScore_Close"] = "Near mean — no large statistical extension." 
    else:
        out["ZScore_Close"] = "Unknown"

    # Volatility
    if not np.isnan(vol):
        if vol > 0.02:
            out["Volatility"] = "Elevated volatility — expect larger intraday swings."
        elif vol < 0.01:
            out["Volatility"] = "Low volatility — likely range-bound or quiet action."
        else:
            out["Volatility"] = "Normal volatility — typical intraday movement."
    else:
        out["Volatility"] = "Unknown"

    # ROC
    if not np.isnan(roc):
        out["ROC_10"] = "Positive ROC indicates recent price appreciation; negative indicates recent weakness." if roc >= 0 else "Recent downward price change; short-term weakness."
    else:
        out["ROC_10"] = "Unknown"

    # Momentum
    if not np.isnan(mom):
        out["Momentum_10"] = "Positive momentum supports continuation; negative supports further weakness." if mom >= 0 else "Negative momentum — sellers have short-term control."
    else:
        out["Momentum_10"] = "Unknown"

    # RVOL
    if not np.isnan(rvol):
        if rvol > 1.3:
            out["RVOL"] = "High relative volume — moves have participation and conviction."
        elif rvol < 0.7:
            out["RVOL"] = "Low relative volume — moves may lack conviction."
        else:
            out["RVOL"] = "Typical relative volume."
    else:
        out["RVOL"] = "Unknown"

    return out