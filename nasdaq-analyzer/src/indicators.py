# This file contains functions for calculating various financial indicators used in the analysis.

import pandas as pd

def moving_average(data: pd.Series, window: int) -> pd.Series:
    """Calculate the moving average of a given data series."""
    return data.rolling(window=window).mean()

def exponential_moving_average(data: pd.Series, span: int) -> pd.Series:
    """Calculate the exponential moving average of a given data series."""
    return data.ewm(span=span, adjust=False).mean()

def relative_strength_index(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index (RSI) for a given data series."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def bollinger_bands(data: pd.Series, window: int = 20, num_std_dev: int = 2) -> pd.DataFrame:
    """Calculate Bollinger Bands for a given data series."""
    rolling_mean = moving_average(data, window)
    rolling_std = data.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return pd.DataFrame({'upper_band': upper_band, 'lower_band': lower_band})

def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate the Average True Range (ATR) for a given set of price data."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()