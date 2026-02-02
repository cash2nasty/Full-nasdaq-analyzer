import pytest
from src.nasdaq_analyzer3 import load_full_dataframe, analyze_historical_patterns

def test_load_full_dataframe():
    df = load_full_dataframe()
    assert df is not None
    assert not df.empty
    assert 'Close' in df.columns
    assert 'Open' in df.columns
    assert 'High' in df.columns
    assert 'Low' in df.columns

def test_analyze_historical_patterns():
    df = load_full_dataframe()
    result = analyze_historical_patterns(df, lookback=60)
    assert isinstance(result, dict)
    assert 'summary' in result
    assert 'recommended_bias' in result
    assert 'explanation' in result
    assert 'signals' in result
    assert 'bias_score' in result
    assert 'stochastic' in result
    assert 'momentum' in result
    assert 'window_start' in result
    assert 'window_end' in result
    assert 'most_impact_date' in result
    assert 'most_impact_value' in result
    assert 'most_impact_contributors' in result

def test_analyze_historical_patterns_empty_dataframe():
    empty_df = pd.DataFrame()
    result = analyze_historical_patterns(empty_df)
    assert result == {}