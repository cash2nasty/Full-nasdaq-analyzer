#!/usr/bin/env python3
import pandas as pd
from data_fetch import build_base_dataframe
from indicators import build_indicators
from bias_engine import add_bias_columns

# Build dataframe and compute bias
print('Building base dataframe (may take a moment)...')
df = build_base_dataframe()
df = build_indicators(df)
df = add_bias_columns(df)

# normalize index
try:
    df.index = pd.to_datetime(df.index)
except Exception:
    pass

q = pd.to_datetime('2026-02-02')
if q in df.index:
    row = df.loc[q]
    print('DAILY_BIAS:', row.get('Daily_Bias'))
else:
    print('DATE_NOT_IN_DF')
    prior_idx = df.index[df.index < q]
    if len(prior_idx) > 0:
        prior = prior_idx[-1]
        print('PRIOR_DATE:', prior)
        print('PRIOR_DAILY_BIAS:', df.loc[prior].get('Daily_Bias'))
    else:
        print('NO_PRIOR_ROW')
