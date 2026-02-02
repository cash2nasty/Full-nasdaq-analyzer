import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from data_fetch import get_intraday_data, _load_external_ticks
import pandas as pd

print('Checking external ticks file...')
ext = _load_external_ticks()
print('external ticks:', 'None' if ext is None else f'{len(ext)} rows')
if ext is not None:
    print(ext.tail(5))

print('\nLoading intraday data via get_intraday_data()...')
idf = get_intraday_data(period='2d', interval='5m')
print('intraday rows:', len(idf))
if not idf.empty:
    print('index tz-aware:', hasattr(idf.index, 'tz') and idf.index.tz is not None)
    print('last index:', idf.index.max())
    print(idf.tail(5))
else:
    print('intraday is empty')
