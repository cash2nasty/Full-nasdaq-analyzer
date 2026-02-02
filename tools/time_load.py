import time
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from data_fetch import get_intraday_data, get_daily_data, build_base_dataframe

print('Timing get_intraday_data (2d,5m)...')
start=time.time()
idf=get_intraday_data(period='2d', interval='5m')
print('intraday rows:', len(idf))
print('took %.2fs' % (time.time()-start))

print('\nTiming get_daily_data (6mo)...')
start=time.time()
df=get_daily_data(period='6mo')
print('daily rows:', len(df))
print('took %.2fs' % (time.time()-start))

print('\nTiming build_base_dataframe()...')
start=time.time()
bb=build_base_dataframe()
print('base df rows:', len(bb))
print('took %.2fs' % (time.time()-start))
