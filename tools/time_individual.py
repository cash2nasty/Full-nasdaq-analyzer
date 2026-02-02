import time
import sys, traceback, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from data_fetch import get_intraday_data, get_daily_data, build_base_dataframe

def run_and_time(func, *args, **kwargs):
    name = func.__name__
    print(f"\nRunning {name}...")
    start = time.time()
    try:
        res = func(*args, **kwargs)
        took = time.time() - start
        print(f"{name} completed in {took:.2f}s; result type: {type(res)}, length: {len(res) if hasattr(res, '__len__') else 'n/a'}")
    except Exception as e:
        took = time.time() - start
        print(f"{name} raised after {took:.2f}s")
        traceback.print_exc()

if __name__ == '__main__':
    # intraday: small window
    run_and_time(get_intraday_data, '2d', '5m')
    # daily: shorter period to be faster
    run_and_time(get_daily_data, '3mo')
    # base dataframe: full build
    run_and_time(build_base_dataframe)
