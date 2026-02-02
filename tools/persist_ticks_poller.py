import sys
import pathlib
import time
import argparse

# ensure project root on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from data_fetch import _fetch_seekingalpha_tick, _fetch_nasdaq_com_tick, _persist_external_tick


def poll_once():
    ticks = []
    try:
        sa = _fetch_seekingalpha_tick('NQ=F')
        if sa is not None:
            ticks.append(('seekingalpha', sa))
    except Exception:
        pass
    try:
        nq = _fetch_nasdaq_com_tick('NQ')
        if nq is not None:
            ticks.append(('nasdaq.com', nq))
    except Exception:
        pass

    for src, df in ticks:
        try:
            _persist_external_tick(df, source=src)
            print('Persisted tick from', src)
        except Exception as e:
            print('Failed to persist', src, e)

    if not ticks:
        print('No ticks fetched')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=int, default=60, help='Poll interval seconds')
    parser.add_argument('--once', action='store_true', help='Run a single poll and exit')
    args = parser.parse_args()

    if args.once:
        poll_once()
    else:
        try:
            while True:
                poll_once()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print('Stopping poller')
