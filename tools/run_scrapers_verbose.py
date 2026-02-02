import sys
import pathlib
import traceback

# Ensure project root is on sys.path so imports from the repo work when this
# script is executed from the tools/ folder.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from data_fetch import fetch_seekingalpha_news, _fetch_forexfactory_for_date, POLY_TICKER, _fetch_nasdaq_com_tick, _load_external_ticks
import pandas as pd


def try_sa():
    try:
        sa = fetch_seekingalpha_news(POLY_TICKER, max_items=10)
        print('SA count', len(sa))
        for a in sa:
            print('-', (a.get('title') or '')[:120], a.get('url'))
    except Exception:
        print('SA exception:')
        traceback.print_exc()


def try_ff():
    try:
        ff = _fetch_forexfactory_for_date(pd.Timestamp('2026-02-02').date())
        print('FF count', len(ff))
        for e in ff[:10]:
            print('-', e.get('time'), e.get('currency'), e.get('impact'), e.get('event'))
    except Exception:
        print('FF exception:')
        traceback.print_exc()


def try_nq():
    try:
        nq = _fetch_nasdaq_com_tick(POLY_TICKER)
        print('Nasdaq tick:', None if nq is None else nq.to_dict(orient='records'))
    except Exception:
        print('Nasdaq tick exception:')
        traceback.print_exc()


def try_load_ext():
    try:
        ext = _load_external_ticks()
        print('External ticks loaded:', None if ext is None else len(ext))
    except Exception:
        print('Load external ticks exception:')
        traceback.print_exc()


if __name__ == '__main__':
    print('Running verbose scraper test...')
    try_sa()
    try_ff()
    try_nq()
    try_load_ext()
