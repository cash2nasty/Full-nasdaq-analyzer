from data_fetch import get_intraday_data, _fetch_seekingalpha_tick, _load_seekingalpha_ticks
import pandas as pd

intra = get_intraday_data()
print("Intraday empty:", intra.empty)
if not intra.empty:
    print("Last 6 intraday rows:")
    try:
        print(intra.tail(6).to_string())
    except Exception:
        print(intra.tail(6))

print("\nSeeking Alpha tick:")
try:
    print(_fetch_seekingalpha_tick("NQ=F"))
except Exception as e:
    print("Error fetching SA tick:", e)

print("\nPersisted Seeking Alpha ticks (if any):")
try:
    p = _load_seekingalpha_ticks()
    if p is None:
        print("None")
    else:
        try:
            print(p.tail(6).to_string())
        except Exception:
            print(p.tail(6))
except Exception as e:
    print("Error loading persisted ticks:", e)
