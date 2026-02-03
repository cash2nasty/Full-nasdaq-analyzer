import streamlit as st
import pandas as pd
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo
import os
import requests

from data_fetch import build_base_dataframe, get_intraday_data, compute_forexfactory_features, POLY_TICKER, _try_polygon_daily
from data_fetch import fetch_seekingalpha_news, _fetch_forexfactory_for_date
from indicators import build_indicators, summarize_indicators
from bias_engine import add_bias_columns, get_overextension_status, predict_next_day_bias
from notes import add_note_for_date, get_notes_for_date, update_close_history
from utils import style_stats_table
import json
from pathlib import Path


st.set_page_config(page_title="Nasdaq Analyzer 3", layout="wide")


# Session stats persistence helpers
def _persist_session_stats(session_date: date, session_type: str, stats: dict, data_dir: str = "data") -> None:
    """Persist session stats (Asia/London) to a JSON file after they complete.
    
    Args:
        session_date: The calendar date of the trading day
        session_type: "asia" or "london"
        stats: Dictionary with session stats (open, close, high, low, volume, return, etc.)
        data_dir: Directory to store the session stats
    """
    try:
        p = Path(data_dir)
        p.mkdir(parents=True, exist_ok=True)
        
        # Store stats in a consolidated sessions file
        out = p / "session_stats.json"
        
        # Load existing stats if present
        existing = {}
        if out.exists():
            try:
                with open(out, 'r') as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        
        # Initialize date entry if not present
        date_str = session_date.isoformat()
        if date_str not in existing:
            existing[date_str] = {}
        
        # Store the session stats with completion timestamp
        stats_copy = stats.copy()
        stats_copy["persisted_at"] = datetime.utcnow().isoformat()
        existing[date_str][session_type] = stats_copy
        
        # Write back
        with open(out, 'w') as f:
            json.dump(existing, f, indent=2, default=str)
    except Exception as e:
        # Silently fail if persistence doesn't work
        pass


def _load_session_stats(session_date: date, session_type: str, data_dir: str = "data") -> dict | None:
    """Load previously persisted session stats for a given date and type.
    
    Args:
        session_date: The calendar date of the trading day
        session_type: "asia" or "london"
        data_dir: Directory where session stats are stored
    
    Returns:
        Dictionary with session stats if found, None otherwise
    """
    try:
        p = Path(data_dir) / "session_stats.json"
        if not p.exists():
            return None
        
        with open(p, 'r') as f:
            all_stats = json.load(f)
        
        date_str = session_date.isoformat()
        if date_str in all_stats and session_type in all_stats[date_str]:
            return all_stats[date_str][session_type]
        
        return None
    except Exception:
        return None


def _get_last_daily_update(data_dir: str = "data") -> date | None:
    """Return the date of the last automatic daily update, if available."""
    try:
        p = Path(data_dir) / "last_daily_update.json"
        if not p.exists():
            return None
        with open(p, 'r') as f:
            j = json.load(f)
        if not j or 'last_update' not in j:
            return None
        return pd.to_datetime(j['last_update']).date()
    except Exception:
        return None


def _set_last_daily_update(d: date, data_dir: str = "data") -> None:
    try:
        p = Path(data_dir)
        p.mkdir(parents=True, exist_ok=True)
        out = p / "last_daily_update.json"
        with open(out, 'w') as f:
            json.dump({"last_update": d.isoformat()}, f)
    except Exception:
        pass


def perform_daily_update(force: bool = False) -> bool:
    """Perform the daily update: refresh daily dataframe, intraday, persist session stats, and update close history.

    This is safe to call repeatedly; it will skip work if already run for today unless `force` is True.
    """
    try:
        now_est = pd.Timestamp.now(tz=ZoneInfo("America/New_York"))
        today_est = now_est.date()
        last = _get_last_daily_update()
        if last == today_est and not force:
            return False

        # Rebuild daily dataframe (do not rely on cached `load_full_dataframe` here)
        try:
            new_df = build_base_dataframe()
            new_df = build_indicators(new_df)
            new_df = add_bias_columns(new_df)
            globals()['df'] = new_df
        except Exception:
            pass

        # Update close history CSV
        try:
            if 'df' in globals() and globals().get('df') is not None:
                update_close_history(globals()['df'])
        except Exception:
            pass

        # Refresh intraday data (fetch last few days to ensure session windows are covered)
        try:
            new_intra = get_intraday_data(period="3d")
            if new_intra is not None and not new_intra.empty:
                globals()['intra'] = new_intra
        except Exception:
            pass

        # Persist session stats for today (sessions should be complete by 17:00 EST)
        try:
            # target trading date is today_est
            td = today_est
            # Asia session
            try:
                a_start, a_end = asia_window_for_date(td)
                if 'intra' in globals() and not globals().get('intra').empty:
                    a_slice = globals().get('intra').loc[(globals().get('intra').index >= a_start) & (globals().get('intra').index < a_end)]
                else:
                    a_slice = pd.DataFrame()
                a_stats = session_stats_from_slice(a_slice)
                if a_stats:
                    a_stats['complete'] = True
                    _persist_session_stats(td, 'asia', a_stats)
            except Exception:
                pass

            # London session
            try:
                l_start, l_end = london_window_for_date(td)
                if 'intra' in globals() and not globals().get('intra').empty:
                    l_slice = globals().get('intra').loc[(globals().get('intra').index >= l_start) & (globals().get('intra').index < l_end)]
                else:
                    l_slice = pd.DataFrame()
                l_stats = session_stats_from_slice(l_slice)
                if l_stats:
                    l_stats['complete'] = True
                    _persist_session_stats(td, 'london', l_stats)
            except Exception:
                pass
        except Exception:
            pass

        # Record last update
        try:
            _set_last_daily_update(today_est)
        except Exception:
            pass
        return True
    except Exception:
        return False


def daily_update_if_due() -> None:
    """Check wall-clock and run daily update once per day after 17:00 EST."""
    try:
        now_est = pd.Timestamp.now(tz=ZoneInfo("America/New_York"))
        today_est = now_est.date()
        last = _get_last_daily_update()
        # run after 17:00 EST
        if now_est.time() >= time(17, 0) and (last is None or last < today_est):
            updated = perform_daily_update()
            # If we performed an update, rerun the app so UI (sidebar bounds) refreshes
            try:
                if updated:
                    st.rerun()
            except Exception:
                pass
    except Exception:
        pass


def _load_all_session_stats(data_dir: str = "data") -> dict | None:
    """Load all session stats from the persistent file.
    
    Args:
        data_dir: Directory where session stats are stored
    
    Returns:
        Dictionary with all session stats (by date and session type), or None if file doesn't exist
    """
    try:
        p = Path(data_dir) / "session_stats.json"
        if not p.exists():
            return None
        
        with open(p, 'r') as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_full_dataframe() -> pd.DataFrame:
    df = build_base_dataframe()
    df = build_indicators(df)
    df = add_bias_columns(df)
    return df


# Cached small CSV loaders to speed app startup (non-blocking, cached by Streamlit)
@st.cache_data(show_spinner=False)
def load_external_ticks() -> pd.DataFrame:
    # try common locations for the persisted external ticks file
    candidates = ("data/external_ticks.csv", "external_ticks.csv")
    for p in candidates:
        try:
            if os.path.exists(p):
                return pd.read_csv(p, parse_dates=True)
        except Exception:
            try:
                return pd.read_csv(p)
            except Exception:
                continue
    # return empty DataFrame if not found/readable
    return pd.DataFrame()


def analyze_historical_patterns(df: pd.DataFrame, target_date: date | None = None, lookback: int = 60, feature_cols: list | None = None) -> dict:
    """Analyze historical daily data to detect patterns and produce signal metrics and
    a recommended bias.

    Returns a dict with summary stats, stochastic values, tested signal metrics,
    and a `recommended_bias` string plus a short `explanation`.
    """
    import numpy as _np

    if df is None or df.empty:
        return {}

    # normalize index and choose window
    dindex = pd.to_datetime(df.index)
    df_local = df.copy()
    df_local.index = dindex

    if target_date is None:
        end_date = pd.to_datetime(df_local.index.max()).date()
    else:
        end_date = pd.to_datetime(target_date).date()

    start_date = end_date - timedelta(days=lookback)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    win = df_local.loc[(df_local.index >= start_ts) & (df_local.index < end_ts)]
    if win.empty:
        return {}

    # basic stats
    win = win.copy()
    win['Ret'] = win['Close'].pct_change()
    stats = {
        'n_days': int(len(win)),
        'mean_return': float(win['Ret'].mean()) if 'Ret' in win else None,
        'std_return': float(win['Ret'].std()) if 'Ret' in win else None,
        'avg_range_pct': float(((win['High'] - win['Low']) / win['Close']).mean()) if {'High','Low','Close'}.issubset(win.columns) else None,
        'avg_volume': float(win['Volume'].mean()) if 'Volume' in win.columns else None,
        'zscore_mean': float(win['ZScore_Close'].mean()) if 'ZScore_Close' in win.columns else None,
    }

    # stochastic oscillator (K,D) using 14-day default
    k_period = 14
    stoch_k = None
    stoch_d = None
    try:
        low_k = win['Low'].rolling(k_period).min()
        high_k = win['High'].rolling(k_period).max()
        if not pd.isna(low_k.iloc[-1]) and not pd.isna(high_k.iloc[-1]) and (high_k.iloc[-1] - low_k.iloc[-1]) != 0:
            stoch_k = float((win['Close'].iloc[-1] - low_k.iloc[-1]) / (high_k.iloc[-1] - low_k.iloc[-1]) * 100)
            stoch_d = float(win['Close'].pct_change().rolling(3).mean().iloc[-1]) if 'Close' in win else None
    except Exception:
        stoch_k = None
        stoch_d = None

    # short-term momentum (1/5/10 days)
    mom = {}
    try:
        mom['r1'] = float(win['Close'].pct_change(1).iloc[-1])
    except Exception:
        mom['r1'] = None
    for p in (5, 10):
        try:
            mom[f'r{p}'] = float(win['Close'].pct_change(p).iloc[-1])
        except Exception:
            mom[f'r{p}'] = None

    # Candidate signals to test historically (simple heuristics)
    signals = {}
    # ensure MA_20 present or compute
    if 'MA_20' not in win.columns and 'Close' in win.columns:
        win['MA_20'] = win['Close'].rolling(20).mean()

    # define signal lambdas
    cand = {
        'close_gt_MA20': (lambda r: r['Close'] > r.get('MA_20', _np.nan)),
        'z_gt_1': (lambda r: (r.get('ZScore_Close') is not None) and (r.get('ZScore_Close') > 1.0)),
        'rv_gt_1.2': (lambda r: (r.get('RVOL') is not None) and (r.get('RVOL') > 1.2)),
        'prev_up_day': (lambda r: (r.get('Ret') is not None) and (r.get('Ret') > 0)),
        'bullish_day': (lambda r: (r.get('Close') is not None and r.get('Open') is not None) and (r.get('Close') > r.get('Open'))),
    }

    # compute next-day open-close move as target (use next day's close vs open)
    nd_open = win['Open'].shift(-1)
    nd_close = win['Close'].shift(-1)
    nd_move = (nd_close - nd_open) / nd_open

    sig_map = {}
    for name, fn in cand.items():
        sig_series = []
        for i, row in win.iterrows():
            try:
                sig = bool(fn(row))
            except Exception:
                sig = False
            sig_series.append(sig)
        sig_s = pd.Series(sig_series, index=win.index)
        sig_map[name] = sig_s
        # compute accuracy where next-day move exists
        mask = ~nd_move.isna()
        if mask.sum() == 0:
            signals[name] = {'accuracy': None, 'avg_next_return': None, 'count': int(sig_s.sum()), 'current_signal': bool(sig_s.iloc[-1])}
            continue
        relevant = sig_s & mask
        if relevant.sum() == 0:
            acc = None
            meanr = None
        else:
            # treat positive next-day move as up
            nd_dir = nd_move[relevant] > 0
            acc = float((nd_dir).sum() / len(nd_dir))
            meanr = float(nd_move[relevant].mean())
        signals[name] = {'accuracy': acc, 'avg_next_return': meanr, 'count': int(sig_s.sum()), 'current_signal': bool(sig_s.iloc[-1])}

    # aggregate a simple score from candidate signals (accuracy-weighted)
    score = 0.0
    weight_sum = 0.0
    for n, m in signals.items():
        if m.get('accuracy') is None:
            continue
        w = (m['accuracy'] - 0.5)  # accuracy above random
        sign = 1 if m.get('current_signal') else 0
        score += w * sign
        weight_sum += abs(w)

    # normalize
    bias_score = (score / weight_sum) if weight_sum != 0 else 0.0
    if bias_score > 0.15:
        recommended = 'Bullish'
    elif bias_score < -0.15:
        recommended = 'Bearish'
    else:
        recommended = 'Neutral'

    explanation = f"Pattern analysis over last {len(win)} days suggests {recommended} (score {bias_score:.3f})."

    # compute per-date impact: sum of avg_next_return of signals active on that date
    try:
        impact = pd.Series(0.0, index=win.index)
        for name, sig_s in sig_map.items():
            avgr = signals.get(name, {}).get('avg_next_return')
            if avgr is None:
                continue
            impact = impact.add(sig_s.astype(float) * float(avgr), fill_value=0.0)
        if not impact.empty:
            impact_abs = impact.abs()
            most_idx = impact_abs.idxmax()
            most_val = float(impact.loc[most_idx])
            # list contributing signals on that date
            contrib = [n for n, s in sig_map.items() if bool(s.loc[most_idx])]
            most_impact_date = str(pd.to_datetime(most_idx).date())
        else:
            most_impact_date = None
            most_val = None
            contrib = []
    except Exception:
        most_impact_date = None
        most_val = None
        contrib = []
    return {
        'summary': stats,
        'stochastic': {'k': stoch_k, 'd': stoch_d},
        'momentum': mom,
        'signals': signals,
        'recommended_bias': recommended,
        'bias_score': float(bias_score),
        'explanation': explanation,
        'window_start': str(start_date),
        'window_end': str(end_date),
        'most_impact_date': most_impact_date,
        'most_impact_value': most_val,
        'most_impact_contributors': contrib,
    }

# present selectable date list as plain python dates for Streamlit widgets
# ensure the daily dataframe `df` is loaded and available
if 'df' not in globals() or globals().get('df') is None:
    df = load_full_dataframe()
else:
    df = globals().get('df')

# coerce index to datetimes and build date list safely
if df is None or df.empty:
    all_dates = []
else:
    all_dates = [d.date() for d in pd.to_datetime(df.index)]

# allow selecting up to today's date (EST) even if daily df hasn't been updated yet
today_est_sidebar = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).date()

# determine latest available date, fallback to today EST when no data present
latest_date = all_dates[-1] if len(all_dates) > 0 else today_est_sidebar

st.sidebar.title("Sidebar Notes & Controls")

# Auto-refresh mechanism: if current time is past a session end time and we haven't checked recently,
# automatically refresh intraday data to capture just-completed session stats
def check_and_refresh_for_completed_sessions():
    """Check if current time is past session ends and trigger a refresh if needed."""
    now_est = pd.Timestamp.now(tz=ZoneInfo("America/New_York"))
    today_est = now_est.date()
    
    # Check if we should refresh (simple heuristic: refresh if we're past session ends and haven't checked in ~5 min)
    asia_end_est = pd.Timestamp.combine(today_est, time(1, 0)).tz_localize(ZoneInfo("America/New_York"))
    london_end_est = pd.Timestamp.combine(today_est, time(8, 0)).tz_localize(ZoneInfo("America/New_York"))
    
    # If past either session end and session might have just finished, try to refresh
    should_refresh = False
    if now_est >= london_end_est and now_est < london_end_est + pd.Timedelta(minutes=15):
        should_refresh = True
    elif now_est >= asia_end_est and now_est < asia_end_est + pd.Timedelta(minutes=15) and now_est.hour < 4:
        should_refresh = True
    
    if should_refresh and 'intra' in globals():
        try:
            new_intra = get_intraday_data()
            if new_intra is not None and not new_intra.empty:
                # Check if we got new data since last load
                old_intra = globals().get('intra')
                if old_intra is None or old_intra.empty or new_intra.index.max() > old_intra.index.max():
                    globals()['intra'] = new_intra
                    # Rerun to pick up new session stats
                    st.rerun()
        except Exception:
            pass

# Trigger the auto-refresh check
try:
    check_and_refresh_for_completed_sessions()
except Exception:
    pass

# Run daily update if it's past 17:00 EST and not run yet today
try:
    daily_update_if_due()
except Exception:
    pass

# Refresh control: clear caches, reload data and recompute derived stats
if st.sidebar.button("Refresh data & recompute", key="refresh_data"):
    st.sidebar.info("Refreshing caches and reloading datasets...")
    # Attempt to clear Streamlit caches (best-effort across versions)
    try:
        st.cache_data.clear()
    except Exception:
        try:
            st.experimental_memo.clear()
        except Exception:
            pass

    missing = []
    # reload main dataframe (builds indicators and bias columns)
    try:
        new_df = load_full_dataframe()
        if new_df is None or new_df.empty:
            missing.append("daily dataframe")
        else:
            globals()['df'] = new_df
    except Exception as e:
        missing.append(f"daily dataframe ({str(e)})")

    # reload intraday
    try:
        new_intra = get_intraday_data()
        if new_intra is None or (isinstance(new_intra, pd.DataFrame) and new_intra.empty):
            missing.append("intraday data")
        else:
            globals()['intra'] = new_intra
    except Exception as e:
        missing.append(f"intraday data ({str(e)})")

    # reload external ticks (optional fallback)
    try:
        new_ticks = load_external_ticks()
        if new_ticks is None or (isinstance(new_ticks, pd.DataFrame) and new_ticks.empty):
            # not fatal but inform the user
            missing.append("external ticks (fallback)")
        else:
            st.session_state['external_ticks'] = new_ticks
    except Exception:
        missing.append("external ticks (fallback)")

    # Provide user feedback about any missing data
    if missing:
        st.sidebar.warning("The following data/stat(s) are unavailable: " + ", ".join(missing) + ". Check back later.")
    else:
        st.sidebar.success("Datasets refreshed successfully.")

    # Rerun to pick up reloaded globals and recompute UI.
    # Try the direct API first; if unavailable, fall back to changing query params
    # (which triggers a rerun) and finally a small JS reload as last resort.
    try:
        st.experimental_rerun()
    except Exception:
        try:
            # prefer changing query params to force Streamlit to rerun the script
            import time as _time
            try:
                # Use the modern query-params API: assign to st.query_params to trigger a rerun
                st.query_params = {"_refresh": str(int(_time.time()))}
                # stop the current run so Streamlit will rerun automatically
                st.stop()
            except Exception:
                pass

            # last-resort: inject a tiny JS snippet to reload the page (works in most browsers)
            try:
                from streamlit.components.v1 import html as _st_html
                _st_html("<script>window.location.reload();</script>", height=0)
                st.stop()
            except Exception:
                # If everything fails, inform the user and stop.
                st.sidebar.info("Refresh complete — please reload the app page to see updated data.")
                st.stop()
        except Exception:
            st.sidebar.info("Refresh complete — please reload the app page to see updated data.")
            st.stop()

# Polygon API key input: allow pasting/saving the key to a local .env and validating it
current_poly = os.getenv("POLYGON_API_KEY")
st.sidebar.subheader("Polygon API Key")
poly_input = st.sidebar.text_input("Paste Polygon API key (optional)", value=current_poly or "", type="password")
if st.sidebar.button("Save Polygon API key"):
    if poly_input.strip():
        try:
            env_path = ".env"
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                new_lines = []
                found = False
                for ln in lines:
                    if ln.startswith("POLYGON_API_KEY="):
                        new_lines.append(f"POLYGON_API_KEY={poly_input.strip()}\n")
                        found = True
                    else:
                        new_lines.append(ln)
                if not found:
                    new_lines.append(f"POLYGON_API_KEY={poly_input.strip()}\n")
                with open(env_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
            else:
                with open(env_path, "w", encoding="utf-8") as f:
                    f.write(f"POLYGON_API_KEY={poly_input.strip()}\n")
        except Exception:
            pass
        os.environ["POLYGON_API_KEY"] = poly_input.strip()
        st.sidebar.success("Polygon key saved to .env and applied to session. Restart the app to reload data.")
        st.sidebar.write("Or press 'Validate Polygon key' below to test connectivity (data reload may still require restart).")

if st.sidebar.button("Validate Polygon key"):
    key_to_test = poly_input.strip() or current_poly
    if not key_to_test:
        st.sidebar.error("No key provided.")
    else:
        try:
            # quick validation: attempt a small polygon daily query
            # try the configured POLY_TICKER first, then fall back to the yfinance futures symbol
            tried = []
            ok = False
            try:
                _try_polygon_daily(POLY_TICKER, period="1mo")
                ok = True
                tried.append(POLY_TICKER)
            except Exception as e:
                tried.append(f"{POLY_TICKER}: {str(e)}")
            from data_fetch import YF_TICKER
            if not ok:
                try:
                    _try_polygon_daily(YF_TICKER, period="1mo")
                    ok = True
                    tried.append(YF_TICKER)
                except Exception as e:
                    tried.append(f"{YF_TICKER}: {str(e)}")

            if ok:
                st.sidebar.success("Polygon key appears valid (daily query succeeded).")
            else:
                st.sidebar.error(
                    "Polygon validation failed: Polygon returned no usable historical results for the tickers tried. "
                    "Try verifying your API key and consider a different Polygon ticker symbol (e.g. 'NQ', 'NQ=F' or a CME-prefixed ticker)."
                )
                # show brief details of attempts
                for t in tried:
                    st.sidebar.write(f"- {t}")
        except Exception as e:
            st.sidebar.error(f"Polygon validation failed: {str(e)}")

# Diagnostic helper: attempt v3 reference search and v2 aggs for the candidate tickers and show JSON
if st.sidebar.button("Run Polygon diagnostic"):
    key_to_test = poly_input.strip() or current_poly
    if not key_to_test:
        st.sidebar.error("No key provided for diagnostic.")
    else:
        try:
            st.sidebar.write("Running diagnostic queries (showing trimmed JSON)...")
            headers = {"User-Agent": "Mozilla/5.0"}
            # v3 reference tickers search
            q = "NQ"
            url_ref = f"https://api.polygon.io/v3/reference/tickers?search={q}&active=true&limit=5&apiKey={key_to_test}"
            r_ref = requests.get(url_ref, headers=headers, timeout=10)
            r_ref.raise_for_status()
            jref = r_ref.json()
            st.sidebar.write("v3 reference tickers (search=NQ):")
            st.sidebar.write(str(jref)[:400])

            # try aggs for configured POLY_TICKER and fallback YF_TICKER (date window small)
            from data_fetch import POLY_TICKER, YF_TICKER
            start = (pd.Timestamp.now() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
            end = pd.Timestamp.now().strftime("%Y-%m-%d")
            for t in (POLY_TICKER, YF_TICKER):
                try:
                    pt = t.lstrip('^')
                    url_aggs = f"https://api.polygon.io/v2/aggs/ticker/{pt}/range/1/day/{start}/{end}?adjusted=true&sort=asc&limit=5&apiKey={key_to_test}"
                    r = requests.get(url_aggs, headers=headers, timeout=10)
                    r.raise_for_status()
                    ja = r.json()
                    st.sidebar.write(f"aggs v2 for {pt} -> status {r.status_code}")
                    st.sidebar.write(str(ja)[:400])
                except Exception as e:
                    st.sidebar.write(f"aggs v2 for {t} failed: {str(e)}")
        except Exception as e:
            st.sidebar.error(f"Diagnostic failed: {str(e)}")

# prevent weekend selection: keep re-prompting if weekend is selected
# compute max selectable date: allow next trading day after 17:00 EST
now_est_sidebar = pd.Timestamp.now(tz=ZoneInfo("America/New_York"))
max_candidate = today_est_sidebar
if now_est_sidebar.time() >= time(17, 0):
    # propose next calendar day
    cand = today_est_sidebar + timedelta(days=1)
    # if cand falls on weekend, advance to Monday
    if cand.weekday() >= 5:
        days_to_add = 7 - cand.weekday()
        cand = cand + timedelta(days=days_to_add)
    max_candidate = cand

while True:
    default_val = latest_date if latest_date <= max_candidate else max_candidate
    selected_date = st.sidebar.date_input(
        "Select date (weekdays only)",
        value=default_val,
        min_value=all_dates[0],
        max_value=max_candidate,
    )
    if isinstance(selected_date, datetime):
        selected_date = selected_date.date()
    
    # Check if weekend (Saturday=5, Sunday=6)
    if selected_date.weekday() >= 5:
        st.sidebar.error(f"⚠️ {selected_date.strftime('%A')} is not a trading day. Please select a weekday (Monday-Friday).")
        st.rerun()
    else:
        break

# quick sidebar toggle to render News & Events panel above tabs (helpful if tab is off-screen)
show_news_panel = st.sidebar.checkbox("Show News & Events panel", value=False)
if show_news_panel:
    st.markdown("---")
    st.header("News & Events (Sidebar Panel)")
    st.write(f"Selected date: {selected_date.isoformat()}")
    try:
        sa_news_today = fetch_seekingalpha_news(POLY_TICKER, max_items=5)
    except Exception:
        sa_news_today = []
    if sa_news_today:
        st.subheader("Seeking Alpha — Top Headlines")
        for a in sa_news_today:
            st.write(f"- {a.get('title')}")
    else:
        st.write("No Seeking Alpha headlines available.")
    try:
        ff_events = _fetch_forexfactory_for_date(pd.to_datetime(selected_date).date())
        if ff_events:
            hi = sum(1 for e in ff_events if e.get('impact','').lower()=='high')
            st.write(f"ForexFactory events — High:{hi}, total:{len(ff_events)}")
        else:
            st.write("No ForexFactory events for selected date.")
    except Exception:
        st.write("Failed to fetch ForexFactory events.")

# Sidebar toggle to run the new pattern analysis helper
show_pattern_panel = st.sidebar.checkbox("Show Pattern Analysis panel", value=False)
if show_pattern_panel:
    st.markdown("---")
    st.header("Pattern Analysis (Sidebar Panel)")
    st.write(f"Selected date: {selected_date.isoformat()}")
    try:
        pa = analyze_historical_patterns(df, target_date=selected_date, lookback=60)
    except Exception:
        pa = {}
    if pa:
        st.subheader("Recommended Bias")
        rec_bias = pa.get('recommended_bias')
        st.write(rec_bias)
        st.write(pa.get('explanation'))

        # Interpret the bias score: it's normalized to [-1, 1]
        try:
            score = float(pa.get('bias_score', 0.0))
        except Exception:
            score = 0.0
        pct = abs(score) * 100.0
        # strength categorization
        if abs(score) >= 0.5:
            strength = 'Strong'
        elif abs(score) >= 0.25:
            strength = 'Moderate'
        elif abs(score) >= 0.15:
            strength = 'Weak'
        else:
            strength = 'Very Weak / Neutral'

        st.subheader('Score Interpretation')
        st.write(f"Score: {score:.3f} (normalized scale -1 → 1, where +1 is maximally bullish and -1 is maximally bearish)")
        st.write(f"Confidence (absolute): {pct:.1f}% — Strength: {strength}")

        # short meaning for the selected trading date
        try:
            sd = selected_date.isoformat()
        except Exception:
            sd = str(pa.get('window_end'))
        if rec_bias and rec_bias.lower().startswith('bull') and score > 0:
            st.write(f"For {sd}: historical signals produced a net bullish score ({strength}). Expect higher probability of an upward open-to-close move based on past patterns, with {pct:.1f}% relative confidence.")
        elif rec_bias and rec_bias.lower().startswith('bear') and score < 0:
            st.write(f"For {sd}: historical signals produced a net bearish score ({strength}). Expect higher probability of a downward open-to-close move based on past patterns, with {pct:.1f}% relative confidence.")
        else:
            st.write(f"For {sd}: the combined historical signals do not give a strong directional preference (strength={strength}). Treat the bias as low-confidence guidance and watch session stats for confirmation.")

        # U.S. Market Open (09:30am EST) bias: combine pattern score with early session returns
        try:
            base = float(pa.get('bias_score', 0.0))
        except Exception:
            base = 0.0
        adj = 0.0
        asia_ret = None
        london_ret = None
        try:
            asiast = globals().get('asia_stats')
            if isinstance(asiast, dict):
                asia_ret = asiast.get('return')
                if asia_ret is not None:
                    adj += float(asia_ret) * 0.5
        except Exception:
            asia_ret = None
        try:
            londst = globals().get('london_stats')
            if isinstance(londst, dict):
                london_ret = londst.get('return')
                if london_ret is not None:
                    adj += float(london_ret) * 0.3
        except Exception:
            london_ret = None

        open_score = base + adj
        if open_score > 0.15:
            open_bias = 'Bullish Open'
        elif open_score < -0.15:
            open_bias = 'Bearish Open'
        else:
            open_bias = 'Neutral / Unclear Open'

        st.subheader('U.S. Market Open (09:30am EST) Bias')
        st.write(open_bias)
        try:
            expl_parts = [f"Base pattern score {base:.3f}"]
            if asia_ret is not None:
                expl_parts.append(f"Asia return {asia_ret:+.2%} (weight 0.5)")
            if london_ret is not None:
                expl_parts.append(f"London return {london_ret:+.2%} (weight 0.3)")
            expl_parts.append(f"Combined open score {open_score:.3f}")
            st.write(". ".join(expl_parts) + ".")
            if open_bias.startswith('Bull'):
                st.write("Interpretation: Historical patterns plus early-session direction favor an upward open; watch first 30–60 minutes for confirmation.")
            elif open_bias.startswith('Bear'):
                st.write("Interpretation: Historical patterns plus early-session direction favor a downward open; manage risk and watch for gap pressure.")
            else:
                st.write("Interpretation: No clear open-direction signal — treat as low-confidence and wait for session confirmation before trading directionally.")
        except Exception:
            pass

        # Session-adjusted daily bias: incorporate Asia/London session returns into the daily pattern score
        try:
            asia_ret = None
            london_ret = None
            # Safely fetch session stats from globals to avoid NameError if variables aren't yet defined
            asiast = globals().get('asia_stats')
            londst = globals().get('london_stats')
            if isinstance(asiast, dict):
                asia_ret = asiast.get('return')
            if isinstance(londst, dict):
                london_ret = londst.get('return')

            sess_adj = 0.0
            parts = []
            if asia_ret is not None:
                sess_adj += float(asia_ret) * 0.4
                parts.append(f"Asia {asia_ret:+.2%} *0.4")
            if london_ret is not None:
                sess_adj += float(london_ret) * 0.3
                parts.append(f"London {london_ret:+.2%} *0.3")

            base_score = float(pa.get('bias_score', 0.0))
            combined_score = base_score + sess_adj
            if combined_score > 0.15:
                combined_bias = 'Bullish (session-adjusted)'
            elif combined_score < -0.15:
                combined_bias = 'Bearish (session-adjusted)'
            else:
                combined_bias = 'Neutral (session-adjusted)'

            st.subheader('Session-Adjusted Daily Bias')
            st.write(combined_bias)
            expl = f"Base pattern score {base_score:.3f}. Session adj {sess_adj:.3f} ({'; '.join(parts)}) -> combined {combined_score:.3f}."
            st.write(expl)
        except Exception:
            pass
        st.subheader("Summary Stats")
        try:
            st.json(pa.get('summary'))
        except Exception:
            st.write(pa.get('summary'))
        st.subheader("Signals (sample)")
        sigs = pa.get('signals', {})
        # Short descriptions for each known signal to explain meaning for the selected date
        signal_desc = {
            'close_gt_MA20': 'Close > 20-day MA — price above recent trend (short-term bullish).',
            'z_gt_1': 'Z-Score > 1 — close unusually high vs recent distribution (strong momentum/overbought).',
            'rv_gt_1.2': 'RVOL > 1.2 — elevated volume vs typical (move has conviction).',
            'prev_up_day': 'Previous day was up — short-term momentum may continue.',
            'bullish_day': 'Close > Open — intra-day buyers controlled the session (bullish day).',
        }

        if sigs:
            for k, v in sigs.items():
                desc = signal_desc.get(k, '')
                active = v.get('current_signal')
                acc = v.get('accuracy')
                cnt = v.get('count')
                extra = f" (active today)" if active else ""
                acc_text = f"accuracy={acc:.2f}" if isinstance(acc, float) else f"accuracy={acc}"
                st.write(f"- {k}: count={cnt}, current={active}, {acc_text}{extra} — {desc}")
        else:
            st.write("No signal data available.")
        # Show most impactful historical date (if computed)
        mid = pa.get('most_impact_date')
        mval = pa.get('most_impact_value')
        mcont = pa.get('most_impact_contributors', [])
        if mid:
            st.subheader("Most Impactful Historical Date")
            try:
                st.write(f"Date: {mid}")
                if mval is not None:
                    st.write(f"Impact value (aggregated avg next-return): {mval:.4f}")
                if mcont:
                    st.write(f"Contributing signals: {', '.join(mcont)}")
                # attempt to show the raw stats from that date
                try:
                    row = df.loc[pd.to_datetime(mid)]
                    quick = {
                        'Open': row.get('Open'),
                        'High': row.get('High'),
                        'Low': row.get('Low'),
                        'Close': row.get('Close'),
                        'Volume': row.get('Volume'),
                        'ZScore_Close': row.get('ZScore_Close'),
                    }
                    st.json(quick)
                except Exception:
                    st.write("(No daily row available in dataset for that date)")

                # concise human-friendly explanation
                try:
                    rec = pa.get('recommended_bias') or 'Neutral'
                    expl = pa.get('explanation') or ''
                    st.subheader('Why this recommendation')
                    st.write(f"Recommendation: {rec}. {expl}")
                    if mcont:
                        st.write(f"The most impactful date {mid} influenced this recommendation because the signals {', '.join(mcont)} on that date historically produced the aggregated effect shown above.")
                except Exception:
                    pass
            except Exception:
                pass
    else:
        st.write("No pattern analysis available for the selected date/window.")

existing_notes = get_notes_for_date(selected_date)
st.sidebar.subheader(f"Notes for {selected_date.isoformat()}")
if existing_notes:
    for i, n in enumerate(existing_notes, 1):
        st.sidebar.write(f"{i}. {n}")
else:
    st.sidebar.write("(No notes yet.)")

new_note = st.sidebar.text_area("Add a new note", "")
if st.sidebar.button("Save note"):
    if new_note.strip():
        add_note_for_date(selected_date, new_note.strip())
        st.sidebar.success("Note saved. Reload to see it listed.")

# Quick access to Summary Search in case tabs overflow
st.sidebar.markdown("---")

tab_options = [
    "Summary",
    "Day Summary",
    "Price & MAs",
    "VWAP",
    "Indicators",
    "Next-Day Forecast",
    "Asia Session",
    "London Session",
    "Previous Days Stats",
    "Compare Days",
    "Summary Search",
    "News & Events",
]

# Single dropdown navigation replacing multiple tab buttons
selected_tab = st.selectbox("Select view", tab_options, index=0)
# Provide a tabs-like sequence so legacy "with tabs[i]:" blocks don't raise NameError;
# use Streamlit containers so the UI remains consistent with the selectbox navigation.
tabs = [st.container() for _ in tab_options]

# make top tab row horizontally scrollable and prevent stacking
st.markdown(
    '<style>'
    'div[role="tablist"]{display:flex; flex-wrap:nowrap; gap:8px; overflow-x:auto; white-space:nowrap; -webkit-overflow-scrolling:touch; padding-bottom:6px;}'
    'div[role="tablist"] > div {display:inline-flex; align-items:center;}'
    'div[role="tablist"] button[role="tab"]{flex:0 0 auto; display:inline-flex; align-items:center; margin-right:8px; padding:6px 10px; font-size:12px; min-width:90px; border-radius:6px;}'
    'div[role="tablist"] button[role="tab"]:not(.stButton){white-space:nowrap;}'
    '</style>',
    unsafe_allow_html=True,
)

# additional visual horizontal scrollbar synced with the Streamlit tablist
st.markdown(
        '''
        <div id="tab-scrollbar" style="overflow-x:auto; overflow-y:hidden; height:18px; margin-top:6px;">
            <div style="width:3000px; height:1px;"></div>
        </div>
        <script>
        (function(){
            function trySync(){
                var tl = document.querySelector('[role="tablist"]');
                var sb = document.getElementById('tab-scrollbar');
                if(!tl || !sb) return false;
                sb.onscroll = function(){ tl.scrollLeft = sb.scrollLeft; };
                tl.onscroll = function(){ sb.scrollLeft = tl.scrollLeft; };
                return true;
            }
            var tries = 0;
            var t = setInterval(function(){ if(trySync() || ++tries>50) clearInterval(t); }, 100);
        })();
        </script>
        ''',
        unsafe_allow_html=True,
)

# Quick jump link to make News & Events more accessible (anchors to the section below)
st.markdown('<div style="margin:8px 0 12px 0;"><a href="#news-section">Jump to News & Events</a></div>', unsafe_allow_html=True)

# Quick News & Events expander (small summary duplicated above tabs for easier access)
with st.expander("News & Events (Quick)", expanded=False):
    st.write(f"Selected date: {selected_date.isoformat()}")
    try:
        quick_sa = fetch_seekingalpha_news(POLY_TICKER, max_items=3)
    except Exception:
        quick_sa = []
    if quick_sa:
        st.subheader("Seeking Alpha — Top (Quick)")
        for a in quick_sa:
            st.write(f"- {a.get('title')}")
    else:
        st.write("No Seeking Alpha headlines available (quick view).")
    try:
        ff_q = _fetch_forexfactory_for_date(pd.to_datetime(selected_date).date())
    except Exception:
        ff_q = []
    if ff_q:
        hi = sum(1 for e in ff_q if e.get('impact','').lower()=='high')
        st.write(f"ForexFactory events (Quick) — High:{hi}, total:{len(ff_q)}")
    else:
        st.write("No ForexFactory events (quick view).")

try:
    selected_row = df.loc[pd.to_datetime(selected_date)]
    selected_in_df = True
except Exception:
    # selected date not present in daily dataframe — fall back to the last prior trading row
    prior_idx = df.index[pd.to_datetime(df.index).date < selected_date]
    if len(prior_idx) > 0:
        selected_row = df.loc[prior_idx[-1]]
    else:
        selected_row = df.iloc[-1]
    selected_in_df = False

# prepare intraday data and session slices in US/Eastern so multiple tabs can use them
intra = get_intraday_data()
try:
    intra.index = pd.to_datetime(intra.index).tz_convert(ZoneInfo("America/New_York"))
except Exception:
    intra.index = pd.to_datetime(intra.index).tz_localize("UTC").tz_convert(ZoneInfo("America/New_York"))

sel_date = pd.to_datetime(selected_date).date()

# If daily row is missing or contains NaNs (e.g., download failed), try to populate
# basic display fields from the latest intraday tick(s) so the UI can show data.
try:
    needs_fill = False
    for k in ("Close", "Open", "High", "Low", "Volume"):
        if (k not in selected_row) or pd.isna(selected_row.get(k)):
            needs_fill = True
            break
    if needs_fill and not intra.empty:
        # use most recent intraday row to fill display metrics
        last_ts = intra.index.max()
        last_row = intra.loc[last_ts]
        # if last_row is a DataFrame slice, take the last row
        if isinstance(last_row, pd.DataFrame):
            last_row = last_row.iloc[-1]
        for k in ("Close", "Open", "High", "Low", "Volume"):
            try:
                if k in last_row and not pd.isna(last_row.get(k)):
                    selected_row[k] = last_row.get(k)
            except Exception:
                continue
except Exception:
    pass

# ensure previous-trading-day variables exist early so UI render can't raise NameError
prev_trading_date = None
prev_asia_stats = {"complete": False}
prev_london_stats = {"complete": False}
def asia_window_for_date(d: date):
    """Return (start, end) timestamps for the Asia session for calendar date `d`.

    Asia session runs 19:00 previous calendar day -> 01:00 on `d` (America/New_York).
    Example: Monday's Asia session -> starts Sunday 19:00 and ends Monday 01:00.
    """
    start = pd.Timestamp.combine(d - timedelta(days=1), time(19, 0)).tz_localize(ZoneInfo("America/New_York"))
    end = pd.Timestamp.combine(d, time(1, 0)).tz_localize(ZoneInfo("America/New_York"))
    return start, end


def london_window_for_date(d: date):
    """Return (start, end) timestamps for the London session for calendar date `d`.

    London session runs 03:00 -> 08:00 on `d` (America/New_York).
    Example: Monday's London session -> starts Monday 03:00 and ends Monday 08:00.
    """
    start = pd.Timestamp.combine(d, time(3, 0)).tz_localize(ZoneInfo("America/New_York"))
    end = pd.Timestamp.combine(d, time(8, 0)).tz_localize(ZoneInfo("America/New_York"))
    return start, end


# Asia session for selected date
asia_start, asia_end = asia_window_for_date(sel_date)
asia_slice = intra.loc[(intra.index >= asia_start) & (intra.index < asia_end)]

# London session for selected date
london_start, london_end = london_window_for_date(sel_date)
london_slice = intra.loc[(intra.index >= london_start) & (intra.index < london_end)]

# determine current time in EST to decide whether sessions are complete
now_est = pd.Timestamp.now(tz=ZoneInfo("America/New_York"))

# determine if the selected trading day is already over (so current-day stats should be treated as unavailable)
today_est = now_est.date()
trading_day_over = sel_date < today_est

# if the trading day is over, hide current-day session slices (they become previous-day)
if trading_day_over:
    asia_slice_current = pd.DataFrame()
    london_slice_current = pd.DataFrame()
else:
    asia_slice_current = asia_slice
    london_slice_current = london_slice

def session_stats_from_slice(slc: pd.DataFrame) -> dict:
    if slc is None or slc.empty:
        return {}
    o = slc.iloc[0]["Open"]
    c = slc.iloc[-1]["Close"]
    h = slc["High"].max()
    l = slc["Low"].min()
    vol = slc["Volume"].sum()
    ret = (c - o) / o if o and not pd.isna(o) else None
    return {"open": o, "close": c, "high": h, "low": l, "volume": vol, "return": ret, "first_ts": slc.index[0] if len(slc.index)>0 else None, "last_ts": slc.index[-1] if len(slc.index)>0 else None}

# compute raw stats for slices
asia_stats_raw = session_stats_from_slice(asia_slice)
london_stats_raw = session_stats_from_slice(london_slice)


# Helper: derive session stats from persisted external ticks if intraday bars are missing
def session_stats_from_external_ticks(df_ticks: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    if df_ticks is None or df_ticks.empty:
        return {}
    # find a datetime-like column
    dt_col = None
    for c in df_ticks.columns:
        if 'time' in c.lower() or 'date' in c.lower() or 'ts' in c.lower() or 'timestamp' in c.lower():
            dt_col = c
            break
    if dt_col is None:
        # fallback: first column
        dt_col = df_ticks.columns[0]
    df_local = df_ticks.copy()
    try:
        df_local[dt_col] = pd.to_datetime(df_local[dt_col], utc=True)
    except Exception:
        try:
            df_local[dt_col] = pd.to_datetime(df_local[dt_col])
        except Exception:
            return {}
    # convert to EST for comparisons
    try:
        df_local[dt_col] = df_local[dt_col].dt.tz_convert(ZoneInfo("America/New_York"))
    except Exception:
        try:
            df_local[dt_col] = df_local[dt_col].dt.tz_localize("UTC").dt.tz_convert(ZoneInfo("America/New_York"))
        except Exception:
            pass

    mask = (df_local[dt_col] >= start) & (df_local[dt_col] < end)
    sel = df_local.loc[mask]
    if sel.empty:
        return {}

    # find a numeric price column
    price_col = None
    for c in sel.columns:
        if c == dt_col:
            continue
        if pd.api.types.is_numeric_dtype(sel[c]):
            price_col = c
            break
    if price_col is None:
        # try common names
        for name in ("price", "tick", "value", "last", "px"):
            if name in sel.columns:
                price_col = name
                break
    if price_col is None:
        return {}

    prices = sel[price_col].astype(float)
    o = prices.iloc[0]
    cval = prices.iloc[-1]
    h = prices.max()
    l = prices.min()
    vol = sel.get('size', sel.get('volume', pd.Series([0]*len(sel)))).sum() if 'size' in sel.columns or 'volume' in sel.columns else float(len(sel))
    return {"open": o, "close": cval, "high": h, "low": l, "volume": vol, "return": (cval - o) / o if o else None, "first_ts": sel[dt_col].iloc[0], "last_ts": sel[dt_col].iloc[-1]}


def analyze_historical_patterns(df: pd.DataFrame, target_date: date | None = None, lookback: int = 60, feature_cols: list | None = None) -> dict:
    """Analyze historical daily data to detect patterns and produce signal metrics and
    a recommended bias.

    Returns a dict with summary stats, stochastic values, tested signal metrics,
    and a `recommended_bias` string plus a short `explanation`.
    """
    import numpy as _np

    if df is None or df.empty:
        return {}

    # normalize index and choose window
    dindex = pd.to_datetime(df.index)
    df_local = df.copy()
    df_local.index = dindex

    if target_date is None:
        end_date = pd.to_datetime(df_local.index.max()).date()
    else:
        end_date = pd.to_datetime(target_date).date()

    start_date = end_date - timedelta(days=lookback)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    win = df_local.loc[(df_local.index >= start_ts) & (df_local.index < end_ts)]
    if win.empty:
        return {}

    # basic stats
    win = win.copy()
    win['Ret'] = win['Close'].pct_change()
    stats = {
        'n_days': int(len(win)),
        'mean_return': float(win['Ret'].mean()) if 'Ret' in win else None,
        'std_return': float(win['Ret'].std()) if 'Ret' in win else None,
        'avg_range_pct': float(((win['High'] - win['Low']) / win['Close']).mean()) if {'High','Low','Close'}.issubset(win.columns) else None,
        'avg_volume': float(win['Volume'].mean()) if 'Volume' in win.columns else None,
        'zscore_mean': float(win['ZScore_Close'].mean()) if 'ZScore_Close' in win.columns else None,
    }

    # stochastic oscillator (K,D) using 14-day default
    k_period = 14
    stoch_k = None
    stoch_d = None
    try:
        low_k = win['Low'].rolling(k_period).min()
        high_k = win['High'].rolling(k_period).max()
        if not pd.isna(low_k.iloc[-1]) and not pd.isna(high_k.iloc[-1]) and (high_k.iloc[-1] - low_k.iloc[-1]) != 0:
            stoch_k = float((win['Close'].iloc[-1] - low_k.iloc[-1]) / (high_k.iloc[-1] - low_k.iloc[-1]) * 100)
            stoch_d = float(win['Close'].pct_change().rolling(3).mean().iloc[-1]) if 'Close' in win else None
    except Exception:
        stoch_k = None
        stoch_d = None

    # short-term momentum (1/5/10 days)
    mom = {}
    try:
        mom['r1'] = float(win['Close'].pct_change(1).iloc[-1])
    except Exception:
        mom['r1'] = None
    for p in (5, 10):
        try:
            mom[f'r{p}'] = float(win['Close'].pct_change(p).iloc[-1])
        except Exception:
            mom[f'r{p}'] = None

    # Candidate signals to test historically (simple heuristics)
    signals = {}
    # ensure MA_20 present or compute
    if 'MA_20' not in win.columns and 'Close' in win.columns:
        win['MA_20'] = win['Close'].rolling(20).mean()

    # define signal lambdas
    cand = {
        'close_gt_MA20': (lambda r: r['Close'] > r.get('MA_20', _np.nan)),
        'z_gt_1': (lambda r: (r.get('ZScore_Close') is not None) and (r.get('ZScore_Close') > 1.0)),
        'rv_gt_1.2': (lambda r: (r.get('RVOL') is not None) and (r.get('RVOL') > 1.2)),
        'prev_up_day': (lambda r: (r.get('Ret') is not None) and (r.get('Ret') > 0)),
        'bullish_day': (lambda r: (r.get('Close') is not None and r.get('Open') is not None) and (r.get('Close') > r.get('Open'))),
    }

    # compute next-day open-close move as target (use next day's close vs open)
    nd_open = win['Open'].shift(-1)
    nd_close = win['Close'].shift(-1)
    nd_move = (nd_close - nd_open) / nd_open

    for name, fn in cand.items():
        vals = []
        sig_series = []
        for i, row in win.iterrows():
            try:
                sig = bool(fn(row))
            except Exception:
                sig = False
            sig_series.append(sig)
        sig_s = pd.Series(sig_series, index=win.index)
        # compute accuracy where next-day move exists
        mask = ~nd_move.isna()
        if mask.sum() == 0:
            signals[name] = {'accuracy': None, 'avg_next_return': None, 'count': int(sig_s.sum()), 'current_signal': bool(sig_s.iloc[-1])}
            continue
        relevant = sig_s & mask
        if relevant.sum() == 0:
            acc = None
            meanr = None
        else:
            # treat positive next-day move as up
            nd_dir = nd_move[relevant] > 0
            acc = float((nd_dir).sum() / len(nd_dir))
            meanr = float(nd_move[relevant].mean())
        signals[name] = {'accuracy': acc, 'avg_next_return': meanr, 'count': int(sig_s.sum()), 'current_signal': bool(sig_s.iloc[-1])}

    # aggregate a simple score from candidate signals (accuracy-weighted)
    score = 0.0
    weight_sum = 0.0
    for n, m in signals.items():
        if m.get('accuracy') is None:
            continue
        w = (m['accuracy'] - 0.5)  # accuracy above random
        sign = 1 if m.get('current_signal') else 0
        score += w * sign
        weight_sum += abs(w)

    # normalize
    bias_score = (score / weight_sum) if weight_sum != 0 else 0.0
    if bias_score > 0.15:
        recommended = 'Bullish'
    elif bias_score < -0.15:
        recommended = 'Bearish'
    else:
        recommended = 'Neutral'

    explanation = f"Pattern analysis over last {len(win)} days suggests {recommended} (score {bias_score:.3f})."

    return {
        'summary': stats,
        'stochastic': {'k': stoch_k, 'd': stoch_d},
        'momentum': mom,
        'signals': signals,
        'recommended_bias': recommended,
        'bias_score': float(bias_score),
        'explanation': explanation,
        'window_start': str(start_date),
        'window_end': str(end_date),
    }
# mark completeness:
# - If the selected date is in the past, consider its sessions complete (even if we lack intraday rows),
#   so the UI will show previous-day session sections and any available stats.
# - If the selected date is today, require the time to have passed AND intraday rows to mark complete.
if sel_date < today_est:
    asia_complete = True
    london_complete = True
else:
    # allow completion if wall-clock has passed the session end AND we have either intraday bars
    # or persisted external ticks covering the session window (tolerance allowed)
    tol = pd.Timedelta(minutes=6)
    # intraday-based checks
    asia_intraday_ok = asia_slice_current is not None and not asia_slice_current.empty and (asia_slice_current.index.max() >= (asia_end - tol))
    london_intraday_ok = london_slice_current is not None and not london_slice_current.empty and (london_slice_current.index.max() >= (london_end - tol))

    # external ticks fallback: load persisted external ticks file once and use it
    try:
        df_ticks = load_external_ticks()
    except Exception:
        df_ticks = pd.DataFrame()
    try:
        asia_ext = session_stats_from_external_ticks(df_ticks, asia_start, asia_end)
    except Exception:
        asia_ext = {}
    try:
        london_ext = session_stats_from_external_ticks(df_ticks, london_start, london_end)
    except Exception:
        london_ext = {}

    asia_ext_ok = bool(asia_ext and asia_ext.get("last_ts") is not None and pd.to_datetime(asia_ext.get("last_ts")) >= (asia_end - tol))
    london_ext_ok = bool(london_ext and london_ext.get("last_ts") is not None and pd.to_datetime(london_ext.get("last_ts")) >= (london_end - tol))

    # mark complete if the session wall-clock has passed, or if we have intraday/external evidence
    asia_complete = (now_est >= asia_end) or asia_intraday_ok or asia_ext_ok
    london_complete = (now_est >= london_end) or london_intraday_ok or london_ext_ok

# Build stats dicts: prefer computed slice stats, but expose a 'complete' flag even when slice is empty for past dates
if asia_slice is not None and not asia_slice.empty:
    asia_stats = {**session_stats_from_slice(asia_slice), "complete": asia_complete}
    # If session is complete, persist the stats for future use
    if asia_complete:
        _persist_session_stats(sel_date, "asia", asia_stats)
else:
    # Check for persisted stats first (these were saved after session completed)
    persisted_asia = _load_session_stats(sel_date, "asia")
    if persisted_asia:
        asia_stats = persisted_asia
        asia_complete = True
    else:
        # fallback to external ticks if available
        try:
            asia_from_ext = session_stats_from_external_ticks(df_ticks, asia_start, asia_end)
        except Exception:
            asia_from_ext = {}
        if asia_from_ext:
            asia_stats = {**asia_from_ext, "complete": asia_complete}
        else:
            asia_stats = {"complete": asia_complete}

if london_slice is not None and not london_slice.empty:
    london_stats = {**session_stats_from_slice(london_slice), "complete": london_complete}
    # If session is complete, persist the stats for future use
    if london_complete:
        _persist_session_stats(sel_date, "london", london_stats)
else:
    # Check for persisted stats first (these were saved after session completed)
    persisted_london = _load_session_stats(sel_date, "london")
    if persisted_london:
        london_stats = persisted_london
        london_complete = True
    else:
        try:
            london_from_ext = session_stats_from_external_ticks(df_ticks, london_start, london_end)
        except Exception:
            london_from_ext = {}
        if london_from_ext:
            london_stats = {**london_from_ext, "complete": london_complete}
        else:
            london_stats = {"complete": london_complete}

# compute previous trading date and its session stats so previous-session sections can reference them
prev_trading_date = None
prev_asia_stats = {"complete": False}
prev_london_stats = {"complete": False}
try:
    # Find the trading date BEFORE the selected date
    # First, try to find it from daily df
    sel_pdt = pd.to_datetime(selected_date)
    prior_idx = df.index[df.index < sel_pdt]
    if len(prior_idx) > 0:
        prev_dt = prior_idx[-1]
        prev_trading_date = pd.to_datetime(prev_dt).date()
    else:
        # Fallback: look in intraday data for any prior trading date
        if not intra.empty:
            intra_dates = pd.to_datetime(intra.index).date
            unique_intra_dates = sorted(set(intra_dates))
            prior_dates = [d for d in unique_intra_dates if d < sel_date]
            if prior_dates:
                prev_trading_date = prior_dates[-1]
    
    # If still not found, check session_stats.json for any available previous trading dates
    if prev_trading_date is None:
        try:
            persisted_all = _load_all_session_stats()
            if persisted_all:
                available_dates = sorted([pd.to_datetime(d).date() for d in persisted_all.keys() if pd.to_datetime(d).date() < sel_date], reverse=True)
                if available_dates:
                    prev_trading_date = available_dates[0]
        except Exception:
            pass

    if prev_trading_date is not None:
        # Previous-day Asia session: use helper to compute 19:00 prior calendar day -> 01:00 prev_trading_date (EST)
        prev_asia_start, prev_asia_end = asia_window_for_date(prev_trading_date)
        prev_asia_slice = intra.loc[(intra.index >= prev_asia_start) & (intra.index < prev_asia_end)] if not intra.empty else pd.DataFrame()
        prev_asia_stats_raw = session_stats_from_slice(prev_asia_slice)
        # For previous trading days, treat sessions as complete (they are in the past).
        # First check for persisted stats, then fall back to computed stats
        persisted_prev_asia = _load_session_stats(prev_trading_date, "asia")
        if persisted_prev_asia:
            prev_asia_stats = persisted_prev_asia
        elif prev_asia_stats_raw:
            prev_asia_stats = {**prev_asia_stats_raw, "complete": True}
        else:
            prev_asia_stats = {"complete": True, "open": None, "close": None, "high": None, "low": None, "volume": 0, "return": None, "first_ts": None, "last_ts": None}

        # Previous-day London session: use helper to compute 03:00 -> 08:00 prev_trading_date (EST)
        prev_london_start, prev_london_end = london_window_for_date(prev_trading_date)
        prev_london_slice = intra.loc[(intra.index >= prev_london_start) & (intra.index < prev_london_end)] if not intra.empty else pd.DataFrame()
        prev_london_stats_raw = session_stats_from_slice(prev_london_slice)
        # First check for persisted stats, then fall back to computed stats
        persisted_prev_london = _load_session_stats(prev_trading_date, "london")
        if persisted_prev_london:
            prev_london_stats = persisted_prev_london
        elif prev_london_stats_raw:
            prev_london_stats = {**prev_london_stats_raw, "complete": True}
        else:
            prev_london_stats = {"complete": True, "open": None, "close": None, "high": None, "low": None, "volume": 0, "return": None, "first_ts": None, "last_ts": None}
except Exception:
    prev_trading_date = None
    prev_asia_stats = {"complete": False}
    prev_london_stats = {"complete": False}


if selected_tab == "Summary":
    st.header(f"Summary for {selected_date.isoformat()}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Close", f"{selected_row['Close']:.2f}")
    col1.metric("Open", f"{selected_row['Open']:.2f}")
    col1.metric("High", f"{selected_row['High']:.2f}")
    col1.metric("Low", f"{selected_row['Low']:.2f}")

    col2.metric("Volume", f"{selected_row['Volume']:.0f}")
    col2.metric("RVOL", f"{selected_row.get('RVOL', float('nan')):.2f}")
    col2.metric("Volatility (20d)", f"{selected_row.get('Volatility', float('nan')):.4f}")

    col3.metric("VWAP Daily", f"{selected_row.get('VWAP_Daily', float('nan')):.2f}")
    col3.metric("VWAP Weekly", f"{selected_row.get('VWAP_Weekly', float('nan')):.2f}")
    col3.metric("Z-Score (Close)", f"{selected_row.get('ZScore_Close', float('nan')):.2f}")

    # Recompute Daily Bias for the finished trading day using the prior trading day's data
    # (source = trading day before selected_date) and the selected day's Asia/London session slices.
    display_expl = selected_row.get('Bias_Explanation', '')
    if display_expl and not str(display_expl).startswith('Explanation for'):
        display_expl = f"Explanation for {selected_date.isoformat()}: {display_expl}"

    recomputed_bias = None
    recomputed_expl = None
    recomputed_vol = None
    try:
        prior_idx = df.index[pd.to_datetime(df.index).date < selected_date]
        if len(prior_idx) > 0:
            src_for_selected = df.loc[prior_idx[-1]]
        else:
            src_for_selected = df.iloc[-1]
        vol_history = df["Volatility"] if "Volatility" in df.columns else None

        # pass in the selected day's session stats so the predictor can incorporate them
        try:
            recomputed_bias, recomputed_expl, recomputed_vol = predict_next_day_bias(
                src_for_selected,
                vol_history=vol_history,
                asia_stats=asia_stats if asia_stats and asia_stats.get("first_ts") is not None else None,
                london_stats=london_stats if london_stats and london_stats.get("first_ts") is not None else None,
                forexfactory={
                    "selected": {
                        "score": selected_row.get("FF_EventScore"),
                        "high": selected_row.get("FF_HighImpactCount"),
                        "mid": selected_row.get("FF_MidImpactCount"),
                    },
                    "forecast": None,
                },
            )
        except Exception:
            recomputed_bias = None

    except Exception:
        recomputed_bias = None

    # compute actual move and accuracy only if the trading day is complete
    accuracy_msg = None
    try:
        if not trading_day_over:
            accuracy_msg = "Trading day not complete — accuracy will be evaluated after the day finishes."
        else:
            o = selected_row.get("Open")
            c = selected_row.get("Close")
            if o is None or c is None or pd.isna(o) or pd.isna(c):
                accuracy_msg = "No price data available to evaluate bias accuracy."
            else:
                actual_dir = "up" if c > o else ("down" if c < o else "flat")
                move_pct = (c - o) / o if o else 0
                # determine predicted direction text
                ptxt = str(recomputed_bias or selected_row.get('Daily_Bias', '') or "").lower()
                if any(k in ptxt for k in ("bull", "buy", "long", "up")):
                    pred_dir = "up"
                elif any(k in ptxt for k in ("bear", "sell", "short", "down")):
                    pred_dir = "down"
                else:
                    pred_dir = "neutral"

                if pred_dir == "neutral":
                    accuracy_msg = f"Predicted: {recomputed_bias or selected_row.get('Daily_Bias', '')} (neutral). Actual: {actual_dir} ({move_pct:.2%})."
                else:
                    correct = (pred_dir == actual_dir)
                    accuracy_pct = abs(move_pct) if correct else 0.0
                    verdict = "Correct" if correct else "Incorrect"
                    accuracy_msg = f"Predicted: {recomputed_bias or selected_row.get('Daily_Bias', '')} — {verdict}. Actual: {actual_dir} ({move_pct:.2%}). Accuracy: {accuracy_pct:.2%}."
    except Exception:
        accuracy_msg = None

    st.subheader("Daily Bias")
    st.write(f"**Bias:** {recomputed_bias or selected_row.get('Daily_Bias', '')}")
    st.write(get_overextension_status(selected_row))
    st.write(display_expl)
    try:
        if accuracy_msg:
            st.write(accuracy_msg)
    except Exception:
        pass
    


if selected_tab == "Price & MAs":
    st.header("Price & Moving Averages")
    chart_df = df[["Close", "MA_10", "MA_20", "MA_50", "MA_100", "MA_200"]].copy()
    # keep rows with a price and forward-fill moving averages so charts render
    chart_df.index = pd.to_datetime(chart_df.index)
    chart_df = chart_df[chart_df["Close"].notna()].ffill()
    if chart_df.empty:
        st.write("Not enough data to render moving averages.")
    else:
        st.line_chart(chart_df)

    # interpretation based on latest available point
    last = chart_df.iloc[-1]
    interpr = []
    if last["Close"] > last.get("MA_20", float("nan")):
        interpr.append("Close > MA20 (short-term bullish)")
    else:
        interpr.append("Close ≤ MA20 (short-term bearish)")
    if last["Close"] > last.get("MA_50", float("nan")):
        interpr.append("Close > MA50 (intermediate bullish)")
    else:
        interpr.append("Close ≤ MA50 (intermediate bearish)")
    st.subheader("Interpretation")
    for s in interpr:
        st.write(f"- {s}")


if selected_tab == "VWAP":
    st.header("VWAP vs Price")
    vwap_df = df[["Close", "VWAP_Daily", "VWAP_Weekly"]].dropna()
    vwap_df.index = pd.to_datetime(vwap_df.index)
    st.line_chart(vwap_df)


if selected_tab == "Indicators":
    with tabs[4]:
        st.header("Indicators")
    ind_df = df[
        ["ZScore_Close", "Volatility", "ROC_10", "Momentum_10", "RVOL"]
    ].dropna()
    ind_df.index = pd.to_datetime(ind_df.index)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Z-Score (Close)")
        st.line_chart(ind_df[["ZScore_Close"]])
        st.subheader("Volatility (20d)")
        st.line_chart(ind_df[["Volatility"]])
        try:
            latest_ind = ind_df.iloc[-1]
            summ = summarize_indicators(latest_ind)
            st.write(summ.get("ZScore_Close"))
            st.write(summ.get("Volatility"))
        except Exception:
            pass
    with c2:
        st.subheader("ROC 10")
        st.line_chart(ind_df[["ROC_10"]])
        st.subheader("Momentum 10")
        st.line_chart(ind_df[["Momentum_10"]])
        st.subheader("RVOL")
        st.line_chart(ind_df[["RVOL"]])
        try:
            st.write(summ.get("ROC_10"))
        except Exception:
            pass
        try:
            st.write(summ.get("Momentum_10"))
        except Exception:
            pass
        try:
            st.write(summ.get("RVOL"))
        except Exception:
            pass


if selected_tab == "Previous Days Stats":
    with tabs[8]:
        st.header("Previous Days Stats")

    stats_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "RVOL",
        "Volatility",
        "VWAP_Daily",
        "VWAP_Weekly",
        "ZScore_Close",
        "ROC_10",
        "Momentum_10",
        "Daily_Bias",
    ]
    table_df = df[stats_cols].copy()
    table_df["Overextension"] = [
        get_overextension_status(df.loc[d]) for d in df.index
    ]
    table_df.index.name = "Date"
    styled = style_stats_table(table_df.reset_index())
    st.dataframe(styled, width="stretch")


if selected_tab == "Compare Days":
    with tabs[9]:
        st.header("Compare Days")

    col1, col2 = st.columns(2)
    with col1:
        d1 = st.date_input(
            "First date",
            value=latest_date,
            min_value=all_dates[0],
            max_value=latest_date,
            key="cmp1",
        )
    with col2:
        d2 = st.date_input(
            "Second date",
            value=all_dates[-2] if len(all_dates) > 1 else latest_date,
            min_value=all_dates[0],
            max_value=latest_date,
            key="cmp2",
        )

    if isinstance(d1, datetime):
        d1 = d1.date()
    if isinstance(d2, datetime):
        d2 = d2.date()

    pdt_d1 = pd.to_datetime(d1)
    pdt_d2 = pd.to_datetime(d2)
    if pdt_d1 in df.index and pdt_d2 in df.index:
        r1 = df.loc[pdt_d1]
        r2 = df.loc[pdt_d2]
        metrics = [
            "Close",
            "Volume",
            "RVOL",
            "Volatility",
            "VWAP_Daily",
            "VWAP_Weekly",
            "ZScore_Close",
            "ROC_10",
            "Momentum_10",
        ]
        comp_data = []
        for m in metrics:
            comp_data.append(
                {
                    "Metric": m,
                    "First Date": d1.isoformat(),
                    "First Value": r1.get(m, float("nan")),
                    "Second Date": d2.isoformat(),
                    "Second Value": r2.get(m, float("nan")),
                    "Difference": r2.get(m, float("nan")) - r1.get(m, float("nan")),
                }
            )
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, width="stretch")
    else:
        st.write("Select valid dates within the data range.")


if selected_tab == "Day Summary":
    with tabs[1]:
        st.header(f"Day Summary: {selected_date.isoformat()}")

    summary = {
        "Date": selected_date.isoformat(),
        "Open": selected_row["Open"],
        "High": selected_row["High"],
        "Low": selected_row["Low"],
        "Close": selected_row["Close"],
        "Volume": selected_row["Volume"],
        "MA_20": selected_row.get("MA_20", float("nan")),
        "MA_50": selected_row.get("MA_50", float("nan")),
        "Volatility": selected_row.get("Volatility", float("nan")),
        "VWAP_Daily": selected_row.get("VWAP_Daily", float("nan")),
        "VWAP_Weekly": selected_row.get("VWAP_Weekly", float("nan")),
        "ZScore_Close": selected_row.get("ZScore_Close", float("nan")),
        "RVOL": selected_row.get("RVOL", float("nan")),
        "ROC_10": selected_row.get("ROC_10", float("nan")),
        "Momentum_10": selected_row.get("Momentum_10", float("nan")),
        "Daily_Bias": selected_row.get("Daily_Bias", ""),
        "Overextension": get_overextension_status(selected_row),
    }

    st.json(summary)

    # Z-score textual summary
    try:
        from indicators import summarize_indicators
        zsum = summarize_indicators(selected_row).get("ZScore_Close")
        st.subheader("Z-score Summary")
        st.write(zsum)
    except Exception:
        pass

    st.subheader("Notes for this day")
    if existing_notes:
        for i, n in enumerate(existing_notes, 1):
            st.write(f"{i}. {n}")
    else:
        st.write("(No notes for this date.)")


if selected_tab == "Asia Session":
    with tabs[6]:
        st.header("Asia Session")
    st.write("Session window: 19:00 previous day → 01:00 current day (EST)")
    # Current-day Asia session (top)
    st.subheader("Current Day Asia Session")
    if now_est < asia_start:
        st.info("Asia session has not started yet for this date — check back when the session is over.")
    else:
        # Show stats if available (either from slice or from persisted data)
        if asia_stats.get("first_ts") is not None or asia_stats.get("open") is not None:
            status = "(complete)" if asia_stats.get("complete", False) else "(in progress)"
            st.write(f"Session status: {status}")
            st.write(asia_stats)
            # range and adjective
            if asia_stats.get("high") is not None and asia_stats.get("low") is not None:
                rng = asia_stats["high"] - asia_stats["low"]
                ref = selected_row.get("Close", 1.0)
                pct = rng / ref if ref else 0
                if pct >= 0.01:
                    size_adj = "Huge"
                elif pct >= 0.005:
                    size_adj = "Big"
                elif pct >= 0.002:
                    size_adj = "Medium/Average"
                else:
                    size_adj = "Small"
                st.write(f"Range: {rng:.2f} ({pct:.3%}) — {size_adj}")

                # trend detection (only if we have actual slice data)
                if asia_slice is not None and not asia_slice.empty and asia_slice.shape[0] >= 2:
                    slc = asia_slice
                    s_first = slc.iloc[0]["Open"]
                    s_last = slc.iloc[-1]["Close"]
                    slope = (s_last - s_first) / s_first if s_first else 0
                    stdp = slc["Close"].std()
                    if abs(slope) < 0.002 and stdp < rng * 0.2:
                        state = "Mostly consolidating"
                    elif slope >= 0.002:
                        state = "Bullish trend"
                    elif slope <= -0.002:
                        state = "Bearish trend"
                    else:
                        state = "Mixed"
                    st.write(f"Behavior: {state} (slope {slope:.4f}, std {stdp:.4f})")
        else:
            if not asia_stats.get("complete", False):
                st.info("Asia session data is unavailable for this date — check back when the session is over.")
            else:
                st.write("No Asia session data available for this date.")

    # Previous-day Asia session (below)
    st.subheader("Previous Trading Day Asia Session")
    # If prev_trading_date wasn't found earlier, attempt to locate one from persisted stats
    if prev_trading_date is None:
        try:
            persisted_all = _load_all_session_stats()
            if persisted_all:
                available_dates = sorted([pd.to_datetime(d).date() for d in persisted_all.keys() if pd.to_datetime(d).date() < sel_date], reverse=True)
                if available_dates:
                    cand = available_dates[0]
                    persisted_prev_asia = _load_session_stats(cand, "asia")
                    if persisted_prev_asia:
                        prev_trading_date = cand
                        prev_asia_stats = persisted_prev_asia
        except Exception:
            pass

    if prev_trading_date is not None:
        # Show if complete (persisted data) or if has actual data values
        if prev_asia_stats.get("complete", False) or (prev_asia_stats.get("open") is not None and isinstance(prev_asia_stats.get("open"), (int, float))):
            st.write(f"Session for previous trading date: {prev_trading_date.isoformat()}")
            st.write(prev_asia_stats)
        else:
            st.write(f"No Asia session data available for previous trading date: {prev_trading_date.isoformat()}")
    else:
        st.write("No previous trading date found.")


if selected_tab == "London Session":
    with tabs[7]:
        st.header("London Session")
    st.write("Session window: 03:00 → 08:00 (EST)")
    # Current-day London session (top)
    st.subheader("Current Day London Session")
    if now_est < london_start:
        st.info("London session has not started yet for this date — check back when the session is over.")
    else:
        # Show stats if available (either from slice or from persisted data)
        if london_stats.get("first_ts") is not None or london_stats.get("open") is not None:
            status = "(complete)" if london_stats.get("complete", False) else "(in progress)"
            st.write(f"Session status: {status}")
            st.write(london_stats)
            # range/trend
            if london_stats.get("high") is not None and london_stats.get("low") is not None:
                rng = london_stats["high"] - london_stats["low"]
                ref = selected_row.get("Close", 1.0)
                pct = rng / ref if ref else 0
                if pct >= 0.01:
                    size_adj = "Huge"
                elif pct >= 0.005:
                    size_adj = "Big"
                elif pct >= 0.002:
                    size_adj = "Medium/Average"
                else:
                    size_adj = "Small"
                st.write(f"Range: {rng:.2f} ({pct:.3%}) — {size_adj}")

                # trend detection (only if we have actual slice data)
                if london_slice is not None and not london_slice.empty and london_slice.shape[0] >= 2:
                    slc = london_slice
                    s_first = slc.iloc[0]["Open"]
                    s_last = slc.iloc[-1]["Close"]
                    slope = (s_last - s_first) / s_first if s_first else 0
                    stdp = slc["Close"].std()
                    if abs(slope) < 0.002 and stdp < rng * 0.2:
                        state = "Mostly consolidating"
                    elif slope >= 0.002:
                        state = "Bullish trend"
                    elif slope <= -0.002:
                        state = "Bearish trend"
                    else:
                        state = "Mixed"
                    st.write(f"Behavior: {state} (slope {slope:.4f}, std {stdp:.4f})")

                    # Asia range break detection
                    if asia_stats and asia_stats.get("high") is not None and asia_stats.get("low") is not None:
                        a_high = asia_stats["high"]
                        a_low = asia_stats["low"]
                        broke = None
                        for ts, rowv in slc.iterrows():
                            if rowv["High"] > a_high or rowv["Low"] < a_low:
                                broke = ts
                                break
                        if broke is not None:
                            st.write(f"London broke Asia range at {broke.tz_convert(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S %Z')}")
                        else:
                            st.write("London session did not break the Asia range.")

                        l_close = london_stats.get("close")
                        if l_close is not None:
                            in_range = (l_close <= a_high) and (l_close >= a_low)
                            st.write("London close is inside Asia range." if in_range else "London close is outside Asia range.")
        else:
            if not london_stats.get("complete", False):
                st.info("London session data is unavailable for this date — check back when the session is over.")
            else:
                st.write("No London session data available for this date.")

    # Previous-day London session (below)
    st.subheader("Previous Trading Day London Session")
    # If prev_trading_date wasn't found earlier, attempt to locate one from persisted stats
    if prev_trading_date is None:
        try:
            persisted_all = _load_all_session_stats()
            if persisted_all:
                available_dates = sorted([pd.to_datetime(d).date() for d in persisted_all.keys() if pd.to_datetime(d).date() < sel_date], reverse=True)
                if available_dates:
                    cand = available_dates[0]
                    persisted_prev_london = _load_session_stats(cand, "london")
                    if persisted_prev_london:
                        prev_trading_date = cand
                        prev_london_stats = persisted_prev_london
        except Exception:
            pass

    if prev_trading_date is not None:
        # Show if complete (persisted data) or if has actual data values
        if prev_london_stats.get("complete", False) or (prev_london_stats.get("open") is not None and isinstance(prev_london_stats.get("open"), (int, float))):
            st.write(f"Session for previous trading date: {prev_trading_date.isoformat()}")
            st.write(prev_london_stats)
        else:
            st.write(f"No London session data available for previous trading date: {prev_trading_date.isoformat()}")
    else:
        st.write("No previous trading date found.")


if selected_tab == "Next-Day Forecast":
    with tabs[5]:
        # Next-Day Forecast (dedicated tab)
        st.header("Next-Day Forecast")
    # Use the selected date as the forecast day (user expectation)
    forecast_day = selected_date
    st.subheader(f"Forecast for: {forecast_day.isoformat()} at 9:30am EST")

    # prepare intraday session slices for the forecast date (Asia: prior day 19:00->01:00, London: 03:00->08:00)
    fd = pd.to_datetime(forecast_day).date()
    fd_asia_start, fd_asia_end = asia_window_for_date(fd)
    fd_asia_slice = intra.loc[(intra.index >= fd_asia_start) & (intra.index < fd_asia_end)] if not intra.empty else pd.DataFrame()
    fd_asia_stats_raw = session_stats_from_slice(fd_asia_slice)

    fd_london_start, fd_london_end = london_window_for_date(fd)
    fd_london_slice = intra.loc[(intra.index >= fd_london_start) & (intra.index < fd_london_end)] if not intra.empty else pd.DataFrame()
    fd_london_stats_raw = session_stats_from_slice(fd_london_slice)

    # Determine completeness using intraday coverage (allow small tolerance) or wall-clock
    tol = pd.Timedelta(minutes=6)
    fd_asia_has = fd_asia_slice is not None and not fd_asia_slice.empty
    fd_london_has = fd_london_slice is not None and not fd_london_slice.empty
    fd_asia_reached = fd_asia_has and (fd_asia_slice.index.max() >= (fd_asia_end - tol))
    fd_london_reached = fd_london_has and (fd_london_slice.index.max() >= (fd_london_end - tol))
    fd_asia_complete = (now_est >= fd_asia_end and fd_asia_has) or fd_asia_reached
    fd_london_complete = (now_est >= fd_london_end and fd_london_has) or fd_london_reached

    # Check for persisted stats as fallback (these are saved after sessions complete)
    persisted_fd_asia = _load_session_stats(fd, "asia")
    persisted_fd_london = _load_session_stats(fd, "london")
    
    # Use persisted stats if available and mark as complete
    if persisted_fd_asia and not fd_asia_stats_raw:
        fd_asia_stats = persisted_fd_asia
        fd_asia_complete = True
    else:
        fd_asia_stats = {**fd_asia_stats_raw, "complete": fd_asia_complete} if fd_asia_stats_raw else {"complete": False}
        # Persist if complete
        if fd_asia_complete and fd_asia_stats_raw:
            _persist_session_stats(fd, "asia", fd_asia_stats)
    
    if persisted_fd_london and not fd_london_stats_raw:
        fd_london_stats = persisted_fd_london
        fd_london_complete = True
    else:
        fd_london_stats = {**fd_london_stats_raw, "complete": fd_london_complete} if fd_london_stats_raw else {"complete": False}
        # Persist if complete
        if fd_london_complete and fd_london_stats_raw:
            _persist_session_stats(fd, "london", fd_london_stats)

    # source row for baseline calculations: use the last trading row before the forecast day
    try:
        # compare by calendar date to avoid tz/partial-timestamp issues
        prior_idx_fd = df.index[pd.to_datetime(df.index).date < fd]
        if len(prior_idx_fd) > 0:
            src_idx = prior_idx_fd[-1]
            src_row = df.loc[src_idx]
        else:
            src_idx = df.index[-1]
            src_row = df.iloc[-1]
    except Exception:
        src_idx = df.index[-1]
        src_row = df.iloc[-1]

    from bias_engine import predict_next_day_bias
    vol_history = df["Volatility"] if "Volatility" in df.columns else None

    # prepare ForexFactory feature dicts for selected and forecast dates
    selected_ff = None
    try:
        selected_ff = {
            "score": src_row.get("FF_EventScore"),
            "high": src_row.get("FF_HighImpactCount"),
            "mid": src_row.get("FF_MidImpactCount"),
        }
    except Exception:
        selected_ff = None

    forecast_ff = None
    try:
        fff = compute_forexfactory_features([forecast_day])
        if not fff.empty and forecast_day in fff.index:
            row_ff = fff.loc[forecast_day]
            forecast_ff = {"score": int(row_ff.get("FF_EventScore", 0) if not pd.isna(row_ff.get("FF_EventScore", 0)) else 0), "high": int(row_ff.get("FF_HighImpactCount", 0) if not pd.isna(row_ff.get("FF_HighImpactCount", 0)) else 0), "mid": int(row_ff.get("FF_MidImpactCount", 0) if not pd.isna(row_ff.get("FF_MidImpactCount", 0)) else 0)}
    except Exception:
        forecast_ff = None

    baseline_bias, baseline_expl, baseline_vol = predict_next_day_bias(
        src_row, vol_history=vol_history, asia_stats=None, london_stats=None, forexfactory={"selected": selected_ff, "forecast": forecast_ff}
    )

    # Show which historical row was used as the baseline
    try:
        src_date_display = pd.to_datetime(src_idx).date().isoformat()
    except Exception:
        try:
            src_date_display = str(src_idx)
        except Exception:
            src_date_display = "(unknown)"
    st.write(f"Baseline data source date: {src_date_display} (last trading row before forecast date)")

    asia_bias = None
    asia_expl = None
    if fd_asia_stats and fd_asia_stats.get("first_ts") is not None and fd_asia_stats.get("complete", False):
        asia_bias, asia_expl, _ = predict_next_day_bias(
            src_row, vol_history=vol_history, asia_stats=fd_asia_stats, london_stats=None, forexfactory={"selected": selected_ff, "forecast": forecast_ff}
        )

    final_bias = None
    final_expl = None
    if fd_london_stats and fd_london_stats.get("first_ts") is not None and fd_london_stats.get("complete", False):
        final_bias, final_expl, _ = predict_next_day_bias(
            src_row, vol_history=vol_history, asia_stats=fd_asia_stats if fd_asia_stats else None, london_stats=fd_london_stats, forexfactory={"selected": selected_ff, "forecast": forecast_ff}
        )

    # determine completion status for forecast date
    if not fd_asia_stats.get("complete", False) and not fd_london_stats.get("complete", False):
        pct = 33
        status_msg = "Waiting on Asia session and London session to finish for the full updated bias."
        current_bias = baseline_bias
        current_expl = baseline_expl
    elif fd_asia_stats.get("complete", False) and not fd_london_stats.get("complete", False):
        pct = 66
        status_msg = "Asia session finished — waiting on London session to finalize the bias."
        current_bias = asia_bias if asia_bias is not None else baseline_bias
        current_expl = asia_expl if asia_expl is not None else baseline_expl
    else:
        pct = 100
        status_msg = "All sessions finished — bias is finalized for this forecast period."
        current_bias = final_bias if final_bias is not None else (asia_bias if asia_bias is not None else baseline_bias)
        current_expl = final_expl if final_expl is not None else (asia_expl if asia_expl is not None else baseline_expl)

    # show that baseline comes from the prior trading day, and sessions are for the selected date
    try:
        src_date_display = pd.to_datetime(src_idx).date().isoformat()
    except Exception:
        try:
            src_date_display = str(src_idx)
        except Exception:
            src_date_display = "(unknown)"

    st.subheader(f"Forecast based on previous trading day: {src_date_display} — {pct}% complete")
    st.write(f"(Session data used: Asia & London for selected date {selected_date.isoformat()})")
    if pct < 100:
        st.info(status_msg)
    else:
        st.success(status_msg)

    st.write(f"**Open Bias (current):** {current_bias}")
    st.write(f"**Volatility Expectation (baseline):** {baseline_vol}")
    st.write("**Reasoning (current):**")
    st.write(current_expl)

    # Include ForexFactory-derived signals in the reasoning when available
    try:
        ff_notes = []
        # features joined into df for selected_date via build_base_dataframe
        if isinstance(src_row, (pd.Series,)):
            ff_score = src_row.get("FF_EventScore")
            ff_high = src_row.get("FF_HighImpactCount")
            ff_mid = src_row.get("FF_MidImpactCount")
            if ff_score is not None and not pd.isna(ff_score):
                ff_notes.append(f"Selected date ForexFactory event score: {ff_score} (high:{int(ff_high) if ff_high is not None and not pd.isna(ff_high) else 0}, mid:{int(ff_mid) if ff_mid is not None and not pd.isna(ff_mid) else 0})")

        # also check forecast date calendar directly
        try:
            fff = compute_forexfactory_features([forecast_day])
            if not fff.empty and forecast_day in fff.index:
                row_ff = fff.loc[forecast_day]
                fh = int(row_ff.get("FF_HighImpactCount", 0)) if not pd.isna(row_ff.get("FF_HighImpactCount", 0)) else 0
                fm = int(row_ff.get("FF_MidImpactCount", 0)) if not pd.isna(row_ff.get("FF_MidImpactCount", 0)) else 0
                fs = int(row_ff.get("FF_EventScore", 0)) if not pd.isna(row_ff.get("FF_EventScore", 0)) else 0
                ff_notes.append(f"Forecast date ForexFactory events — high:{fh}, mid:{fm}, score:{fs}")
        except Exception:
            pass

        if ff_notes:
            st.subheader("ForexFactory signals")
            for n in ff_notes:
                st.write(f"- {n}")
    except Exception:
        pass

    st.subheader("Assumptions / What to watch for")
    assumps = []
    if not pd.isna(src_row.get("VWAP_Daily")):
        assumps.append("If price gaps above daily VWAP at open, expect bullish follow-through.")
    if not pd.isna(src_row.get("VWAP_Weekly")):
        assumps.append("If price opens below weekly VWAP, watch for range-bound or bearish continuation.")
    if not pd.isna(src_row.get("RVOL")) and src_row.get("RVOL") > 1.3:
        assumps.append("Elevated RVOL suggests higher conviction moves at open.")
    if assumps:
        for a in assumps:
            st.write(f"- {a}")
    else:
        st.write("- No specific edge detected; monitor opening prints and VWAP levels.")

    # show session summaries for the forecast date
    st.subheader("Session Summaries (EST) for forecast date")
    st.write("Asia session (19:00 previous day → 01:00 current day):")
    if fd_asia_stats.get("first_ts") is not None:
        if not fd_asia_stats.get("complete", False):
            st.write("Asia session not yet finished — showing partial stats.")
        st.write(fd_asia_stats)
    else:
        st.write("No Asia session data available for forecast date.")

    st.write("London session (03:00 → 08:00 current day):")
    if fd_london_stats.get("first_ts") is not None:
        if not fd_london_stats.get("complete", False):
            st.write("London session not yet finished — showing partial stats.")
        st.write(fd_london_stats)
    else:
        st.write("No London session data available for forecast date.")

    # previous trading day relative to forecast date
    prev_fd = None
    try:
        prior_idx_fd = df.index[pd.to_datetime(df.index).date < fd]
        if len(prior_idx_fd) > 0:
            prev_fd = pd.to_datetime(prior_idx_fd[-1]).date()
    except Exception:
        prev_fd = None

    st.subheader("Previous Trading Day Sessions (for forecast date)")
    if prev_fd is not None:
        # previous-day Asia
        p_asia_start, p_asia_end = asia_window_for_date(prev_fd)
        p_asia_slice = intra.loc[(intra.index >= p_asia_start) & (intra.index < p_asia_end)] if not intra.empty else pd.DataFrame()
        p_asia_stats = session_stats_from_slice(p_asia_slice)
        # Check for persisted stats as fallback
        if not p_asia_stats:
            p_asia_stats = _load_session_stats(prev_fd, "asia")
        if p_asia_stats:
            st.write(f"Previous trading date: {prev_fd.isoformat()} (Asia)")
            st.write({**p_asia_stats, "complete": True})
        else:
            st.write(f"No previous-day Asia session data for {prev_fd.isoformat()}.")

        # previous-day London
        p_london_start, p_london_end = london_window_for_date(prev_fd)
        p_london_slice = intra.loc[(intra.index >= p_london_start) & (intra.index < p_london_end)] if not intra.empty else pd.DataFrame()
        p_london_stats = session_stats_from_slice(p_london_slice)
        # Check for persisted stats as fallback
        if not p_london_stats:
            p_london_stats = _load_session_stats(prev_fd, "london")
        if p_london_stats:
            st.write(f"Previous trading date: {prev_fd.isoformat()} (London)")
            st.write({**p_london_stats, "complete": True})
        else:
            st.write(f"No previous-day London session data for {prev_fd.isoformat()}.")
    else:
        st.write("No previous trading date found for forecast date.")


def render_summary_search():
    st.header("Summary Search")
    st.write("Type a date (YYYY-MM-DD) or pick a date to view that day's full summary and stats.")
    txt = st.text_input("Enter date (YYYY-MM-DD)", "", key="ss_txt")
    dpick = st.date_input("Or pick a date", value=latest_date, min_value=all_dates[0], max_value=latest_date, key="ss_date")

    query_date = None
    if txt.strip():
        try:
            query_date = pd.to_datetime(txt.strip()).date()
        except Exception:
            st.error("Couldn't parse the typed date. Use YYYY-MM-DD.")
    else:
        query_date = dpick if isinstance(dpick, date) else dpick.date()

    if query_date is not None:
        pdt_q = pd.to_datetime(query_date)
        if pdt_q in df.index:
            row = df.loc[pdt_q]
            st.subheader(f"Summary for {query_date.isoformat()}")
            st.json({
                "Date": query_date.isoformat(),
                "Open": row.get("Open"),
                "High": row.get("High"),
                "Low": row.get("Low"),
                "Close": row.get("Close"),
                "Volume": row.get("Volume"),
                "MA_20": row.get("MA_20"),
                "MA_50": row.get("MA_50"),
                "Volatility": row.get("Volatility"),
                "VWAP_Daily": row.get("VWAP_Daily"),
                "VWAP_Weekly": row.get("VWAP_Weekly"),
                "ZScore_Close": row.get("ZScore_Close"),
                "RVOL": row.get("RVOL"),
                "ROC_10": row.get("ROC_10"),
                "Momentum_10": row.get("Momentum_10"),
                "Daily_Bias": row.get("Daily_Bias"),
                "Bias_Explanation": row.get("Bias_Explanation"),
            })
            # trading suggestion (extracted from explanation if present)
            expl = row.get("Bias_Explanation", "")
            st.subheader("Trade Suggestion")
            if "Suggestion:" in expl:
                try:
                    sug = expl.split("Suggestion:", 1)[1].split(".", 1)[0].strip()
                    st.write(sug)
                except Exception:
                    st.write("No explicit suggestion available.")
            else:
                st.write("No explicit suggestion available.")
        else:
            st.write("No data for that date — make sure it's within the dataset range.")

if selected_tab == "Summary Search":
    with tabs[10]:
        render_summary_search()

# If user wants to open the summary search from the sidebar (when tabs overflow), render it here as well
if st.session_state.get("open_summary_search"):
    st.markdown("---")
    render_summary_search()


if selected_tab == "News & Events":
    with tabs[11]:
        # anchor for quick-jump links
        st.markdown('<a id="news-section"></a>', unsafe_allow_html=True)
        st.header("News & Events")
        st.write(f"Showing Seeking Alpha and ForexFactory items for selected date: {selected_date.isoformat()}")

        # Seeking Alpha news for the symbol (selected date and previous trading day)
        try:
            sa_news_today = fetch_seekingalpha_news(POLY_TICKER, max_items=8)
        except Exception:
            sa_news_today = []

        # Previous trading date news (baseline)
        try:
            prior_idx = df.index[pd.to_datetime(df.index).date < selected_date]
            if len(prior_idx) > 0:
                prev_dt = pd.to_datetime(prior_idx[-1]).date()
            else:
                prev_dt = None
        except Exception:
            prev_dt = None

        st.subheader("Seeking Alpha — Selected Date")
        pos_k = ['beat','upgrade','raise','positive','profit','growth','record','outperform','beat','good','strong']
        neg_k = ['miss','downgrade','cut','warn','loss','negative','sell','disappoint','weak','fail']
        if sa_news_today:
            sa_summ_scores = []
            for a in sa_news_today:
                text = (a.get('title','') + ' ' + a.get('snippet','')).lower()
                score = sum(text.count(k) for k in pos_k) - sum(text.count(k) for k in neg_k)
                sent = 'Positive' if score>0 else ('Negative' if score<0 else 'Neutral')
                st.write(f"- {a.get('title')} — {sent}")
                sa_summ_scores.append(score)
            overall = sum(sa_summ_scores)
            overall_sent = 'Positive' if overall>0 else ('Negative' if overall<0 else 'Neutral')
            st.write(f"Overall Seeking Alpha sentiment (selected date): {overall_sent}")
        else:
            st.write("No Seeking Alpha news found for selected date.")

    st.subheader("Seeking Alpha — Previous Trading Day")
    if prev_dt is not None:
        try:
            sa_prev = fetch_seekingalpha_news(POLY_TICKER, max_items=6)
        except Exception:
            sa_prev = []
        if sa_prev:
            prev_scores = []
            for a in sa_prev:
                text = (a.get('title','') + ' ' + a.get('snippet','')).lower()
                score = sum(text.count(k) for k in pos_k) - sum(text.count(k) for k in neg_k)
                prev_scores.append(score)
                st.write(f"- {a.get('title')} — {'Positive' if score>0 else ('Negative' if score<0 else 'Neutral')}")
            overall_prev = sum(prev_scores)
            st.write(f"Overall Seeking Alpha sentiment (previous trading day): {'Positive' if overall_prev>0 else ('Negative' if overall_prev<0 else 'Neutral')}")
        else:
            st.write("No Seeking Alpha news found for previous trading day.")
    else:
        st.write("No previous trading day found to fetch Seeking Alpha news.")

    # ForexFactory events for selected date and previous trading date
    st.subheader("ForexFactory Calendar — Selected Date")
    try:
        ff_events = _fetch_forexfactory_for_date(pd.to_datetime(selected_date).date())
        if ff_events:
            hi = sum(1 for e in ff_events if e.get('impact','').lower()=='high')
            mid = sum(1 for e in ff_events if e.get('impact','').lower()=='medium')
            low = sum(1 for e in ff_events if e.get('impact','').lower()=='low')
            st.write(f"High:{hi}, Mid:{mid}, Low:{low}")
            for e in ff_events:
                st.write(f"- {e.get('time')} {e.get('currency')} {e.get('impact')} {e.get('event')}")
            if hi>0:
                st.write("Summary: High-impact economic events scheduled — expect higher volatility and possible gap/open reactions.")
            elif mid>0:
                st.write("Summary: Moderate-impact events scheduled — monitor for intraday volatility.")
            else:
                st.write("Summary: No major economic events — market moves likely price/flow-driven.")
        else:
            st.write("No ForexFactory events found for selected date.")
    except Exception:
        st.write("Failed to fetch ForexFactory events for selected date.")

    st.subheader("ForexFactory Calendar — Previous Trading Day")
    if prev_dt is not None:
        try:
            ff_prev = _fetch_forexfactory_for_date(prev_dt)
            if ff_prev:
                hi = sum(1 for e in ff_prev if e.get('impact','').lower()=='high')
                st.write(f"High:{hi}, total events:{len(ff_prev)}")
                for e in ff_prev:
                    st.write(f"- {e.get('time')} {e.get('currency')} {e.get('impact')} {e.get('event')}")
                st.write("Summary: Previous-day events may have influenced price action; incorporate into context for today's open.")
            else:
                st.write("No ForexFactory events found for previous trading day.")
        except Exception:
            st.write("Failed to fetch ForexFactory events for previous trading day.")
    else:
        st.write("No previous trading day available for ForexFactory lookup.")