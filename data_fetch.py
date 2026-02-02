import os
import io
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import yfinance as yf
from datetime import date, datetime
import re
from typing import List, Dict
from pathlib import Path


# Target instrument: Nasdaq-100 futures
YF_TICKER = "NQ=F"  # yfinance futures symbol
POLY_TICKER = "NQ"  # polygon ticker base (no suffix)
NASDAQ_COM_SYMBOL = "NDX"  # nasdaq.com symbol for Nasdaq-100 index


def _nasdaqtrader_has_symbol(sym: str) -> bool:
    """Check nasdaqtrader symbol directory for presence of a symbol (best-effort)."""
    try:
        url = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        text = r.text
        # file is pipe-delimited with header; search for symbol in the text
        return sym in text
    except Exception:
        return False


def _http_session_with_retries(retries: int = 3, backoff: float = 0.3, status_forcelist=(429, 500, 502, 503, 504)) -> requests.Session:
    """Create a requests.Session with Retry configured to use polite headers."""
    s = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff, status_forcelist=status_forcelist, allowed_methods=frozenset(['GET','POST']))
    adapter = HTTPAdapter(max_retries=retry)
    s.mount('https://', adapter)
    s.mount('http://', adapter)
    s.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com/',
    })
    return s


def _period_to_start(period: str):
    try:
        if period.endswith("mo"):
            months = int(period[:-2])
            return (pd.Timestamp.now() - pd.DateOffset(months=months)).date()
        if period.endswith("y"):
            years = int(period[:-1])
            return (pd.Timestamp.now() - pd.DateOffset(years=years)).date()
    except Exception:
        pass
    # default ~180 days
    return (pd.Timestamp.now() - pd.Timedelta(days=180)).date()


def _try_polygon_daily(ticker: str, period: str = "6mo") -> pd.DataFrame:
    key = os.getenv("POLYGON_API_KEY")
    if not key:
        raise RuntimeError("No POLYGON_API_KEY in environment")

    start = _period_to_start(period).strftime("%Y-%m-%d")
    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    poly_ticker = ticker.lstrip("^")
    url = f"https://api.polygon.io/v2/aggs/ticker/{poly_ticker}/range/1/day/{start}/{end}?adjusted=true&sort=asc&limit=50000&apiKey={key}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    j = r.json()
    results = j.get("results", [])
    if not results:
        raise RuntimeError("Polygon returned no results")
    rows = []
    for r in results:
        ts = pd.to_datetime(r["t"], unit="ms")
        rows.append((ts.date(), r["o"], r["h"], r["l"], r["c"], r["v"]))
    df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df.set_index("Date", inplace=True)
    return df


def _try_nasdaq_com_daily(ticker: str, period: str = "6mo") -> pd.DataFrame:
    # Nasdaq.com historical JSON endpoint (public) — may be rate-limited; use polite headers
    start = _period_to_start(period).strftime("%m/%d/%Y")
    end = pd.Timestamp.now().strftime("%m/%d/%Y")
    # prefer explicit Nasdaq symbol if present
    sym = ticker.lstrip("^")
    if not _nasdaqtrader_has_symbol(sym):
        # fall back to NASDAQ_COM_SYMBOL mapping for Nasdaq-100
        sym = NASDAQ_COM_SYMBOL
    url = f"https://api.nasdaq.com/api/quote/{sym}/historical?assetclass=index&fromdate={start}&todate={end}"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    j = r.json()
    data = j.get("data", {}).get("historical", [])
    if not data:
        raise RuntimeError("nasdaq.com returned no historical data")
    rows = []
    for item in data:
        # item fields usually: date,close,open,high,low,volume
        try:
            d = pd.to_datetime(item.get("date")).date()
            o = float(item.get("open", "nan").replace(",", ""))
            h = float(item.get("high", "nan").replace(",", ""))
            l = float(item.get("low", "nan").replace(",", ""))
            c = float(item.get("close", "nan").replace(",", ""))
            v = int(item.get("volume", "0").replace(",", ""))
            rows.append((d, o, h, l, c, v))
        except Exception:
            continue
    df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"]) 
    df.set_index("Date", inplace=True)
    return df


def get_daily_data(period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Preferred sources: nasdaq.com (with nasdaqtrader symbol validation), then Polygon (if key), then yfinance fallback."""
    # Primary: nasdaq.com (uses nasdaqtrader to prefer correct symbol mapping)
    try:
        return _try_nasdaq_com_daily(POLY_TICKER, period=period)
    except Exception:
        pass

    # Secondary: Polygon (if key present)
    try:
        return _try_polygon_daily(POLY_TICKER, period=period)
    except Exception:
        pass

    # Fallback to yfinance
    df = yf.download(YF_TICKER, period=period, interval=interval, auto_adjust=False)
    df.dropna(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).date
    return df


def get_intraday_data(period: str = "10d", interval: str = "5m") -> pd.DataFrame:
    """Try polygon intraday (if key), otherwise yfinance intraday."""
    key = os.getenv("POLYGON_API_KEY")
    if key:
        try:
            # polygon intraday aggregate for last `period` days at `interval` minute bars
            poly_ticker = POLY_TICKER.lstrip("^")
            # compute from/to
            start = (pd.Timestamp.now() - pd.Timedelta(days= int(period.rstrip('d')) if period.endswith('d') else 10)).strftime('%Y-%m-%d')
            end = pd.Timestamp.now().strftime('%Y-%m-%d')
            # interval in minutes
            mins = int(interval.rstrip('m')) if interval.endswith('m') else 5
            url = f"https://api.polygon.io/v2/aggs/ticker/{poly_ticker}/range/{mins}/minute/{start}/{end}?adjusted=true&sort=asc&limit=50000&apiKey={key}"
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            j = r.json()
            results = j.get('results', [])
            rows = []
            for it in results:
                ts = pd.to_datetime(it['t'], unit='ms').tz_localize(None)
                rows.append((ts, it['o'], it['h'], it['l'], it['c'], it['v']))
            idf = pd.DataFrame(rows, columns=['Datetime','Open','High','Low','Close','Volume'])
            idf.set_index('Datetime', inplace=True)
            idf['Date'] = pd.to_datetime(idf.index).date
            return idf
        except Exception:
            pass

    # Fallback to yfinance intraday
    df = yf.download(YF_TICKER, period=period, interval=interval, auto_adjust=False)
    df.dropna(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['Date'] = pd.to_datetime(df.index).date
    # Try appending a quick Seeking Alpha tick for the Nasdaq futures symbol if available
    try:
        sa_tick = _fetch_seekingalpha_tick(YF_TICKER)
        if sa_tick is not None and not sa_tick.empty:
            # ensure index types align; if df has tz-naive index, keep sa_tick naive
            try:
                # convert sa_tick index to match df.index tz if possible
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    sa_tick.index = pd.to_datetime(sa_tick.index).tz_localize('UTC').tz_convert(df.index.tz)
            except Exception:
                pass
            # append only if newer than existing last timestamp
            try:
                if df.empty or sa_tick.index.max() > df.index.max():
                    df = pd.concat([df, sa_tick])
                    # persist the tick for later session aggregation
                    try:
                        _persist_seekingalpha_tick(sa_tick)
                    except Exception:
                        pass
            except Exception:
                df = pd.concat([df, sa_tick])
                try:
                    _persist_seekingalpha_tick(sa_tick)
                except Exception:
                    pass
    except Exception:
        pass
    # also append any previously persisted Seeking Alpha ticks collected by a poller
    try:
        persisted = _load_seekingalpha_ticks()
        if persisted is not None and not persisted.empty:
            # convert persisted ticks to match df index timezone (if present)
            try:
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    persisted.index = persisted.index.tz_convert(df.index.tz)
                else:
                    # make persisted tz-naive to match df
                    persisted.index = persisted.index.tz_localize(None)
            except Exception:
                pass
            df = pd.concat([df, persisted])
    except Exception:
        pass

    # Try nasdaq.com quick tick and persist as external tick store
    try:
        nas_tick = _fetch_nasdaq_com_tick(POLY_TICKER)
        if nas_tick is not None and not nas_tick.empty:
            try:
                # align tz like we did for SA ticks
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    nas_tick.index = nas_tick.index.tz_localize('UTC').tz_convert(df.index.tz)
                else:
                    nas_tick.index = nas_tick.index.tz_localize(None)
            except Exception:
                pass
            # append and persist to external store
            try:
                df = pd.concat([df, nas_tick])
                _persist_external_tick(nas_tick, source='nasdaq.com')
            except Exception:
                pass
    except Exception:
        pass

    # Load any previously persisted external ticks (seekingalpha + nasdaq combined store)
    try:
        ext = _load_external_ticks()
        if ext is not None and not ext.empty:
            try:
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    ext.index = ext.index.tz_convert(df.index.tz)
                else:
                    ext.index = ext.index.tz_localize(None)
            except Exception:
                pass
            df = pd.concat([df, ext])
    except Exception:
        pass

    # normalize and deduplicate by index
    try:
        df = df[~df.index.duplicated(keep='last')].sort_index()
    except Exception:
        pass

    return df


def _fetch_seekingalpha_tick(symbol: str) -> pd.DataFrame | None:
    """Attempt to fetch a lightweight latest-quote tick from Seeking Alpha for `symbol`.

    Returns a single-row DataFrame with index=timestamp and columns Open/High/Low/Close/Volume.
    This is a best-effort scraper and may break if Seeking Alpha changes page layout.
    """
    try:
        sa_sym = symbol.split('=')[0]
        session = _http_session_with_retries()
        url = f"https://seekingalpha.com/symbol/{sa_sym}"
        r = session.get(url, timeout=8)
        r.raise_for_status()
        text = r.text

        # Try a handful of patterns: JSON-like, data-react-props, meta tags, and numeric spans
        price = None
        # JSON-like keys
        for patt in (r'"last"\s*[:=]\s*([0-9]{1,6}\.[0-9]{1,4})', r'"currentPrice"\s*[:=]\s*([0-9]{1,6}\.[0-9]{1,4})'):
            m = re.search(patt, text)
            if m:
                price = float(m.group(1).replace(',', ''))
                break

        # try data-react-props or other embedded JSON
        if price is None:
            m = re.search(r'data-react-props="([^"]+)"', text)
            if m:
                js = m.group(1)
                nm = re.search(r'([0-9]{1,7}\.[0-9]{1,4})', js)
                if nm:
                    price = float(nm.group(1))

        # fallback: look for price inside meta description or og:description
        if price is None:
            m = re.search(r'property="og:description" content="[^"]*?([0-9]{1,6}\.[0-9]{1,4})', text)
            if m:
                price = float(m.group(1))

        # final fallback: find first numeric span with class that looks like price
        if price is None:
            m = re.search(r'<span[^>]*class="[^"]*(price|last|quote)[^"]*"[^>]*>([0-9]{1,7}\.[0-9]{1,4})', text, re.I)
            if m:
                price = float(m.group(2))

        # also try JSON keys that sometimes appear as "price" or "lastPrice"
        if price is None:
            m = re.search(r'"lastPrice"\s*[:=]\s*"?([0-9]{1,7}\.[0-9]{1,4})"?', text)
            if m:
                price = float(m.group(1))
        if price is None:
            m = re.search(r'"price"\s*[:=]\s*"?([0-9]{1,7}\.[0-9]{1,4})"?', text)
            if m:
                price = float(m.group(1))

        if price is None:
            return None

        ts = datetime.utcnow()
        row = {'Open': price, 'High': price, 'Low': price, 'Close': price, 'Volume': 0, 'Source': 'SeekingAlpha'}
        idf = pd.DataFrame([row], index=[pd.Timestamp(ts, tz='UTC')])
        idf.index.name = 'Datetime'
        idf['Date'] = pd.to_datetime(idf.index).date
        return idf
    except Exception:
        return None


def fetch_seekingalpha_news(symbol: str, max_items: int = 10) -> List[Dict]:
    """Fetch recent news/articles for a Seeking Alpha symbol (best-effort scraper).

    Returns a list of dicts: {"title": str, "url": str, "snippet": str, "time": str}
    """
    try:
        sa_sym = symbol.split('=')[0]
        urls = [f"https://seekingalpha.com/symbol/{sa_sym}/news", f"https://seekingalpha.com/symbol/{sa_sym}"]
        session = _http_session_with_retries()
        html = None
        for url in urls:
            try:
                r = session.get(url, timeout=8)
                r.raise_for_status()
                html = r.text
                break
            except Exception:
                # try Selenium fallback for Seeking Alpha if simple request fails
                try:
                    from selenium import webdriver  # type: ignore
                    from selenium.webdriver.chrome.options import Options  # type: ignore
                    from selenium.webdriver.chrome.service import Service  # type: ignore
                    from webdriver_manager.chrome import ChromeDriverManager  # type: ignore

                    opts = Options()
                    opts.add_argument('--headless=new')
                    opts.add_argument('--no-sandbox')
                    opts.add_argument('--disable-dev-shm-usage')
                    service = Service(ChromeDriverManager().install())
                    driver = webdriver.Chrome(service=service, options=opts)
                    try:
                        driver.get(url)
                        html = driver.page_source
                        break
                    finally:
                        driver.quit()
                except Exception:
                    continue
        if not html:
            return []

        articles = []
        # match article and news links; include /article/ and /news/ patterns
        for m in re.finditer(r'<a[^>]+href=["\'](?P<h>(/article/|/news/)[^"\']+)["\'][^>]*>(?P<t>[^<]{10,300})</a>', html, re.I):
            href = m.group('h')
            title = re.sub('<[^<]+?>', '', m.group('t')).strip()
            if href.startswith('/'):
                url = f"https://seekingalpha.com{href}"
            else:
                url = href
            start = m.end()
            snippet = ''
            snip_m = re.search(r'<p[^>]*>([^<]{20,300})</p>', html[start:start+600], re.I)
            if snip_m:
                snippet = re.sub('<[^<]+?>', '', snip_m.group(1)).strip()
            articles.append({"title": title, "url": url, "snippet": snippet, "time": None})
            if len(articles) >= max_items:
                break

        # if none found via anchors, try JSON-LD or scripts with article lists
        if not articles:
            for m in re.finditer(r'"url"\s*:\s*"(?P<u>/article/[^"]+)"\s*,\s*"headline"\s*:\s*"(?P<h>[^\"]+)"', html, re.I):
                href = m.group('u')
                title = m.group('h')
                url = f"https://seekingalpha.com{href}"
                articles.append({"title": title, "url": url, "snippet": '', "time": None})
                if len(articles) >= max_items:
                    break

        return articles
    except Exception:
        return []


def _persist_seekingalpha_tick(idf: pd.DataFrame, data_dir: str = "data") -> None:
    """Append a seeking-alpha tick DataFrame to a CSV store (idempotent by timestamp)."""
    try:
        p = Path(data_dir)
        p.mkdir(parents=True, exist_ok=True)
        out = p / "seekingalpha_ticks.csv"
        # prepare frame for CSV
        tmp = idf.copy()
        tmp = tmp.reset_index()
        tmp['Datetime'] = tmp['Datetime'].astype(str)
        if out.exists():
            existing = pd.read_csv(out)
            # avoid duplicate Datetime
            combined = pd.concat([existing, tmp], ignore_index=True)
            combined = combined.drop_duplicates(subset=['Datetime'], keep='last')
            combined.to_csv(out, index=False)
        else:
            tmp.to_csv(out, index=False)
    except Exception:
        pass


def _load_seekingalpha_ticks(data_dir: str = "data") -> pd.DataFrame | None:
    try:
        p = Path(data_dir) / "seekingalpha_ticks.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p)
        if 'Datetime' in df.columns:
            # parse as UTC timestamps and set as index
            df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
            df.set_index('Datetime', inplace=True)
            # ensure Date column exists
            if 'Date' not in df.columns:
                df['Date'] = pd.to_datetime(df.index).date
            return df
        return None
    except Exception:
        return None


def _persist_external_tick(idf: pd.DataFrame, source: str = 'external', data_dir: str = 'data') -> None:
    """Append a generic external tick DataFrame to a CSV store (idempotent by timestamp).

    Stores rows with columns: Datetime, Source, Open, High, Low, Close, Volume
    """
    try:
        p = Path(data_dir)
        p.mkdir(parents=True, exist_ok=True)
        out = p / "external_ticks.csv"
        tmp = idf.copy().reset_index()
        if 'Source' not in tmp.columns:
            tmp['Source'] = source
        tmp['Datetime'] = tmp['Datetime'].astype(str)
        cols = ['Datetime', 'Source', 'Open', 'High', 'Low', 'Close', 'Volume']
        tmp = tmp[[c for c in cols if c in tmp.columns]]
        if out.exists():
            existing = pd.read_csv(out)
            combined = pd.concat([existing, tmp], ignore_index=True)
            combined = combined.drop_duplicates(subset=['Datetime','Source'], keep='last')
            combined.to_csv(out, index=False)
        else:
            tmp.to_csv(out, index=False)
    except Exception:
        pass


def _load_external_ticks(data_dir: str = 'data') -> pd.DataFrame | None:
    try:
        p = Path(data_dir) / 'external_ticks.csv'
        if not p.exists():
            return None
        df = pd.read_csv(p)
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
            if 'Source' not in df.columns:
                df['Source'] = 'external'
            df.set_index('Datetime', inplace=True)
            if 'Date' not in df.columns:
                df['Date'] = pd.to_datetime(df.index).date
            return df
        return None
    except Exception:
        return None


def _fetch_nasdaq_com_tick(symbol: str) -> pd.DataFrame | None:
    """Best-effort quick quote from nasdaq.com for a symbol.

    Returns a single-row DataFrame indexed by UTC timestamp with OHLCV and Source.
    """
    try:
        # prefer mapped symbol (POLY_TICKER may be used for index/futures mapping)
        sym = symbol.split('=')[0]
        # Try Nasdaq API JSON endpoint first (if available)
        session = _http_session_with_retries()
        api_urls = [
            f'https://api.nasdaq.com/api/quote/{sym}/info?assetclass=indices',
            f'https://api.nasdaq.com/api/quote/{sym}/summary',
        ]
        price = None
        for url in api_urls:
            try:
                r = session.get(url, timeout=8)
                r.raise_for_status()
                j = r.json()
                # navigate common fields
                pdata = j.get('data', {})
                # try known nested keys
                for k in ('primaryData','lastSalePrice','last'):
                    if isinstance(pdata, dict) and k in pdata:
                        val = pdata.get(k)
                        if isinstance(val, str) and re.search(r'[0-9]+\.', val):
                            price = float(re.sub(r'[^0-9\.]','', val))
                            break
                # some endpoints return lastSalePrice inside primaryData
                if not price and isinstance(pdata.get('primaryData', {}), dict):
                    cand = pdata['primaryData'].get('lastSalePrice') or pdata['primaryData'].get('last')
                    if cand:
                        m = re.search(r'([0-9]{1,7}\.[0-9]{1,4})', str(cand))
                        if m:
                            price = float(m.group(1))
                if price is not None:
                    break
            except Exception:
                continue

        # Fallback: scrape the public quote page and look for JSON-LD or price meta
        if price is None:
            page_urls = [
                f'https://www.nasdaq.com/market-activity/index/{sym.lower()}',
                f'https://www.nasdaq.com/market-activity/quotes/{sym.lower()}',
                f'https://www.nasdaq.com/market-activity/{sym.lower()}',
            ]
            for url in page_urls:
                try:
                    r = session.get(url, timeout=8)
                    r.raise_for_status()
                    html = r.text
                    m = re.search(r'"lastSalePrice"\s*[:=]\s*"?([0-9]{1,7}\.[0-9]{1,4})"?', html)
                    if not m:
                        m = re.search(r'property="og:description" content="[^"]*?([0-9]{1,7}\.[0-9]{1,4})', html)
                    if not m:
                        # try more generic price keys
                        m = re.search(r'"lastPrice"\s*[:=]\s*"?([0-9]{1,7}\.[0-9]{1,4})"?', html)
                    if not m:
                        m = re.search(r'"price"\s*[:=]\s*"?([0-9]{1,7}\.[0-9]{1,4})"?', html)
                    if m:
                        price = float(m.group(1))
                        break
                except Exception:
                    continue

        if price is None:
            return None

        ts = datetime.utcnow()
        row = {'Open': price, 'High': price, 'Low': price, 'Close': price, 'Volume': 0, 'Source': 'nasdaq.com'}
        idf = pd.DataFrame([row], index=[pd.Timestamp(ts, tz='UTC')])
        idf.index.name = 'Datetime'
        idf['Date'] = pd.to_datetime(idf.index).date
        return idf
    except Exception:
        return None


def compute_daily_vwap(intra: pd.DataFrame) -> pd.DataFrame:
    tp = (intra["High"] + intra["Low"] + intra["Close"]) / 3.0
    intra = intra.copy()
    intra["TP"] = tp
    intra["TPxVol"] = intra["TP"] * intra["Volume"]
    daily = intra.groupby("Date").agg(
        TPxVol_sum=("TPxVol", "sum"),
        Vol_sum=("Volume", "sum"),
    )
    daily["VWAP_Daily"] = daily["TPxVol_sum"] / daily["Vol_sum"]
    return daily[["VWAP_Daily"]]


def compute_weekly_vwap(intra: pd.DataFrame) -> pd.DataFrame:
    intra = intra.copy()
    tp = (intra["High"] + intra["Low"] + intra["Close"]) / 3.0
    intra["TP"] = tp
    intra["TPxVol"] = intra["TP"] * intra["Volume"]
    # note: converting to Period may drop tz info — this is OK for weekly grouping
    intra["YearWeek"] = pd.to_datetime(intra.index).to_period("W").astype(str)

    weekly = intra.groupby("YearWeek").agg(
        TPxVol_sum=("TPxVol", "sum"),
        Vol_sum=("Volume", "sum"),
    )
    weekly["VWAP_Weekly"] = weekly["TPxVol_sum"] / weekly["Vol_sum"]
    weekly = weekly[["VWAP_Weekly"]]

    intra_week = intra.groupby("Date")["YearWeek"].first()
    weekly_map = intra_week.to_frame().join(weekly, on="YearWeek")
    weekly_map = weekly_map[["VWAP_Weekly"]]
    return weekly_map


def build_base_dataframe() -> pd.DataFrame:
    daily = get_daily_data()
    intra = get_intraday_data()
    daily_vwap = compute_daily_vwap(intra)
    weekly_vwap = compute_weekly_vwap(intra)
    df = daily.join(daily_vwap, how="left")
    df = df.join(weekly_vwap, how="left")
    # augment with ForexFactory economic calendar-derived features where available
    try:
        ff_feats = compute_forexfactory_features(df.index)
        if not ff_feats.empty:
            ff_feats.index = pd.to_datetime(ff_feats.index).date
            # join on date index
            df = df.join(ff_feats, how="left")
    except Exception:
        # non-fatal: if forexfactory parsing fails, continue without features
        pass
    return df


def _fetch_forexfactory_for_date(d: date) -> List[Dict]:
    """Fetch ForexFactory calendar entries for a single date.

    Returns list of event dicts with keys: time, currency, impact, event, actual, forecast, previous
    """
    url = f"https://www.forexfactory.com/calendar.php?day={d.strftime('%Y-%m-%d')}"
    session = _http_session_with_retries()
    html = None
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        # if requests is blocked (403), try a Selenium fallback to retrieve the page
        try:
            from selenium import webdriver  # type: ignore
            from selenium.webdriver.chrome.options import Options  # type: ignore
            from selenium.webdriver.chrome.service import Service  # type: ignore
            from webdriver_manager.chrome import ChromeDriverManager  # type: ignore

            opts = Options()
            opts.add_argument('--headless=new')
            opts.add_argument('--no-sandbox')
            opts.add_argument('--disable-dev-shm-usage')
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=opts)
            try:
                driver.get(url)
                html = driver.page_source
            finally:
                driver.quit()
        except Exception:
            # re-raise original exception to let caller handle
            raise

    events = []
    # best-effort parsing: find table rows containing calendar__event
    # rows often contain data-event attributes or classes; fall back to regex for common columns
    # pattern to capture time, currency, impact (class), event name, actual/forecast/previous
    row_re = re.compile(r'<tr[^>]*class="calendar__row[^"]*"[\s\S]*?>([\s\S]*?)</tr>', re.I)
    cells_re = re.compile(r'<td[^>]*>([\s\S]*?)</td>', re.I)
    impact_re = re.compile(r'class="impact impact-(low|medium|high)"', re.I)
    # iterate matches
    for m in row_re.finditer(html):
        row_html = m.group(1)
        cells = cells_re.findall(row_html)
        if not cells:
            continue
        try:
            # common layout: time, currency, impact+event, actual, forecast, previous
            tcell = re.sub('<[^<]+?>', '', cells[0]).strip()
            ccell = re.sub('<[^<]+?>', '', cells[1]).strip() if len(cells) > 1 else ""
            impact_match = impact_re.search(row_html)
            impact = impact_match.group(1).capitalize() if impact_match else ""
            # event name often in a link/text in cell 2 or 3; try join
            evt = re.sub('<[^<]+?>', '', cells[2]).strip() if len(cells) > 2 else ""
            actual = re.sub('<[^<]+?>', '', cells[3]).strip() if len(cells) > 3 else ""
            forecast = re.sub('<[^<]+?>', '', cells[4]).strip() if len(cells) > 4 else ""
            previous = re.sub('<[^<]+?>', '', cells[5]).strip() if len(cells) > 5 else ""
            events.append({"time": tcell, "currency": ccell, "impact": impact, "event": evt, "actual": actual, "forecast": forecast, "previous": previous})
        except Exception:
            continue
    return events


def compute_forexfactory_features(dates_index) -> pd.DataFrame:
    """Aggregate ForexFactory calendar features for a set of dates (index of daily df).

    Returns DataFrame indexed by date with columns:
      - FF_HighImpactCount
      - FF_MidImpactCount
      - FF_EventScore (weighted: high=3, medium=2, low=1)
    """
    results = []
    uniq_dates = sorted({pd.to_datetime(d).date() for d in dates_index})
    for d in uniq_dates:
        try:
            evts = _fetch_forexfactory_for_date(d)
        except Exception:
            # on failure, append zeros
            results.append((d, 0, 0, 0))
            continue
        high = sum(1 for e in evts if e.get("impact", "").lower() == "high")
        mid = sum(1 for e in evts if e.get("impact", "").lower() == "medium")
        low = sum(1 for e in evts if e.get("impact", "").lower() == "low")
        score = high * 3 + mid * 2 + low * 1
        results.append((d, high, mid, score))

    ff_df = pd.DataFrame(results, columns=["Date", "FF_HighImpactCount", "FF_MidImpactCount", "FF_EventScore"]) 
    ff_df.set_index("Date", inplace=True)
    return ff_df