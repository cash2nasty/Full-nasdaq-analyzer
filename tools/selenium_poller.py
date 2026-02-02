"""
Selenium-based headless poller for sites that block simple HTTP requests.
Requirements:
  pip install selenium webdriver-manager

This script fetches SeekingAlpha and Nasdaq pages using a headless Chrome webdriver
and extracts a best-effort price. It persists ticks via the same `_persist_external_tick`.

Note: the runtime environment must have Chrome/Chromium installed. On first run,
`webdriver_manager` will download a matching driver.
"""
import sys
import pathlib
import argparse
import time

# ensure repo import
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from data_fetch import _persist_external_tick

from selenium import webdriver  # type: ignore[import]
from selenium.webdriver.chrome.options import Options  # type: ignore[import]
from selenium.webdriver.chrome.service import Service  # type: ignore[import]
from webdriver_manager.chrome import ChromeDriverManager  # type: ignore[import]
from selenium.webdriver.common.by import By  # type: ignore[import]


def get_driver(headless=True):
    opts = Options()
    if headless:
        opts.add_argument('--headless=new')
    opts.add_argument('--no-sandbox')
    opts.add_argument('--disable-dev-shm-usage')
    opts.add_argument('--disable-gpu')
    opts.add_argument('--window-size=1200,800')
    # Use Service(...) and pass options explicitly to avoid multiple-values error
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    return driver


def extract_price_from_page(driver, url):
    try:
        driver.get(url)
        time.sleep(1.0)
        # try common selectors
        candidates = [
            (By.CSS_SELECTOR, 'meta[property="og:description"]'),
            (By.CSS_SELECTOR, 'span.price'),
            (By.CSS_SELECTOR, 'div.last-sale'),
            (By.CSS_SELECTOR, 'div.quote'),
        ]
        text = driver.page_source
        # fallback: try to find numeric in page
        import re
        m = re.search(r'([0-9]{1,7}\.[0-9]{1,4})', text)
        if m:
            return float(m.group(1))
    except Exception:
        return None
    return None


def poll_once():
    driver = None
    try:
        driver = get_driver(headless=True)
        # SeekingAlpha symbol page
        sa_url = 'https://seekingalpha.com/symbol/NQ'
        price = extract_price_from_page(driver, sa_url)
        if price is not None:
            import pandas as pd
            from datetime import datetime
            df = pd.DataFrame([{'Open':price,'High':price,'Low':price,'Close':price,'Volume':0}], index=[pd.Timestamp(datetime.utcnow(), tz='UTC')])
            df.index.name = 'Datetime'
            _persist_external_tick(df, source='seekingalpha.selenium')
            print('Persisted SA via selenium', price)

        # Nasdaq page
        nas_url = 'https://www.nasdaq.com/market-activity/index/ndx'
        price = extract_price_from_page(driver, nas_url)
        if price is not None:
            import pandas as pd
            from datetime import datetime
            df = pd.DataFrame([{'Open':price,'High':price,'Low':price,'Close':price,'Volume':0}], index=[pd.Timestamp(datetime.utcnow(), tz='UTC')])
            df.index.name = 'Datetime'
            _persist_external_tick(df, source='nasdaq.selenium')
            print('Persisted NASDAQ via selenium', price)

    finally:
        if driver:
            driver.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.once:
        poll_once()
    else:
        try:
            while True:
                poll_once()
                time.sleep(60)
        except KeyboardInterrupt:
            print('Stopping selenium poller')
