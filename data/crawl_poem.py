#!/usr/bin/env python3
"""Crawl poem metadata from thivien.net (PoemType=16 - 5-word poems)."""

import logging
import os
import random
import re
import sys
import time
import urllib.parse

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, WebDriverException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("crawl_poem.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

SORT_CONFIGS = [
    {"sort": "", "order": ""},
    {"sort": "Author", "order": "asc"},
    {"sort": "Author", "order": "desc"},
    {"sort": "Views", "order": "desc"},
    {"sort": "Views", "order": "asc"},
    {"sort": "Date", "order": "asc"},
    {"sort": "Date", "order": "desc"},
    {"sort": "Poster", "order": "asc"},
    {"sort": "Poster", "order": "desc"},
]

BASE_SEARCH_URL = "https://www.thivien.net/search-poem.php?PoemType=16&Country=2"
MAX_PAGES_PER_CONFIG = 10
RETRY_MAX = 3
RETRY_BASE_DELAY = 5.0
SLEEP_MIN = 2.0
SLEEP_MAX = 4.0
OUTPUT_DIR = "raws"


def init_driver() -> webdriver.Chrome:
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("window-size=1920x1080")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options,
    )
    driver.implicitly_wait(10)
    return driver


def build_search_url(sort: str, order: str, page: int) -> str:
    url = BASE_SEARCH_URL
    if sort:
        url += f"&Sort={sort}"
    if order:
        url += f"&SortOrder={order}"
    url += f"&Page={page}"
    return url


def build_author_search_url(author_name: str, page: int) -> str:
    encoded = urllib.parse.quote(author_name)
    return f"https://www.thivien.net/search-poem.php?PoemType=16&Author={encoded}&Page={page}"


def random_sleep():
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))


def safe_get(driver: webdriver.Chrome, url: str) -> bool:
    """Navigate to URL with retry + exponential backoff."""
    for attempt in range(1, RETRY_MAX + 1):
        try:
            driver.get(url)
            random_sleep()
            return True
        except WebDriverException as e:
            wait_time = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            log.warning(f"[Retry {attempt}/{RETRY_MAX}] {url} -> {e}. Wait {wait_time:.0f}s...")
            time.sleep(wait_time)
    log.error(f"[FAIL] Could not access after {RETRY_MAX} attempts: {url}")
    return False


def is_blocked(driver: webdriver.Chrome) -> bool:
    block_signals = ["Danh sách quá dài", "Access denied", "Too many requests", "403 Forbidden"]
    page_src = driver.page_source
    return any(signal in page_src for signal in block_signals)


def get_total_pages(driver: webdriver.Chrome, context: str = "") -> int:
    """Parse total pages from current page."""
    match = re.search(r"tổng số (\d+) trang", driver.page_source)
    if match:
        return int(match.group(1))

    match = re.search(r"Trang\s+\d+\s*/\s*(\d+)", driver.page_source)
    if match:
        return int(match.group(1))

    try:
        page_links = driver.find_elements(By.XPATH, '//a[contains(@href,"Page=")]')
        page_nums = []
        for a in page_links:
            m = re.search(r"Page=(\d+)", a.get_attribute("href") or "")
            if m:
                page_nums.append(int(m.group(1)))
        if page_nums:
            return max(page_nums)
    except Exception:
        pass

    return 1


def extract_author_for_poem_links(driver: webdriver.Chrome) -> list:
    """Extract poem metadata from list page."""
    results = []
    try:
        items = driver.find_elements(By.XPATH, '//div[@class="list-item"]')
        for item in items:
            try:
                title_el = item.find_element(By.XPATH, './/h4[@class="list-item-header"]/a')
                title = title_el.text.strip()
                url = title_el.get_attribute("href") or ""
                if not url:
                    continue

                try:
                    author_el = item.find_element(By.XPATH, './/a[contains(@href,"/author-")]')
                    author = author_el.text.strip() or "Unknown"
                except NoSuchElementException:
                    author = "Unknown"

                results.append({"title": title, "url": url, "author": author})
            except Exception:
                continue
    except Exception as e:
        log.warning(f"[WARN] extract_author_for_poem_links: {e}")
    return results


def collect_metadata_by_authors(driver: webdriver.Chrome) -> list:
    """Collect poem metadata using sort configs and author pages."""
    all_authors: set = set()
    seen_urls: set = set()
    metadata_list: list = []

    # Step 1: Crawl by sort configs
    log.info("=" * 70)
    log.info("STEP 1: CRAWL BY SORT CONFIGS")
    log.info("=" * 70)

    for cfg in SORT_CONFIGS:
        sort = cfg["sort"]
        order = cfg["order"]
        label = f"Sort={sort or 'default'}/Order={order or 'default'}"
        log.info(f"\n[{label}]")

        first_url = build_search_url(sort, order, 1)
        total_pages = 1
        if safe_get(driver, first_url) and not is_blocked(driver):
            total_pages = get_total_pages(driver, context=label)

        pages_to_crawl = min(total_pages, MAX_PAGES_PER_CONFIG)
        log.info(f"  Total pages: {total_pages} | Will crawl: {pages_to_crawl}")

        for page in range(1, pages_to_crawl + 1):
            if page > 1:
                url = build_search_url(sort, order, page)
                if not safe_get(driver, url):
                    continue
                if is_blocked(driver):
                    log.warning(f"  Page {page}: blocked")
                    break

            page_items = extract_author_for_poem_links(driver)
            page_authors = {item["author"] for item in page_items if item["author"] != "Unknown"}

            if not page_items:
                continue

            all_authors.update(page_authors)

            new_count = 0
            for item in page_items:
                if item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    metadata_list.append(item)
                    new_count += 1

            log.info(f"  Page {page}/{pages_to_crawl}: +{new_count} new | Total: {len(metadata_list)}")

    log.info(f"\nAfter Step 1: {len(metadata_list)} poems, {len(all_authors)} authors")

    # Step 2: Crawl by author
    log.info("\n" + "=" * 70)
    log.info("STEP 2: CRAWL BY AUTHOR")
    log.info("=" * 70)

    for author_name in tqdm(sorted(all_authors), desc="Authors"):
        url_p1 = build_author_search_url(author_name, 1)

        if not safe_get(driver, url_p1):
            continue
        if is_blocked(driver):
            continue

        total_pages = get_total_pages(driver, context=author_name)
        pages_to_crawl = min(total_pages, MAX_PAGES_PER_CONFIG)

        for page in range(1, pages_to_crawl + 1):
            if page > 1:
                url = build_author_search_url(author_name, page)
                if not safe_get(driver, url):
                    break
                if is_blocked(driver):
                    break

            page_items = extract_author_for_poem_links(driver)
            if not page_items:
                break

            for item in page_items:
                if item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    item["author"] = author_name
                    metadata_list.append(item)

    log.info(f"\nTotal metadata: {len(metadata_list)} poems")
    return metadata_list


def run_phase_1(driver: webdriver.Chrome, output_file: str = None):
    """Collect metadata and save to CSV."""
    if output_file is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, "poem_metadata.csv")

    metadata_list = collect_metadata_by_authors(driver)
    df = pd.DataFrame(metadata_list)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    log.info(f"\nDone! Saved {len(metadata_list)} poems to {output_file}")


if __name__ == "__main__":
    driver = init_driver()
    try:
        output_file = sys.argv[1] if len(sys.argv) > 1 else None
        run_phase_1(driver, output_file)
    except KeyboardInterrupt:
        log.info("\n[INTERRUPTED]")
    finally:
        driver.quit()
