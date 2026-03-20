#!/usr/bin/env python3
"""Scrape poem content from thivien.net using curl_cffi + BeautifulSoup."""

import argparse
import logging
import os
import random
import re
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scrape_poem_content.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

RETRY_MAX = 3
RETRY_BASE_DELAY = 5.0
SLEEP_MIN = 5.0
SLEEP_MAX = 10.0
CHECKPOINT_EVERY = 10

COOKIE_TEST_URL = "https://www.thivien.net/Xu%C3%A2n-Qu%E1%BB%B3nh/S%C3%B3ng/poem-fsd-MCqhqwgCayHKWx-MPg"

THIVIEN_USERNAME = os.environ.get("THIVIEN_USERNAME", "")
THIVIEN_PASSWORD = os.environ.get("THIVIEN_PASSWORD", "")

SESSION = requests.Session()
SESSION.headers.update({
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
    "Referer": "https://www.thivien.net/",
})


def random_sleep():
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))


def is_blocked(html: str) -> bool:
    return "Xác nhận không phải máy" in html


def fetch_html(url: str) -> str | None:
    """Fetch HTML with retry + exponential backoff."""
    for attempt in range(1, RETRY_MAX + 1):
        try:
            resp = SESSION.get(url, timeout=15, impersonate="chrome120")
            resp.raise_for_status()

            if is_blocked(resp.text):
                log.error(f"[BLOCKED] {url}")
                return None

            return resp.text

        except Exception as e:
            wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            log.warning(f"[Retry {attempt}/{RETRY_MAX}] {url} -> {e}. Wait {wait:.0f}s...")
            time.sleep(wait)

    log.error(f"[FAIL] Could not fetch after {RETRY_MAX} attempts: {url}")
    return None


def login(username: str, password: str) -> bool:
    """Login to thivien.net."""
    log.info("Logging in to thivien.net...")
    try:
        resp = SESSION.post(
            "https://www.thivien.net/login.php",
            data={
                "Mode": "Login",
                "Login": username,
                "Password": password,
                "RememberMe": "on",
            },
            impersonate="chrome120",
            timeout=15,
        )

        if "_UserUID" in resp.text and "null" not in resp.text[resp.text.find("_UserUID"):resp.text.find("_UserUID")+30]:
            log.info("Login successful!")
            return True

        test_html = fetch_html(COOKIE_TEST_URL)
        if test_html and "poem-content" in test_html:
            log.info("Login successful!")
            return True

        log.error("Login failed! Check THIVIEN_USERNAME and THIVIEN_PASSWORD.")
        return False

    except Exception as e:
        log.error(f"Login error: {e}")
        return False


def extract_poem_raw(html: str, poem_src: str, poem_url: str, default_title: str = "") -> list:
    """Extract raw HTML from poem-content div."""
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    pattern = re.compile(r"<b>(.*?)</b>\s*\n{2,}", flags=re.IGNORECASE)
    matches = list(pattern.finditer(html))
    poems = []

    if matches:
        for i, m in enumerate(matches):
            title = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(html)
            content = html[start:end].strip("\n")
            poems.append({"title": title, "content": content, "source": poem_src, "url": poem_url})
    else:
        poems.append({"title": default_title, "content": html, "source": poem_src, "url": poem_url})

    return poems


def scrape_poem(url: str, default_title: str = "") -> list:
    html = fetch_html(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    content_div = soup.select_one("div.poem-content")
    if not content_div:
        log.warning(f"[WARN] No div.poem-content found: {url}")
        return []

    inner_html = str(content_div)
    poem_src = ""
    src_el = soup.select_one("div.small")
    if src_el:
        poem_src = src_el.get_text(strip=True)

    return extract_poem_raw(inner_html, poem_src, url, default_title)


def append_to_csv(data: list, filepath: str) -> None:
    df = pd.DataFrame(data)
    if os.path.exists(filepath):
        df.to_csv(filepath, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(filepath, index=False, encoding="utf-8-sig")


def run(metadata_file: str, output_file: str, resume: bool = True) -> None:
    if not login(THIVIEN_USERNAME, THIVIEN_PASSWORD):
        return

    if not os.path.exists(metadata_file):
        log.error(f"Metadata file not found: {metadata_file}")
        return

    df_meta = pd.read_csv(metadata_file, encoding="utf-8-sig")
    log.info(f"Loaded {len(df_meta)} poems from {metadata_file}")

    scraped_urls: set = set()
    if resume and os.path.exists(output_file):
        df_done = pd.read_csv(output_file, encoding="utf-8-sig")
        scraped_urls = set(df_done["url"].dropna().unique())
        log.info(f"Resume: skipping {len(scraped_urls)} already scraped")

    df_todo = df_meta[~df_meta["url"].isin(scraped_urls)].copy()
    log.info(f"Remaining: {len(df_todo)} poems")

    if df_todo.empty:
        log.info("All poems already scraped!")
        return

    batch: list = []
    blocked_count: int = 0

    for _, row in tqdm(df_todo.iterrows(), total=len(df_todo), desc="Scraping"):
        poem_url = row["url"]
        poem_title = row.get("title", "")
        poem_author = row.get("author", "Unknown")

        results = scrape_poem(poem_url, poem_title)

        if not results:
            blocked_count += 1
            if blocked_count >= 5:
                log.error("[STOP] 5 consecutive errors - possibly rate limited.")
                if batch:
                    append_to_csv(batch, output_file)
                break
        else:
            blocked_count = 0
            for item in results:
                item["author"] = poem_author
            batch.extend(results)

        random_sleep()

        if len(batch) >= CHECKPOINT_EVERY:
            append_to_csv(batch, output_file)
            log.info(f"[CHECKPOINT] Saved {len(batch)} poems")
            batch = []

    if batch:
        append_to_csv(batch, output_file)

    log.info(f"Done! Output: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="raws/poem_metadata.csv")
    parser.add_argument("--output", default="raws/poem_dataset.csv")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    run(
        metadata_file=args.metadata,
        output_file=args.output,
        resume=not args.no_resume,
    )
