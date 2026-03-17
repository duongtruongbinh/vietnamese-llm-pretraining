"""
Vietnamese Wikipedia Crawler using MediaWiki Action API
Endpoint: https://vi.wikipedia.org/w/api.php

Usage:
    python crawl_vi_wiki.py [--output OUTPUT_DIR] [--limit MAX_ARTICLES]
                            [--delay DELAY_SECS] [--resume]

API Docs:
    https://www.mediawiki.org/wiki/API:Main_page
    https://www.mediawiki.org/wiki/API:Query
    https://www.mediawiki.org/wiki/API:Allpages
    https://www.mediawiki.org/wiki/API:Revisions
"""

import argparse
import json
import logging
import time
import urllib.parse
from pathlib import Path

import requests

API_ENDPOINT = "https://vi.wikipedia.org/w/api.php"

# Required by API etiquette: identify your bot/tool clearly
USER_AGENT = (
    "vi-wiki-crawler/1.0 "
    "(https://github.com/example/vi-wiki-crawler; contact@example.com) "
    "python-requests/2.x"
)

# Number of page titles to fetch per allpages request (max 500 for bots, 50 for anon)
BATCH_SIZE = 50

# Number of pages whose content we fetch per query (max 50 for anon)
CONTENT_BATCH = 50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _parse_retry_after(value: str | None, default: int) -> int:
    """Parse Retry-After header, which may be an integer (seconds) or an
    HTTP-date string. Returns default on parse failure to avoid ValueError crash."""
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def api_get(session: requests.Session, params: dict, retries: int = 5) -> dict:
    """
    Make a GET request to the MediaWiki API.
    Retries on transient errors with exponential back-off.
    Raises RuntimeError if all retries fail.
    """
    params.setdefault("format", "json")
    params.setdefault("formatversion", "2")

    for attempt in range(1, retries + 1):
        try:
            resp = session.get(API_ENDPOINT, params=params, timeout=30)

            # Handle rate-limiting before raise_for_status so we can retry.
            # Skip sleep on the final attempt — there is nothing left to retry.
            if resp.status_code == 429:
                if attempt == retries:
                    break
                wait = _parse_retry_after(resp.headers.get("Retry-After"), default=60)
                logger.warning("429 Too Many Requests – waiting %d s (attempt %d/%d)", wait, attempt, retries)
                time.sleep(wait)
                continue

            resp.raise_for_status()

            # Guard against non-JSON responses (e.g. proxy error pages).
            try:
                data = resp.json()
            except ValueError as exc:
                logger.warning("Invalid JSON in response (attempt %d/%d)", attempt, retries)
                if attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError("API returned invalid JSON after all retries") from exc

            # API-level errors
            if "error" in data:
                code = data["error"].get("code", "unknown")
                info = data["error"].get("info", "")
                # maxlag: server is lagged, wait and retry
                if code == "maxlag":
                    if attempt == retries:
                        break
                    wait = _parse_retry_after(resp.headers.get("Retry-After"), default=5)
                    logger.warning("maxlag – waiting %d s (attempt %d/%d)", wait, attempt, retries)
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"API error [{code}]: {info}")

            # formatversion=2 wraps warning text in a dict; extract the string.
            if "warnings" in data:
                for module, warn in data["warnings"].items():
                    text = warn.get("warnings", str(warn)) if isinstance(warn, dict) else str(warn)
                    logger.warning("API warning [%s]: %s", module, text)

            return data

        except requests.RequestException as exc:
            logger.warning("Request failed (attempt %d/%d): %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"API request failed after {retries} attempts: {exc}") from exc

    raise RuntimeError(f"API request failed after {retries} attempts (rate-limit or maxlag not resolved)")


def load_checkpoint(path: Path) -> tuple[dict, bool]:
    """Returns (data, is_valid). is_valid=False means the file was corrupt."""
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f), True
        except json.JSONDecodeError:
            logger.warning("Checkpoint file is corrupt: %s", path)
            return {}, False
    return {}, True


def save_checkpoint(path: Path, data: dict) -> None:
    # Write to a temp file then rename atomically so a mid-write kill
    # never leaves a corrupt checkpoint.json.
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def fetch_page_contents(session: requests.Session, page_ids: list[int], delay: float) -> dict[int, dict]:
    """
    Fetch wikitext content for a list of page IDs using prop=revisions.
    Returns a dict mapping pageid -> {title, content}.
    Sends all IDs in a single batched request (pipe-separated pageids) instead
    of one request per page, reducing N requests + N delays to 1 request + 1 delay.
    """
    if not page_ids:
        return {}

    results: dict[int, dict] = {}

    params = {
        "action": "query",
        "pageids": "|".join(str(pid) for pid in page_ids),
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",          # MediaWiki 1.32+ slot-based storage
        "maxlag": 5,
    }

    data = api_get(session, params)

    for page in data["query"]["pages"]:
        page_id = page["pageid"]
        title = page.get("title", "")
        revisions = page.get("revisions", [])
        if not revisions:
            continue
        content = revisions[0].get("slots", {}).get("main", {}).get("content", "")
        results[page_id] = {"title": title, "content": content}

    time.sleep(delay)
    return results


def crawl(
    output_dir: Path,
    max_articles: int | None,
    delay: float,
    resume: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "vi_wiki_articles.jsonl"
    checkpoint_file = output_dir / "checkpoint.json"

    if resume:
        checkpoint, checkpoint_valid = load_checkpoint(checkpoint_file)
        if not checkpoint_valid and output_file.exists():
            logger.error(
                "Checkpoint is corrupt but output file exists — refusing to append "
                "to avoid duplicate records. Delete '%s' and '%s' manually to start fresh.",
                checkpoint_file, output_file,
            )
            raise SystemExit(1)
    else:
        checkpoint = {}
    seen_ids: set[int] = set(checkpoint.get("seen_ids", []))
    article_count: int = checkpoint.get("article_count", 0)
    ap_continue: str | None = checkpoint.get("ap_continue")

    logger.info(
        "Starting crawl. Output: %s | Max articles: %s | Resume: %s",
        output_file, max_articles or "unlimited", resume,
    )

    session = make_session()

    # Build initial params for allpages; reuse continue token if resuming
    allpages_params: dict = {
        "action": "query",
        "list": "allpages",
        "apnamespace": 0,
        "apfilterredir": "nonredirects",
        "aplimit": BATCH_SIZE,
        "maxlag": 5,
        "format": "json",
        "formatversion": "2",
    }
    if ap_continue:
        allpages_params["apcontinue"] = ap_continue
        allpages_params["continue"] = "-||"

    open_mode = "a" if (resume and output_file.exists()) else "w"

    with output_file.open(open_mode, encoding="utf-8") as out_f:
        while True:
            if max_articles is not None and article_count >= max_articles:
                logger.info("Reached max articles limit (%d). Stopping.", max_articles)
                break

            # ── Fetch a batch of page titles ───────────────────────────────
            # Record the token that produced this batch so mid-batch crashes
            # or limit hits can resume from here (seen_ids prevents duplicates).
            batch_start_apcontinue = allpages_params.get("apcontinue")
            data = api_get(session, allpages_params.copy())
            pages_meta = data["query"]["allpages"]
            has_more = "continue" in data
            next_ap_continue = data.get("continue", {}).get("apcontinue")

            new_page_ids = [
                p["pageid"] for p in pages_meta if p["pageid"] not in seen_ids
            ]

            reached_limit = False
            if new_page_ids:
                # ── Fetch content for this batch ───────────────────────────
                for i in range(0, len(new_page_ids), CONTENT_BATCH):
                    if reached_limit:
                        break
                    batch_ids = new_page_ids[i : i + CONTENT_BATCH]
                    contents = fetch_page_contents(session, batch_ids, delay)

                    for pid, article in contents.items():
                        if not article["content"]:
                            continue
                        record = {
                            "id": pid,
                            "title": article["title"],
                            "text": article["content"],
                            "url": "https://vi.wikipedia.org/wiki/" + urllib.parse.quote(
                                article["title"].replace(" ", "_"), safe="/:"
                            ),
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        seen_ids.add(pid)
                        article_count += 1

                        if article_count % 100 == 0:
                            logger.info("Crawled %d articles so far...", article_count)

                        if max_articles is not None and article_count >= max_articles:
                            reached_limit = True
                            break

                    # Save after every content sub-batch. Use batch_start_apcontinue
                    # so a crash or limit mid-batch resumes from this allpages batch;
                    # seen_ids prevents duplicates for already-written pages.
                    out_f.flush()
                    save_checkpoint(checkpoint_file, {
                        "seen_ids": list(seen_ids),
                        "article_count": article_count,
                        "ap_continue": batch_start_apcontinue,
                    })

            if reached_limit:
                # Last sub-batch already saved checkpoint with batch_start_apcontinue.
                # On next resume, this allpages batch is re-fetched but seen_ids
                # filters all written pages, and any skipped pages are recovered.
                break

            # All pages of this allpages batch are processed; advance the checkpoint.
            save_checkpoint(checkpoint_file, {
                "seen_ids": list(seen_ids),
                "article_count": article_count,
                "ap_continue": next_ap_continue,
            })

            if not has_more:
                logger.info("All pages enumerated. Total articles: %d", article_count)
                break

            # ── Prepare next continuation ──────────────────────────────────
            allpages_params.update(data["continue"])
            time.sleep(delay)

    logger.info("Done. %d articles saved to %s", article_count, output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl Vietnamese Wikipedia articles via MediaWiki API"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raws"),
        help="Directory to save output JSONL and checkpoint (default: ./data/raws)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Stop after crawling N articles (default: None = no limit)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        metavar="SECS",
        help="Delay in seconds between API requests (default: 1.0)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if it exists",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    crawl(
        output_dir=args.output,
        max_articles=args.limit,
        delay=args.delay,
        resume=args.resume,
    )
