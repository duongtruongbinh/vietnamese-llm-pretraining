"""
crawl_poem.py
Crawl thơ mới năm chữ (PoemType=16) từ thivien.net bằng Selenium.
Chiến lược: dùng nhiều Sort config để vượt giới hạn 10 trang, sau đó crawl
theo từng tác giả.
"""

# ─── 1. IMPORTS ─────────────────────────────────────────────────────────────
import re
import time
import random
import urllib.parse

import pandas as pd
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager


# ─── 2. SORT CONFIGS ─────────────────────────────────────────────────────────
SORT_CONFIGS = [
    {"sort": "",        "order": ""},
    {"sort": "Author",  "order": "asc"},
    {"sort": "Author",  "order": "desc"},
    {"sort": "Views",   "order": "desc"},
    {"sort": "Views",   "order": "asc"},
    {"sort": "Date",    "order": "asc"},
    {"sort": "Date",    "order": "desc"},
    {"sort": "Poster",  "order": "asc"},
    {"sort": "Poster",  "order": "desc"},
]

BASE_SEARCH_URL = "https://www.thivien.net/search-poem.php?PoemType=16&Country=2"
MAX_PAGES_PER_CONFIG = 10


# ─── 3. KHỞI TẠO DRIVER ─────────────────────────────────────────────────────
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


# ─── 4. HELPER FUNCTIONS ─────────────────────────────────────────────────────
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
    return (
        f"https://www.thivien.net/search-poem.php"
        f"?PoemType=16&Author={encoded}&Page={page}"
    )


def is_blocked(driver: webdriver.Chrome) -> bool:
    return "Danh sách quá dài" in driver.page_source


def get_total_pages(driver: webdriver.Chrome) -> int:
    """Parse text 'trong tổng số X trang' từ trang hiện tại."""
    try:
        el = driver.find_element(
            By.XPATH, '//*[contains(text(),"trong tổng số")]'
        )
        match = re.search(r"tổng số (\d+) trang", el.text)
        if match:
            return int(match.group(1))
    except NoSuchElementException:
        pass

    # Fallback: thử parse "Trang N/M" từ page source
    match = re.search(r"Trang\s+\d+/(\d+)", driver.page_source)
    if match:
        return int(match.group(1))

    return 1


# ─── 5. EXTRACT FUNCTIONS ────────────────────────────────────────────────────
def extract_authors_from_current_page(driver: webdriver.Chrome) -> set:
    """Lấy tên tác giả từ trang danh sách hiện tại."""
    authors = set()
    try:
        elements = driver.find_elements(
            By.XPATH,
            '//div[contains(@class,"list-item-detail")]//a[contains(@href,"/author-")]',
        )
        for el in elements:
            name = el.text.strip()
            if name:
                authors.add(name)
    except Exception as e:
        print(f"    [WARN] extract_authors: {e}")
    return authors


def extract_poem_links_from_current_page(driver: webdriver.Chrome) -> list:
    """Lấy danh sách {'title': ..., 'url': ...} từ trang danh sách hiện tại."""
    poems = []
    try:
        elements = driver.find_elements(
            By.XPATH,
            '//div[@class="list-item"]//h4[@class="list-item-header"]/a',
        )
        for el in elements:
            title = el.text.strip()
            url = el.get_attribute("href") or ""
            if url:
                poems.append({"title": title, "url": url})
    except Exception as e:
        print(f"    [WARN] extract_poem_links: {e}")
    return poems


# ─── 6. CLEAN FUNCTIONS ──────────────────────────────────────────────────────
def clean_poem_html(html: str) -> str:
    # Xóa thẻ <img>
    html = re.sub(r"<img.*?>", "", html, flags=re.IGNORECASE)
    # Xóa thẻ <i>...</i>
    html = re.sub(r"<i>.*?</i>", "", html, flags=re.IGNORECASE | re.DOTALL)
    # Giữ nội dung trong <b> nếu không theo sau bởi 2+ <br>
    html = re.sub(
        r"<b>(.*?)</b>(?!\s*(?:<br\s*/?>\s*){2,})",
        r"\1",
        html,
        flags=re.IGNORECASE,
    )
    # Chuyển <br> thành xuống dòng
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    # Xóa thẻ <p>
    html = re.sub(r"</?p>", "", html, flags=re.IGNORECASE)
    return html.strip()


def process_poem_content(
    html: str, poem_src: str, poem_url: str, default_title: str = ""
) -> list:
    cleaned = clean_poem_html(html)
    pattern = re.compile(r"<b>(.*?)</b>\s*\n{2,}", flags=re.IGNORECASE)
    matches = list(pattern.finditer(cleaned))
    poems = []
    if matches:
        for i, match in enumerate(matches):
            title = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
            content = cleaned[start:end].strip("\n")
            poems.append(
                {
                    "title": title,
                    "content": content,
                    "source": poem_src,
                    "url": poem_url,
                }
            )
    else:
        poems.append(
            {
                "title": default_title,
                "content": cleaned,
                "source": poem_src,
                "url": poem_url,
            }
        )
    return poems


# ─── 7. SCRAPE POEM ──────────────────────────────────────────────────────────
def scrape_poem(
    driver: webdriver.Chrome, poem_url: str, default_title: str = ""
) -> list:
    try:
        driver.get(poem_url)
        wait = WebDriverWait(driver, 10)
        content_div = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.poem-content"))
        )
        inner_html = content_div.get_attribute("innerHTML")

        poem_src = ""
        try:
            src_el = driver.find_element(By.XPATH, '//div[@class="small"]')
            poem_src = src_el.text.strip()
        except NoSuchElementException:
            pass

        return process_poem_content(inner_html, poem_src, poem_url, default_title)
    except Exception as e:
        print(f"    [ERROR] scrape_poem({poem_url}): {e}")
        return []


# ─── 8. BƯỚC 1: THU THẬP TÁC GIẢ ────────────────────────────────────────────
def collect_all_authors(driver: webdriver.Chrome) -> set:
    """
    Duyệt qua tất cả SORT_CONFIGS, mỗi config lấy tối đa 10 trang,
    gom tên tác giả unique vào một set.
    """
    all_authors: set = set()

    print("=" * 60)
    print("BƯỚC 1: Thu thập tên tác giả")
    print("=" * 60)

    for cfg in SORT_CONFIGS:
        sort = cfg["sort"]
        order = cfg["order"]
        label = f"Sort={sort or 'default'}/Order={order or 'default'}"
        print(f"\n[{label}]")

        for page in range(1, MAX_PAGES_PER_CONFIG + 1):
            url = build_search_url(sort, order, page)
            try:
                driver.get(url)
                time.sleep(random.uniform(2, 4))

                if is_blocked(driver):
                    print(f"  Trang {page}: bị block → dừng config này")
                    break

                page_authors = extract_authors_from_current_page(driver)
                if not page_authors:
                    print(f"  Trang {page}: không có tác giả → dừng")
                    break

                all_authors.update(page_authors)

                total_pages = get_total_pages(driver)
                if page >= total_pages:
                    break

            except Exception as e:
                print(f"  [ERROR] Trang {page}: {e}")
                break

        print(f"  [{label}] Tổng tác giả tích lũy: {len(all_authors)}")

    print(f"\n→ Tổng tác giả unique: {len(all_authors)}")
    return all_authors


# ─── 9. BƯỚC 2+3: CRAWL THƠ CỦA TÁC GIẢ ────────────────────────────────────
def crawl_author_poems(
    driver: webdriver.Chrome,
    author_name: str,
    seen_urls: set,
) -> list:
    """
    Crawl tất cả bài thơ của một tác giả (tối đa 10 trang),
    scrape từng bài thơ và trả về list dict.
    """
    collected = []

    # Lấy trang 1 để biết tổng số trang
    url_p1 = build_author_search_url(author_name, 1)
    try:
        driver.get(url_p1)
        time.sleep(random.uniform(2, 4))
    except Exception as e:
        print(f"  [ERROR] Không thể load trang tác giả {author_name}: {e}")
        return collected

    if is_blocked(driver):
        print(f"  {author_name}: bị block → bỏ qua")
        return collected

    total_pages = get_total_pages(driver)
    pages_to_crawl = min(total_pages, MAX_PAGES_PER_CONFIG)

    for page in range(1, pages_to_crawl + 1):
        if page > 1:
            url = build_author_search_url(author_name, page)
            try:
                driver.get(url)
                time.sleep(random.uniform(2, 4))
            except Exception as e:
                print(f"    [ERROR] Trang {page}: {e}")
                break

        if is_blocked(driver):
            print(f"    Trang {page}: Danh sách quá dài → dừng")
            break

        poem_links = extract_poem_links_from_current_page(driver)
        if not poem_links:
            break

        for item in poem_links:
            poem_url = item["url"]
            poem_title = item["title"]

            if poem_url in seen_urls:
                continue
            seen_urls.add(poem_url)

            results = scrape_poem(driver, poem_url, poem_title)
            collected.extend(results)
            time.sleep(random.uniform(2, 4))

    return collected


# ─── 10. PIPELINE CHÍNH ──────────────────────────────────────────────────────
def run_full_pipeline(driver: webdriver.Chrome) -> list:
    all_data = []
    seen_urls: set = set()

    # Bước 1: Thu thập tác giả
    authors = collect_all_authors(driver)

    # Bước 2+3: Crawl thơ của từng tác giả
    print("\n" + "=" * 60)
    print("BƯỚC 2+3: Crawl thơ theo từng tác giả")
    print("=" * 60)

    author_list = sorted(authors)
    for author_name in tqdm(author_list, desc="Tác giả", unit="người"):
        before = len(all_data)
        poems = crawl_author_poems(driver, author_name, seen_urls)
        all_data.extend(poems)
        added = len(all_data) - before
        print(f"  {author_name}: +{added} bài | Tổng: {len(all_data)}")
        time.sleep(random.uniform(1, 2))

    return all_data


# ─── 11. ENTRY POINT ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    driver = init_driver()
    try:
        all_data = run_full_pipeline(driver)
        df = pd.DataFrame(all_data)
        output_file = "poem_dataset.csv"
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\nHoàn thành! Tổng bài thơ: {len(all_data)} | Đã lưu vào {output_file}")
    finally:
        driver.quit()
