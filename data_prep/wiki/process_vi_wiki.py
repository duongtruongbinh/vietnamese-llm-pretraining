"""
Post-process vi_wiki_articles.jsonl:
  1. Strip wikitext markup -> plain text
  2. Save to vi_wiki_articles_clean.jsonl
  3. Convert cleaned JSONL to Parquet

Usage:
    python data_prep/wiki/process_vi_wiki.py [--input INPUT] [--output OUTPUT]
"""

import argparse
import html as _html
import json
import logging
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

from datasets import Dataset

from src.utils import configure_root_logging, format_size

logger = logging.getLogger(__name__)

_DATA_RAW = REPO_ROOT / "data" / "raws"

# Patterns applied in order
_PATTERNS = [
    # Remove HTML comments
    (re.compile(r"<!--.*?-->", re.DOTALL), ""),
    # Remove <ref>...</ref> and <ref ... />
    (re.compile(r"<ref[^>]*/\s*>", re.IGNORECASE), ""),
    (re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE), ""),
    # Remove all other HTML tags (keep text inside block tags)
    (re.compile(r"<(div|span|small|big|b|i|u|s|br|p|table|tr|th|td|ul|ol|li|dl|dt|dd|"
                r"sub|sup|blockquote|nowiki|code|pre|gallery|imagemap|timeline|score|"
                r"syntaxhighlight|source|poem|section|indicator|templatestyles|"
                r"hiero|math|chem)[^>]*>", re.IGNORECASE), " "),
    (re.compile(r"</[a-z]+>", re.IGNORECASE), " "),
    (re.compile(r"<[^>]+>"), ""),  # any remaining tags
    # Remove block-level templates (multi-line) – catch templates that span lines
    # Remove infoboxes / navboxes / sidebars (templates that start a line with {{Name)
    # We do a simpler approach: remove balanced {{ }} recursively
]

def _remove_balanced_braces(text: str) -> str:
    """Remove all {{ ... }} blocks (templates), handling nesting."""
    result = []
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i:i+2] == "{{":
            depth += 1
            i += 2
        elif text[i:i+2] == "}}":
            if depth > 0:
                depth -= 1
            i += 2
        else:
            if depth == 0:
                result.append(text[i])
            i += 1
    return "".join(result)


def _remove_wiki_tables(text: str) -> str:
    """Remove {| ... |} wiki table blocks, handling nesting."""
    result = []
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i:i+2] == "{|":
            depth += 1
            i += 2
        elif text[i:i+2] == "|}":
            if depth > 0:
                depth -= 1
                i += 2
            else:
                # Not inside a table; treat as regular characters to avoid
                # accidentally consuming the closing "|}" of wikitext infoboxes
                # (e.g. "{{Infobox ...|}}") which would corrupt the brace balance.
                result.append(text[i])
                i += 1
        else:
            if depth == 0:
                result.append(text[i])
            i += 1
    return "".join(result)


def _remove_balanced_brackets(text: str) -> str:
    """
    Convert [[File:...]] / [[Image:...]] / [[Tập tin:...]] to empty,
    and [[link|display]] -> display, [[link]] -> link.
    """
    result = []
    i = 0
    n = len(text)
    while i < n:
        if text[i:i+2] == "[[":
            # find matching ]]
            depth = 1
            j = i + 2
            while j < n and depth > 0:
                if text[j:j+2] == "[[":
                    depth += 1
                    j += 2
                elif text[j:j+2] == "]]":
                    depth -= 1
                    j += 2
                else:
                    j += 1
            inner = text[i+2:j-2]
            # File / Image / Tập tin / Category / Thể loại -> skip
            # Normalize underscore to space so "Tập_tin:" and "Tập tin:" both match
            lower = inner.lower().replace("_", " ")
            if any(lower.startswith(p) for p in (
                "file:", "image:", "tập tin:", "hình:", "ảnh:",
                "category:", "thể loại:", "media:"
            )):
                i = j
                continue
            # [[link|display]] -> display, [[link]] -> link
            parts = inner.split("|", 1)
            result.append(parts[-1])
            i = j
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


def _remove_single_brackets(text: str) -> str:
    """Remove/resolve external and interwiki single-bracket links.

    Handles:
      [http://... label]  -> label
      [http://...]        -> ''
      [wikt:word|label]   -> label   (interwiki with display text)
      [wikt:word]         -> word    (interwiki without display text, keep target)
      [fr:Article]        -> ''      (2-letter interlanguage link, invisible metadata)
    """
    # Interwiki with label: [prefix:target|label] -> label
    text = re.sub(r"\[([a-z][a-z0-9_-]*):([^\]\|]*)\|([^\]]+)\]", r"\3", text)
    # 2-letter interlanguage links: [fr:Something] -> ''
    text = re.sub(r"\[[a-z]{2}:[^\]]+\]", "", text)
    # Remaining interwiki without label: [wikt:word] -> word
    text = re.sub(r"\[[a-z][a-z0-9_-]*:([^\]]+)\]", r"\1", text)
    # External link with label: [url label] -> label
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    # External link without label: [url] -> ''
    text = re.sub(r"\[https?://[^\]]+\]", "", text)
    return text


# Horizontal rules, bold/italic markup
# Note: tables handled by _remove_wiki_tables(), section headers by _HEADERS
_MARKUP = re.compile(r"'''|''|----+")

# Magic words / behavior switches
_MAGIC = re.compile(r"__[A-Z_]+__")

# DISPLAYTITLE, DEFAULTSORT etc.
_BEHAVIOR = re.compile(r"\{\{(?:DEFAULTSORT|DISPLAYTITLE|[A-Z_]+)[^}]*\}\}", re.IGNORECASE)

# Collapse whitespace
_WHITESPACE = re.compile(r"[ \t]+")
_NEWLINES   = re.compile(r"\n{3,}")

# Section headers  == Foo == -> Foo  (allows trailing whitespace after closing ==)
_HEADERS = re.compile(r"^=+\s*(.+?)\s*=+\s*$", re.MULTILINE)

# Empty-only header lines (e.g. "====") with no text between the = signs
_EMPTY_HEADERS = re.compile(r"^=+\s*=+\s*$", re.MULTILINE)

# List/definition-list prefix: one or more of *, #, ;, : at the start of a line,
# followed by an optional space.
_LIST_PREFIX = re.compile(r"^([*#;:]+)\s?", re.MULTILINE)

# Terminal Wikipedia sections that contain metadata rather than body text.
# When found as a standalone line, everything from that line onward is removed.
# The =* at both ends also catches lines where == header markers were not stripped
# (e.g. "==Tham khảo== " with a trailing space that defeated _HEADERS).
_TERMINAL_SECTION_RE = re.compile(
    r"^\s*=*\s*("
    r"Tham\s+khảo|Chú\s+thích|Ghi\s+chú|Liên\s+kết\s+ngoài"
    r"|Xem\s+thêm|Thư\s+mục|Đọc\s+thêm|Nguồn\s+tham\s+khảo"
    r"|Chú\s+giải|Ghi\s+chú\s+và\s+tham\s+khảo"
    r"|References?|External\s+links?|See\s+also|Notes?|Bibliography|Further\s+reading"
    r")\s*=*\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _strip_list_prefixes(text: str) -> str:
    """Strip wikitext list/definition-list prefix characters from each line.

    Wiki markup          Meaning
    ----------------     -------
    * item               Unordered (bullet) list item
    # item               Ordered (numbered) list item
    ; term               Definition-list term
    : definition         Definition / generic indent

    For ; lines that contain an inline ": definition", the colon is preserved
    as a natural separator (e.g. "; Apple : A fruit" → "Apple: A fruit").

    Leading whitespace on each line is stripped before checking for prefixes
    so that indented list items (e.g. "  * item") are also handled correctly.
    """
    lines = text.splitlines()
    out = []
    for line in lines:
        stripped = line.lstrip()
        m = _LIST_PREFIX.match(stripped)
        if m:
            prefix = m.group(1)
            content = stripped[m.end():]
            # ; term : inline-definition  →  term: inline-definition
            if ";" in prefix and ": " in content:
                term, _, defn = content.partition(": ")
                content = term.strip()
                if defn.strip():
                    content += ": " + defn.strip()
            # Strip residual leading ": " left behind after template removal
            # e.g. "* {{Flag|USA}}: text" → after template removal → "* : text"
            #       → after prefix strip → ": text" → strip again → "text"
            content = re.sub(r"^[\s:]+", "", content)
            out.append(content)
        else:
            out.append(line)
    return "\n".join(out)


def clean_wikitext(raw: str) -> str:
    text = raw

    # 1. Strip HTML comments, <ref>, and other HTML tags
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)

    # 2. Remove wiki tables {| ... |} (stack-based, handles nesting)
    text = _remove_wiki_tables(text)

    # 3. Remove templates {{ ... }} (stack-based, handles nesting)
    text = _remove_balanced_braces(text)

    # Remove orphaned table row-separators, captions, and cell lines left behind
    # when a table was opened/closed by templates (e.g. {{Certification Table Top}})
    # rather than explicit {|...|} brackets.
    #   |-  table row separator
    #   |+  table caption
    #   ||  multi-cell shorthand
    #   |   table data cell (pipe followed by space or another pipe)
    text = re.sub(r"^\s*\|[-+|].*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\| [^\n]*\n?", "", text, flags=re.MULTILINE)

    # 4. Convert [[wikilinks]] to plain text
    text = _remove_balanced_brackets(text)

    # 5. Resolve/remove external links and interwiki single-bracket links
    text = _remove_single_brackets(text)

    # 6. Strip section headers  == Title == -> Title
    text = _HEADERS.sub(r"\1", text)
    # Remove leftover lines that are only = signs (empty headers)
    text = _EMPTY_HEADERS.sub("", text)

    # 7. Strip bold/italic markers and horizontal rules
    text = _MARKUP.sub("", text)

    # 8. Strip behavior switches (__TOC__, __NOTOC__, etc.)
    text = _MAGIC.sub("", text)

    # 9. Decode HTML entities (&amp; &nbsp; &mdash; &#160; &#x2014; …)
    text = _html.unescape(text)
    # &nbsp; decodes to U+00A0 (non-breaking space) – normalize to regular space
    text = text.replace("\u00a0", " ")

    # 10. Strip list/definition-list prefix characters (*, #, ;, :)
    text = _strip_list_prefixes(text)

    # 11. Remove terminal metadata sections (Tham khảo, Xem thêm, Liên kết ngoài…)
    #     and everything that follows them – these sections contain citations,
    #     external URLs and cross-references, not article body text.
    m = _TERMINAL_SECTION_RE.search(text)
    if m:
        text = text[:m.start()].rstrip()

    # 12. Collapse whitespace
    text = _WHITESPACE.sub(" ", text)
    text = _NEWLINES.sub("\n\n", text)

    # 13. Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines).strip()

    return text


def convert_jsonl_to_parquet(input_path: Path, output_parquet: Path) -> None:
    import os

    logger.info("Converting JSONL to Parquet...")
    logger.info("  Input : %s", input_path)
    logger.info("  Output: %s", output_parquet)

    # Load JSONL into list of dicts
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        logger.error("No valid records found in %s", input_path)
        return

    logger.info("  Loaded: %d records", len(records))

    # Convert to HuggingFace Dataset and save as Parquet
    ds = Dataset.from_list(records)
    ds.to_parquet(output_parquet)

    # Get file size
    file_size = os.path.getsize(output_parquet)
    logger.info("  Saved : %s (%s)", output_parquet, format_size(file_size))


def process(input_path: Path, output_path: Path) -> None:
    total = kept = 0

    with (
        input_path.open("r", encoding="utf-8") as in_f,
        output_path.open("w", encoding="utf-8") as out_f,
    ):
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("JSON decode error on line %d: %s", total, e)
                continue

            record["text"] = clean_wikitext(record.get("text", ""))
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

            if kept % 500 == 0:
                logger.info("Processed %d articles (kept %d so far)...", total, kept)

    logger.info("Done.")
    logger.info("  Total read : %d", total)
    logger.info("  Kept       : %d", kept)
    logger.info("  Output     : %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and filter vi_wiki_articles.jsonl"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_DATA_RAW / "vi_wiki_articles.jsonl",
        help="Input JSONL file (raw wikitext)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DATA_RAW / "vi_wiki_articles_clean.jsonl",
        help="Output JSONL file (cleaned text)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    configure_root_logging()
    args = parse_args()
    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    # Step 1: Clean wikitext -> JSONL
    process(args.input, args.output)

    # Step 2: Convert to Parquet -> data/train/
    logger.info("")
    logger.info("=" * 60)
    train_dir = REPO_ROOT / "data" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    output_parquet = train_dir / args.output.with_suffix(".parquet").name
    convert_jsonl_to_parquet(args.output, output_parquet)
    logger.info("=" * 60)
