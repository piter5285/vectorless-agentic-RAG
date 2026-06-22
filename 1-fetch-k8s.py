"""
Alternative to 1-fetch-confluence.py: scrape the Kubernetes public docs.

Instead of creating chunks (as in hybrid-RAG), this script creates one
Markdown file per K8s documentation section, grouped by topic:
  docs/K8S_CONCEPTS.md   — /docs/concepts/
  docs/K8S_TASKS.md      — /docs/tasks/
  docs/K8S_TUTORIALS.md  — /docs/tutorials/
  docs/K8S_SETUP.md      — /docs/setup/
  docs/K8S_REFERENCE.md  — /docs/reference/glossary/ + /docs/reference/kubectl/

Each file contains all pages in that section, with each page as a ## section,
so that PageIndex can build a hierarchical tree over the natural doc structure.

After this runs, the rest of the pipeline is identical to the Confluence flow:
  uv run 2-build-index.py
  uv run 3-search.py
  uv run 4-agent.py "How do I configure resource limits?"
  uv run streamlit run 6-chatbot.py

Rate limiting: 0.5 s between requests — respectful for a public docs site.
Kubernetes.io robots.txt allows crawling documentation pages.
"""

import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────

DOCS_DIR = Path(__file__).parent / "docs"
DOCS_DIR.mkdir(exist_ok=True)

BASE_URL = "https://kubernetes.io"
SITEMAP_URL = f"{BASE_URL}/en/sitemap.xml"

# Each entry: (doc_key, url_prefix, display_name)
# Pages whose path starts with url_prefix go into docs/K8S_{doc_key}.md
SECTIONS: list[tuple[str, str, str]] = [
    ("K8S_CONCEPTS",   "/docs/concepts/",           "Kubernetes Concepts"),
    ("K8S_TASKS",      "/docs/tasks/",              "Kubernetes Tasks"),
    ("K8S_TUTORIALS",  "/docs/tutorials/",          "Kubernetes Tutorials"),
    ("K8S_SETUP",      "/docs/setup/",              "Kubernetes Setup"),
    ("K8S_REFERENCE",  "/docs/reference/glossary/", "Kubernetes Reference"),
    ("K8S_REFERENCE",  "/docs/reference/kubectl/",  "Kubernetes Reference"),
]

# Prefixes to always skip (too large, low signal for conversational queries)
SKIP_PATTERNS = [
    "/docs/reference/kubernetes-api/",
    "/docs/contribute/",
]

REQUEST_DELAY = 0.5  # seconds between HTTP requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; k8s-vectorless-rag/1.0; "
        "educational project; github.com/anthropics/anthropic-cookbook)"
    )
}

# ── Sitemap ───────────────────────────────────────────────────────────────────


def _sitemap_urls(url: str) -> list[str]:
    """Recursively resolve a sitemap or sitemap-index and return all <loc> URLs."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    children = root.findall("sm:sitemap/sm:loc", ns)
    if children:
        urls: list[str] = []
        for child in children:
            urls.extend(_sitemap_urls(child.text.strip()))
        return urls

    return [loc.text.strip() for loc in root.findall("sm:url/sm:loc", ns)]


def build_section_map() -> dict[str, list[str]]:
    """Return {doc_key: [url, ...]} by matching sitemap URLs against SECTIONS."""
    print(f"Fetching sitemap: {SITEMAP_URL}")
    all_urls = _sitemap_urls(SITEMAP_URL)

    # Build fast lookup: url_prefix → doc_key
    prefix_to_key: dict[str, str] = {}
    for doc_key, prefix, _ in SECTIONS:
        prefix_to_key[prefix] = doc_key

    section_map: dict[str, list[str]] = {}

    for url in sorted(set(all_urls)):
        path = urlparse(url).path
        if any(path.startswith(skip) for skip in SKIP_PATTERNS):
            continue
        for prefix, key in prefix_to_key.items():
            if path.startswith(prefix):
                section_map.setdefault(key, []).append(url)
                break  # first match wins

    total = sum(len(v) for v in section_map.values())
    print(f"Found {total} pages across {len(section_map)} sections\n")
    return section_map


# ── Content extraction ────────────────────────────────────────────────────────


def extract(html: str, url: str) -> tuple[str, str]:
    """Parse a Kubernetes docs page and return (title, markdown_text)."""
    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(" ", strip=True)
    elif soup.title:
        title = soup.title.get_text(" ", strip=True).replace(" | Kubernetes", "").strip()
    else:
        title = urlparse(url).path.rstrip("/").rsplit("/", 1)[-1].replace("-", " ").title()

    content = (
        soup.find(class_="td-content")
        or soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find("article")
    )
    if not content:
        return title, ""

    for tag in content.find_all(["nav", "script", "style", "aside", "footer", "noscript"]):
        tag.decompose()
    for tag in content.find_all(class_=["feedback--widget", "td-sidebar-toc", "td-page-meta"]):
        tag.decompose()

    # Convert to simple markdown by walking the tree
    lines: list[str] = []
    for elem in content.descendants:
        if not hasattr(elem, "name"):
            continue
        name = elem.name
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(name[1]) + 1  # shift: h1→##, h2→###, etc. (page is already ##)
            level = min(level, 6)
            text = elem.get_text(" ", strip=True)
            if text:
                lines.append(f"\n{'#' * level} {text}\n")

    # Fall back to plain text if heading walk produced nothing meaningful
    text = content.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Use the structured lines if we got headings, otherwise raw text
    if any(l.startswith("#") for l in lines):
        structured = "\n".join(lines).strip()
        # Append raw body below the headings outline
        return title, f"{structured}\n\n{text}"

    return title, text


# ── Document writing ──────────────────────────────────────────────────────────


def build_section_doc(
    doc_key: str,
    display_name: str,
    urls: list[str],
) -> None:
    """Fetch all URLs for a section and write docs/{doc_key}.md."""
    print(f"\n{'=' * 60}")
    print(f"Section: {display_name} ({doc_key})  —  {len(urls)} pages")
    print("=" * 60)

    fetched_at = datetime.now(timezone.utc).isoformat()
    header = "\n".join([
        f"<!-- section: {doc_key} | name: {display_name} | pages: {len(urls)} | fetched: {fetched_at} -->",
        "",
        f"# {display_name}",
        "",
        f"> Kubernetes documentation section **{doc_key}** — {len(urls)} pages indexed on {fetched_at[:10]}",
        "",
    ])

    pages_md: list[str] = []
    ok = skipped = errors = 0

    for i, url in enumerate(urls, 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()

            title, text = extract(resp.text, url)

            if not text.strip():
                skipped += 1
                print(f"  [{i:3}/{len(urls)}] SKIP  {url}")
                time.sleep(REQUEST_DELAY)
                continue

            block = "\n".join([
                f"## Page: {title}",
                f"<!-- url: {url} | last_modified: {resp.headers.get('Last-Modified', '')} -->",
                "",
                text,
                "",
                "---",
                "",
            ])
            pages_md.append(block)
            ok += 1
            print(f"  [{i:3}/{len(urls)}] {title[:60]}")

        except requests.HTTPError as e:
            errors += 1
            print(f"  [{i:3}/{len(urls)}] HTTP {e.response.status_code}  {url}")
        except Exception as e:
            errors += 1
            print(f"  [{i:3}/{len(urls)}] ERROR  {url}  —  {e}")

        time.sleep(REQUEST_DELAY)

    if not pages_md:
        print(f"  No content found — skipping {doc_key}.")
        return

    doc_path = DOCS_DIR / f"{doc_key}.md"
    doc_path.write_text(header + "\n".join(pages_md), encoding="utf-8")
    size_kb = doc_path.stat().st_size // 1024
    print(f"\n  Saved: {doc_path}  ({size_kb} KB, {ok} pages, {skipped} empty, {errors} errors)")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    section_map = build_section_map()

    # Collect display names per doc_key (SECTIONS may have multiple prefixes per key)
    key_to_name: dict[str, str] = {}
    for doc_key, _, display_name in SECTIONS:
        key_to_name[doc_key] = display_name  # last one wins (same key → same name)

    for doc_key, urls in sorted(section_map.items()):
        build_section_doc(doc_key, key_to_name[doc_key], urls)

    doc_files = list(DOCS_DIR.glob("K8S_*.md"))
    print(f"\nDone: {len(doc_files)} section files in {DOCS_DIR}/")
    print("\nNext: run 2-build-index.py to build PageIndex trees.")


if __name__ == "__main__":
    main()
