"""
Step 1: Fetch all Confluence pages and save them as Markdown documents.

Key difference from hybrid-RAG's 1-fetch-confluence.py:
  - Hybrid RAG:     each page → many small JSON chunks (for BM25 + embeddings)
  - Vectorless RAG: each space → one large Markdown file (for PageIndex tree)

For each space in CONFLUENCE_SPACE_KEYS this script:
  1. Lists every page in the space via the Confluence REST API.
  2. Converts the HTML body to Markdown, preserving heading hierarchy.
  3. Writes all pages into a single docs/<SPACE_KEY>.md file, with each
     page as a ## section so PageIndex can build a hierarchical tree over it.

PageIndex will then (in step 2) read this Markdown and build a tree where:
  - Top level  → the space itself
  - Sub-nodes  → individual pages (the ## sections)
  - Leaf nodes → headings within pages (### and deeper)

Run once (or after a Confluence update) before building indexes in step 2.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from utils.confluence import ConfluenceClient

load_dotenv()

DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)

SPACE_KEYS = [
    s.strip()
    for s in os.environ.get("CONFLUENCE_SPACE_KEYS", "").split(",")
    if s.strip()
]


def page_to_markdown(client: ConfluenceClient, raw: dict, space_key: str) -> str:
    """
    Convert one Confluence page to a Markdown block suitable for inclusion in
    the space document.

    Returns a string like:
        ## Page: Deployment Process
        <!-- page_id: 12345 | url: https://... | last_modified: 2024-01-10 -->

        ### Overview
        ...markdown body...

        ---
    """
    page_id = raw["id"]
    title = raw["title"]
    html = raw.get("body", {}).get("storage", {}).get("value", "")
    url = client.page_url(space_key, page_id)
    last_modified = raw.get("version", {}).get("when", "")

    # Convert body HTML to Markdown; headings shift by 2 so h1→###, h2→####
    body_md = ConfluenceClient.html_to_markdown(html, heading_offset=2)

    if not body_md.strip():
        return ""

    lines = [
        f"## Page: {title}",
        f"<!-- page_id: {page_id} | url: {url} | last_modified: {last_modified} -->",
        "",
        body_md,
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def process_space(client: ConfluenceClient, space_key: str) -> None:
    """Fetch all pages in a space and write them to docs/<SPACE_KEY>.md."""
    print(f"\nSpace: {space_key}")

    # Get space metadata for the document header
    spaces_meta = {s["key"]: s for s in client.list_spaces()}
    space_name = spaces_meta.get(space_key, {}).get("name", space_key)

    pages_md: list[str] = []
    page_count = 0
    skipped = 0

    for raw_page in client.iter_pages(space_key):
        block = page_to_markdown(client, raw_page, space_key)
        if block:
            pages_md.append(block)
            page_count += 1
        else:
            skipped += 1
        if page_count % 20 == 0 and page_count > 0:
            print(f"  {page_count} pages processed so far...")

    if not pages_md:
        print(f"  No content found in space {space_key} — skipping.")
        return

    fetched_at = datetime.now(timezone.utc).isoformat()
    header = "\n".join([
        f"<!-- space: {space_key} | name: {space_name} | pages: {page_count} | fetched: {fetched_at} -->",
        "",
        f"# {space_name}",
        "",
        f"> Confluence space **{space_key}** — {page_count} pages indexed on {fetched_at[:10]}",
        "",
    ])

    doc_path = DOCS_DIR / f"{space_key}.md"
    doc_path.write_text(header + "\n".join(pages_md), encoding="utf-8")

    # Save metadata alongside the document
    meta_entry = {
        "space_key": space_key,
        "space_name": space_name,
        "page_count": page_count,
        "skipped": skipped,
        "fetched_at": fetched_at,
        "doc_path": str(doc_path),
    }
    meta_path = DOCS_DIR / f"{space_key}.meta.json"
    meta_path.write_text(json.dumps(meta_entry, indent=2), encoding="utf-8")

    print(f"  Done: {page_count} pages → {doc_path} ({doc_path.stat().st_size // 1024} KB)")
    if skipped:
        print(f"  Skipped {skipped} empty pages.")


def main() -> None:
    if not SPACE_KEYS:
        print("CONFLUENCE_SPACE_KEYS is not set in .env.\n")
        print("Listing available spaces so you can choose which to index:")
        client = ConfluenceClient()
        for s in client.list_spaces():
            print(f"  {s['key']:20s}  {s['name']}")
        print("\nSet CONFLUENCE_SPACE_KEYS=KEY1,KEY2 in .env and re-run.")
        return

    client = ConfluenceClient()

    for space_key in SPACE_KEYS:
        process_space(client, space_key)

    print(f"\nAll spaces saved to {DOCS_DIR}/")
    print("Next: run 2-build-index.py to build PageIndex trees.")


if __name__ == "__main__":
    main()
