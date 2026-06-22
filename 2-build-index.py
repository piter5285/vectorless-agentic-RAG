"""
Step 2: Build PageIndex hierarchical tree indexes from the Markdown documents.

Key difference from hybrid-RAG's 2-build-index.py:
  - Hybrid RAG:     BM25 sparse index + NumPy dense embeddings + meta.json
  - Vectorless RAG: PageIndex JSON tree index + indexes/meta.json

For each .md file in docs/ this script:
  1. Checks if the document is already indexed (cached in workspace/).
  2. Calls PageIndexClient.index(md_path) to build the hierarchical tree.
     PageIndex uses an LLM to analyse the document structure and produce a
     machine-readable table of contents with section boundaries and summaries.
  3. Records the {space_key → doc_id} mapping in indexes/meta.json.

What gets stored:
  workspace/    — PageIndex internal workspace (tree JSON, raw page cache)
  indexes/meta.json — {space_key: {doc_id, space_name, page_count}} for tools

No vectors, no BM25, no embeddings matrix.  The "index" is purely structural:
a JSON tree that tells the agent where sections begin and what they cover.

Cost note: PageIndex calls the LLM (OpenAI by default) once per document
section during indexing. For a typical 50-page Confluence space this is
~100-200 LLM calls to gpt-4o-mini, roughly $0.01-0.05. Run once; the
workspace cache avoids re-indexing unchanged documents.

Re-run after a fresh 1-fetch-confluence.py to pick up new spaces or pages.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = Path("docs")
INDEX_DIR = Path("indexes")
INDEX_DIR.mkdir(exist_ok=True)
WORKSPACE_DIR = Path("workspace")
WORKSPACE_DIR.mkdir(exist_ok=True)


def load_doc_meta(space_key: str) -> dict:
    """Load the .meta.json file saved by 1-fetch-confluence.py or 1-fetch-k8s.py."""
    meta_path = DOCS_DIR / f"{space_key}.meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {"space_key": space_key, "space_name": space_key, "page_count": 0}


def main() -> None:
    try:
        from pageindex import PageIndexClient
    except ImportError:
        print(
            "PageIndex is not installed.\n\n"
            "Install it from GitHub:\n"
            "  pip install git+https://github.com/VectifyAI/PageIndex.git\n"
            "or:\n"
            "  git clone https://github.com/VectifyAI/PageIndex.git\n"
            "  cd PageIndex && pip install -e .\n"
        )
        return

    md_files = sorted(DOCS_DIR.glob("*.md"))
    if not md_files:
        print(
            f"No .md files found in {DOCS_DIR}/\n"
            "Run 1-fetch-confluence.py or 1-fetch-k8s.py first."
        )
        return

    print(f"Found {len(md_files)} document(s) to index in {DOCS_DIR}/\n")

    client = PageIndexClient(workspace=str(WORKSPACE_DIR))

    # Load existing meta so we can preserve entries from previous runs
    meta_path = INDEX_DIR / "meta.json"
    existing_meta: dict = {}
    if meta_path.exists():
        existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    updated_meta = dict(existing_meta)

    for md_path in md_files:
        space_key = md_path.stem  # e.g. "ENG" from docs/ENG.md
        doc_meta = load_doc_meta(space_key)
        space_name = doc_meta.get("space_name", space_key)
        page_count = doc_meta.get("page_count", 0)

        print(f"Space: {space_key}  ({space_name},  {page_count} pages)")

        # Check if already indexed by filename in the workspace
        cached_doc_id = next(
            (
                did
                for did, doc in client.documents.items()
                if doc.get("doc_name") == md_path.name
            ),
            None,
        )

        if cached_doc_id:
            print(f"  Already indexed — doc_id: {cached_doc_id}  (skipping)")
            doc_id = cached_doc_id
        else:
            print(f"  Indexing {md_path.name} ...")
            print("  (PageIndex uses the LLM to analyse document structure — may take a moment)")
            doc_id = client.index(md_path)
            print(f"  Indexed — doc_id: {doc_id}")

        updated_meta[space_key] = {
            "doc_id": doc_id,
            "space_name": space_name,
            "page_count": page_count,
            "md_path": str(md_path),
        }

    # Save the updated meta
    meta_path.write_text(
        json.dumps(updated_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved indexes/meta.json ({len(updated_meta)} entries)")

    # Print a quick summary of what was indexed
    print("\nIndexed spaces:")
    for key, entry in updated_meta.items():
        print(f"  {key:20s}  doc_id={entry['doc_id'][:8]}...  ({entry['page_count']} pages)")

    print("\nAll indexes built.")
    print("  Run 3-search.py to explore the tree interactively.")
    print("  Run 4-agent.py 'your question' to try the full agent.")
    print("  Run 5-mcp-server.py to start the MCP server.")


if __name__ == "__main__":
    main()
