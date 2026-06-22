"""
Tool functions shared between the pydantic-ai agent (4-agent.py) and the MCP
server (5-mcp-server.py).

The three-tool interface mirrors the agentic-rag tutorial pattern but uses
hierarchical tree navigation instead of similarity search:

  list_spaces          →  list_files    (discover what knowledge areas exist)
  get_space_structure  →  grep          (explore the tree to locate sections)
  read_section         →  read_file     (read the actual content of a section)

The key conceptual shift from hybrid-RAG:
  - Hybrid RAG:    agent issues a query → retriever finds similar chunks
  - Vectorless RAG: agent reads the tree → reasons about which section to open

Both the standalone agent and the MCP server import these. The PageIndex
client is a singleton — loaded once on first use, reused for all calls.
"""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_BASE_DIR = Path(__file__).parents[1]
_META_PATH = _BASE_DIR / "indexes" / "meta.json"
_WORKSPACE_DIR = _BASE_DIR / "workspace"

_client = None


def _get_client():
    """Lazy singleton for PageIndexClient."""
    global _client
    if _client is None:
        try:
            from pageindex import PageIndexClient
        except ImportError as e:
            raise RuntimeError(
                "PageIndex is not installed. "
                "Run: pip install git+https://github.com/VectifyAI/PageIndex.git"
            ) from e
        _client = PageIndexClient(workspace=str(_WORKSPACE_DIR))
    return _client


def _load_meta() -> dict:
    if not _META_PATH.exists():
        return {}
    return json.loads(_META_PATH.read_text(encoding="utf-8"))


# ------------------------------------------------------------------
# Tool 1: list_spaces
# Equivalent to list_files in agentic-rag.
# ------------------------------------------------------------------


def list_spaces() -> str:
    """
    List the knowledge spaces available in the vectorless index.

    Call this first to understand which knowledge areas are indexed and
    how many pages each space contains. Use the returned space_key values
    in get_space_structure and read_section.

    Returns a JSON array of {space_key, space_name, page_count} objects.
    """
    meta = _load_meta()
    if not meta:
        return (
            "Index not built yet. Run 2-build-index.py first.\n"
            "Make sure docs/ contains .md files from 1-fetch-confluence.py "
            "or 1-fetch-k8s.py."
        )

    spaces = [
        {
            "space_key": key,
            "space_name": entry.get("space_name", key),
            "page_count": entry.get("page_count", 0),
        }
        for key, entry in meta.items()
    ]
    return json.dumps(spaces, indent=2)


# ------------------------------------------------------------------
# Tool 2: get_space_structure
# Equivalent to grep in agentic-rag, but returns a hierarchical tree
# instead of a list of matching snippets.
# ------------------------------------------------------------------


def get_space_structure(space_key: str) -> str:
    """
    Get the full hierarchical tree structure for a Confluence space.

    This is the core of the vectorless approach: instead of running a
    similarity search, the agent reads the document's table of contents
    and reasons about which section is most likely to contain the answer.

    The tree has nodes like:
      {name, description, line_num, sub_nodes: [...]}

    Each node's line_num tells you which line in the source document that
    section starts at. Use those line numbers in read_section to fetch
    the actual content.

    Args:
        space_key: Space key returned by list_spaces (e.g. "ENG", "K8S_CONCEPTS").

    Returns the JSON tree as a formatted string. If the space has many pages,
    only top-level section names are shown — call again with a section name
    hint to drill deeper, or use read_section directly with estimated lines.
    """
    meta = _load_meta()
    if space_key not in meta:
        available = list(meta.keys())
        return (
            f"Space '{space_key}' not found.\n"
            f"Available spaces: {available}\n"
            "Call list_spaces() to see all options."
        )

    doc_id = meta[space_key]["doc_id"]
    try:
        return _get_client().get_document_structure(doc_id)
    except Exception as e:
        return f"Error fetching structure for '{space_key}': {e}"


# ------------------------------------------------------------------
# Tool 3: read_section
# Equivalent to read_file in agentic-rag.
# ------------------------------------------------------------------


def read_section(space_key: str, lines: str) -> str:
    """
    Read the raw Markdown content of a specific line range in a space document.

    Use this after get_space_structure has identified relevant sections via
    their line_num fields. Always use tight ranges — prefer "120-180" over
    "1-500". The agent can call this multiple times to read different sections.

    Args:
        space_key: Space key returned by list_spaces (e.g. "ENG", "K8S_TASKS").
        lines:     Line range to read, e.g. "120-180" (range), "42" (single line),
                   or "10,20,35" (specific lines). Use line_num values from the
                   tree returned by get_space_structure.

    Returns the raw Markdown text for those lines, which is the actual page
    content for that section. Use this to extract facts, procedures, and
    policies to answer the user's question.
    """
    meta = _load_meta()
    if space_key not in meta:
        available = list(meta.keys())
        return (
            f"Space '{space_key}' not found.\n"
            f"Available spaces: {available}\n"
            "Call list_spaces() to see all options."
        )

    doc_id = meta[space_key]["doc_id"]
    try:
        return _get_client().get_page_content(doc_id, lines)
    except Exception as e:
        return f"Error reading section lines '{lines}' in '{space_key}': {e}"
