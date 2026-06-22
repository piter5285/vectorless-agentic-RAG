"""
Step 5: MCP server — Confluence Vectorless RAG (PageIndex).

Exposes three tools to any MCP client (Claude Desktop, Cursor, Claude Code,
or anything that speaks MCP):

  list_confluence_spaces         discover which knowledge areas are indexed
  get_confluence_space_structure get the hierarchical tree for a space
  read_confluence_section        read content from a specific line range

The MCP client (Claude) drives the agentic loop itself — it reads the tree,
reasons about which sections to open, and calls read_confluence_section with
tight line ranges. No vector similarity, no BM25, no pre-scored chunks.

Port 8052 (one higher than hybrid-RAG's 8051 — both can run simultaneously).

──────────────────────────────────────────
Run the server
──────────────────────────────────────────
  uv run 5-mcp-server.py

──────────────────────────────────────────
Connect from Claude Desktop
──────────────────────────────────────────
Add to ~/Library/Application Support/Claude/claude_desktop_config.json
(macOS) or %APPDATA%\\Claude\\claude_desktop_config.json (Windows):

  {
    "mcpServers": {
      "confluence-vectorless": {
        "url": "http://localhost:8052/sse"
      }
    }
  }

──────────────────────────────────────────
Connect from Claude Code (.mcp.json in repo root)
──────────────────────────────────────────
  {
    "mcpServers": {
      "confluence-vectorless": {
        "type": "sse",
        "url": "http://localhost:8052/sse"
      }
    }
  }

──────────────────────────────────────────
Connect from Cursor (Settings → MCP)
──────────────────────────────────────────
  Server URL: http://localhost:8052/sse

──────────────────────────────────────────
Example prompts after connecting
──────────────────────────────────────────
  "Search Confluence for our production deployment process"
  "Find the on-call escalation policy in the ENG space"
  "What does our security policy say about SSH key rotation?"
"""

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from utils.agent_tools import get_space_structure, list_spaces, read_section

load_dotenv()

mcp = FastMCP(
    name="ConfluenceVectorlessRAG",
    host="0.0.0.0",
    port=8052,
    stateless_http=True,
)


@mcp.tool()
def list_confluence_spaces() -> str:
    """
    List all Confluence spaces available in the vectorless index.

    Call this first to understand which knowledge areas are indexed.
    Returns a JSON array of {space_key, space_name, page_count} objects.

    Use a space_key from this list as the first argument in
    get_confluence_space_structure to explore that space's table of contents.
    """
    return list_spaces()


@mcp.tool()
def get_confluence_space_structure(space_key: str) -> str:
    """
    Get the full hierarchical table of contents for a Confluence space.

    This is the core of the vectorless retrieval approach. Instead of running
    a similarity search, you read the document tree and reason about which
    section is most relevant. Each tree node includes:
      - name: section or page title
      - description: summary of what the section covers
      - line_num: the line number where this section starts in the document

    Use the line_num values from this tree in read_confluence_section to
    fetch the actual content. Always read TIGHT ranges (e.g. "120-180"),
    not entire documents.

    Args:
        space_key: Space key from list_confluence_spaces (e.g. "ENG", "K8S_CONCEPTS").
    """
    return get_space_structure(space_key)


@mcp.tool()
def read_confluence_section(space_key: str, lines: str) -> str:
    """
    Read the raw Markdown content of a specific line range in a space document.

    Use this after get_confluence_space_structure identifies the relevant
    sections via their line_num fields. Call with tight ranges — e.g. "120-180"
    rather than "1-5000". You can call this multiple times to read different
    sections or follow cross-references.

    Args:
        space_key: Space key from list_confluence_spaces (e.g. "ENG", "K8S_TASKS").
        lines:     Line range to read. Format options:
                     "120-180"  — lines 120 to 180 (range)
                     "42"       — single line 42
                     "10,20,35" — specific individual lines

    Returns the raw Markdown text for those lines. This is the actual page
    content — use it to extract facts, procedures, and policies for your answer.
    """
    return read_section(space_key, lines)


if __name__ == "__main__":
    transport = "sse"
    print("Confluence Vectorless RAG MCP server starting on http://0.0.0.0:8052")
    print("Transport: SSE (Server-Sent Events)")
    print("Connect from Claude Desktop, Cursor, or Claude Code — see file header.")
    mcp.run(transport=transport)
