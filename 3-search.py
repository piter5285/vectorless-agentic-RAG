"""
Step 3: Interactive demo of the vectorless retrieval loop.

This script shows the three-tool navigation pattern that the agent uses —
without a full agent loop. It lets you explore any indexed space manually:

  Step A) list_spaces()           → discover what's indexed
  Step B) get_space_structure()   → read the hierarchical tree
  Step C) read_section()          → fetch content from a specific line range

This makes the "vectorless" nature explicit: there is no query-embedding,
no cosine similarity, no BM25 score. The retrieval is entirely structural:
you (or the agent) read the table of contents and pick where to look.

Compare with 3-hybrid-search.py from the hybrid-RAG project, where you type
a query and receive a ranked list of chunks. Here, you navigate the tree.

Usage:
  uv run 3-search.py
"""

import json
import sys

from dotenv import load_dotenv

from utils.agent_tools import list_spaces, get_space_structure, read_section

load_dotenv()


def print_tree(node: dict, indent: int = 0) -> None:
    """Recursively print a PageIndex tree node for readability."""
    prefix = "  " * indent
    name = node.get("name", "?")
    desc = node.get("description", "")
    line = node.get("line_num", "")

    line_info = f"  [line {line}]" if line else ""
    desc_info = f"  — {desc[:80]}..." if len(desc) > 80 else (f"  — {desc}" if desc else "")
    print(f"{prefix}• {name}{line_info}{desc_info}")

    for child in node.get("sub_nodes", []):
        print_tree(child, indent + 1)


def demo_space(space_key: str) -> None:
    """Walk through the three retrieval steps for one space."""
    print(f"\n{'=' * 60}")
    print(f"Space: {space_key}")
    print("=" * 60)

    # Step B: read tree
    print("\nStep B: get_space_structure()\n")
    structure_raw = get_space_structure(space_key)
    try:
        tree = json.loads(structure_raw)
        nodes = tree if isinstance(tree, list) else tree.get("sub_nodes", [tree])
        for node in nodes[:10]:  # cap at 10 top-level nodes to keep output readable
            print_tree(node)
        total = len(nodes)
        if total > 10:
            print(f"  ... and {total - 10} more top-level sections")
    except json.JSONDecodeError:
        print(structure_raw[:2000])  # print raw if not valid JSON
        return

    # Step C: ask user for a line range to read
    print("\nStep C: read_section()")
    print(
        "Enter a line range to read (e.g. '50-100', or press Enter to skip): ",
        end="",
        flush=True,
    )
    try:
        line_input = input().strip()
    except EOFError:
        line_input = ""

    if not line_input:
        print("Skipped reading content. Pass a line range next time.")
        return

    print(f"\nContent for lines {line_input}:\n")
    content = read_section(space_key, line_input)
    print(content[:3000])
    if len(content) > 3000:
        print(f"\n... ({len(content) - 3000} more characters)")


def main() -> None:
    # Step A: list spaces
    print("Step A: list_spaces()\n")
    spaces_raw = list_spaces()

    try:
        spaces = json.loads(spaces_raw)
    except (json.JSONDecodeError, TypeError):
        print(spaces_raw)
        sys.exit(1)

    if not spaces:
        print("No spaces indexed. Run 2-build-index.py first.")
        sys.exit(1)

    print("Indexed spaces:")
    for s in spaces:
        print(f"  {s['space_key']:20s}  {s['space_name']}  ({s['page_count']} pages)")

    # Prompt user to pick a space, or use the first one if non-interactive
    print("\nEnter a space_key to explore (or press Enter for the first one): ", end="", flush=True)
    try:
        chosen = input().strip()
    except EOFError:
        chosen = ""

    if not chosen:
        chosen = spaces[0]["space_key"]
        print(f"Using: {chosen}")

    if not any(s["space_key"] == chosen for s in spaces):
        print(f"Space '{chosen}' not found in index.")
        sys.exit(1)

    demo_space(chosen)

    # Offer to continue with another space
    if len(spaces) > 1:
        print("\nOther available spaces:", [s["space_key"] for s in spaces if s["space_key"] != chosen])
        print("Run the script again to explore a different space.")


if __name__ == "__main__":
    main()
