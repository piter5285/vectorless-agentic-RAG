"""
Step 4: Agentic RAG over Confluence with vectorless PageIndex retrieval.

This is the vectorless counterpart to hybrid-RAG's 4-agent.py.

The tool interface is the same three-tool pattern as the agentic-rag tutorial,
but the underlying mechanics are entirely different:

  Hybrid RAG (agentic-rag):
    list_spaces   →  list_files
    hybrid_search →  grep         (BM25 + dense + RRF + rerank)
    get_page_full →  read_file

  Vectorless RAG (this file):
    list_spaces          →  list_files
    get_space_structure  →  grep   (but: returns a tree, not ranked snippets)
    read_section         →  read_file  (by line number, not page_id)

The agent prompt changes accordingly: instead of "issue a query and pick from
ranked results", the agent reads the table of contents, reasons about which
section to open, and reads it by line range. It can open multiple sections
across multiple spaces to build its answer.

The structured output is identical to hybrid-RAG so the two can be compared
directly. Citations include space_key, section title, and the line range read.

Usage:
  uv run 4-agent.py "What is our on-call escalation process?"
  uv run 4-agent.py "How do we request access to prod databases?"
"""

import logging
import sys
import time

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, UsageLimits

from utils.agent_tools import get_space_structure, list_spaces, read_section

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

AGENT_REQUEST_LIMIT = 30  # higher than hybrid-RAG because tree navigation
                           # may take more steps than similarity retrieval


# ------------------------------------------------------------------
# Structured output — same shape as hybrid-RAG for easy comparison.
# ------------------------------------------------------------------


class ConfluenceCitation(BaseModel):
    space_key: str = Field(description="Space key (e.g. 'ENG', 'K8S_CONCEPTS')")
    section_title: str = Field(description="Section or page title from the tree")
    lines_read: str = Field(description="Line range that was read (e.g. '120-180')")
    quote: str = Field(description="Exact excerpt from that section that supports the answer")


class ConfluenceAnswer(BaseModel):
    answer: str = Field(description="Answer in plain English, synthesised from the sections read")
    citations: list[ConfluenceCitation] = Field(
        description="Sections that support the answer, with direct quotes and line references"
    )


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------

agent = Agent(
    "anthropic:claude-sonnet-4-6",
    tools=[list_spaces, get_space_structure, read_section],
    output_type=ConfluenceAnswer,
    instructions=(
        "You are a Confluence search assistant for your company's internal wiki.\n\n"
        "You use VECTORLESS retrieval: instead of similarity search, you navigate a\n"
        "hierarchical table of contents and reason about where information lives.\n\n"
        "Answering process:\n"
        "1. Call list_spaces() to see which knowledge areas are indexed.\n"
        "2. Call get_space_structure(space_key) for the most relevant space.\n"
        "   Read the tree carefully — each node has a name, description, and line_num.\n"
        "3. Reason about which section(s) most likely contain the answer. Look at:\n"
        "   - Section names and descriptions\n"
        "   - Nesting (sub_nodes) to find specific sub-sections\n"
        "   - The relationship between sections (context, prerequisites, appendices)\n"
        "4. Call read_section(space_key, lines) with a TIGHT line range from the\n"
        "   node's line_num field — e.g. '120-180'. Avoid reading entire documents.\n"
        "5. If the content references other sections ('see also', 'refer to'), follow\n"
        "   those references by calling read_section with the appropriate lines.\n"
        "6. If the first space doesn't have the answer, check another space.\n"
        "7. Synthesise findings from all sections into a clear, complete answer.\n"
        "8. Cite every section you used: include space_key, section title, the exact\n"
        "   line range you read, and a direct quote from that content.\n\n"
        "Return 'Error: ...' in the answer field if you cannot find relevant content\n"
        "after checking the most likely spaces and sections."
    ),
)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or "What is our deployment process?"
    print(f"Question: {question}\n")

    start = time.perf_counter()
    result = agent.run_sync(
        question,
        usage_limits=UsageLimits(request_limit=AGENT_REQUEST_LIMIT),
    )
    elapsed = time.perf_counter() - start

    print(f"Answer:\n{result.output.answer}\n")
    print("Citations:")
    for c in result.output.citations:
        print(f"  [{c.space_key} / {c.section_title}] (lines {c.lines_read})")
        for line in c.quote.splitlines():
            print(f"    {line}")

    usage = result.usage()
    print(
        f"\nUsage: {usage.requests} requests | {usage.tool_calls} tool calls | "
        f"{usage.input_tokens} in + {usage.output_tokens} out tokens | {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
