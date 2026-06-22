"""
Step 6: Streamlit chatbot with MCP-powered vectorless Confluence search.

The chatbot connects to 5-mcp-server.py as an MCP client and uses Claude
to drive the agentic tree-navigation loop automatically.

How it works (vectorless, no similarity search):
  User types a question
    → Claude calls list_confluence_spaces() to discover knowledge areas
    → Claude calls get_confluence_space_structure() to read the tree
    → Claude reasons about which sections match the question (no vector math)
    → Claude calls read_confluence_section() with tight line ranges
    → Claude synthesises the content into an answer with citations
    → Answer is shown with an expandable "Tools used" section

Start the MCP server first (in a separate terminal):
  uv run 5-mcp-server.py

Then start the chatbot:
  uv run streamlit run 6-chatbot.py

Dependencies (all already in pyproject.toml):
  streamlit, anthropic, mcp
"""

import asyncio
import concurrent.futures
import os

import anthropic
import streamlit as st
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8052/sse")
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to your company's Confluence wiki. "
    "You use VECTORLESS retrieval — instead of keyword search, you navigate a "
    "hierarchical table of contents and reason about where information lives.\n\n"
    "When a user asks about internal processes, documentation, policies, "
    "architecture decisions, runbooks, or onboarding:\n"
    "1. Call list_confluence_spaces() to see what's indexed.\n"
    "2. Call get_confluence_space_structure() for the most relevant space — "
    "   read the tree and identify sections by their line_num fields.\n"
    "3. Call read_confluence_section() with tight line ranges (e.g. '120-180') "
    "   to read the actual content of relevant sections.\n"
    "4. Follow cross-references if needed by reading adjacent sections.\n"
    "5. Synthesise findings and cite the space_key, section title, and lines read.\n\n"
    "If you cannot find relevant information after checking the most likely sections, "
    "say so clearly rather than guessing."
)

# ── Async/MCP helpers ─────────────────────────────────────────────────────────


def _run_in_thread(coro):
    """Run an async coroutine in a fresh thread with its own event loop."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result(timeout=90)


async def _fetch_tools_async(url: str):
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return result.tools


async def _call_tool_async(url: str, name: str, args: dict) -> str:
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, args)
            return (
                "\n".join(b.text for b in result.content if hasattr(b, "text"))
                or "No result returned."
            )


def fetch_tools(url: str) -> tuple[list, str | None]:
    try:
        tools = _run_in_thread(_fetch_tools_async(url))
        return tools, None
    except Exception as e:
        return [], str(e)


def call_tool(url: str, name: str, args: dict) -> str:
    try:
        return _run_in_thread(_call_tool_async(url, name, args))
    except Exception as e:
        return f"Error calling {name}: {e}"


# ── Agent loop ────────────────────────────────────────────────────────────────


def _serialize_blocks(blocks) -> list[dict]:
    """Convert Anthropic SDK content blocks to plain dicts for re-sending."""
    out = []
    for b in blocks:
        if b.type == "text":
            out.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
    return out


def _to_anthropic_tools(mcp_tools) -> list[dict]:
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.inputSchema,
        }
        for t in mcp_tools
    ]


def run_agent(
    question: str,
    api_history: list[dict],
    mcp_tools,
    mcp_url: str,
    status_placeholder,
) -> tuple[str, list[dict], list[dict]]:
    """
    Run the Claude + MCP tool loop for one user turn.

    Returns:
        answer       — plain text answer from Claude
        new_history  — updated API message list (pass back next turn)
        tool_log     — list of {tool, input, result} for display
    """
    client = anthropic.Anthropic()
    anthropic_tools = _to_anthropic_tools(mcp_tools)
    messages = api_history + [{"role": "user", "content": question}]
    tool_log: list[dict] = []

    while True:
        kwargs: dict = dict(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        response = client.messages.create(**kwargs)

        if response.stop_reason == "end_turn":
            answer = "".join(b.text for b in response.content if hasattr(b, "text"))
            messages.append(
                {"role": "assistant", "content": _serialize_blocks(response.content)}
            )
            return answer, messages, tool_log

        messages.append(
            {"role": "assistant", "content": _serialize_blocks(response.content)}
        )

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            status_placeholder.write(f"Calling `{block.name}`…")
            result = call_tool(mcp_url, block.name, block.input)

            tool_log.append(
                {
                    "tool": block.name,
                    "input": block.input,
                    "result": result[:500],
                }
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
            )

        messages.append({"role": "user", "content": tool_results})


# ── Streamlit app ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Confluence Assistant (Vectorless)",
    page_icon="🌲",
    layout="wide",
)

# ── Session state init ────────────────────────────────────────────────────────

if "api_history" not in st.session_state:
    st.session_state.api_history: list[dict] = []

if "display_history" not in st.session_state:
    st.session_state.display_history: list[dict] = []

if "mcp_tools" not in st.session_state:
    st.session_state.mcp_tools = None

if "mcp_error" not in st.session_state:
    st.session_state.mcp_error: str | None = None

if "mcp_url" not in st.session_state:
    st.session_state.mcp_url: str = DEFAULT_MCP_URL

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    new_url = st.text_input("MCP Server URL", value=st.session_state.mcp_url)
    if new_url != st.session_state.mcp_url:
        st.session_state.mcp_url = new_url
        st.session_state.mcp_tools = None  # force reconnect

    connect_clicked = st.button("Connect / Refresh", use_container_width=True)

    if st.session_state.mcp_tools is None or connect_clicked:
        with st.spinner("Connecting to MCP server…"):
            tools, err = fetch_tools(st.session_state.mcp_url)
            st.session_state.mcp_tools = tools
            st.session_state.mcp_error = err

    if st.session_state.mcp_error:
        st.error(f"Connection failed:\n{st.session_state.mcp_error}")
        st.caption("Is `5-mcp-server.py` running?")
    else:
        n = len(st.session_state.mcp_tools or [])
        st.success(f"Connected — {n} tool{'s' if n != 1 else ''} available")

        if st.session_state.mcp_tools:
            with st.expander("Available tools"):
                for t in st.session_state.mcp_tools:
                    st.markdown(f"**`{t.name}`**")
                    if t.description:
                        st.caption(
                            t.description[:120] + ("…" if len(t.description) > 120 else "")
                        )

    st.divider()

    st.info(
        "**Vectorless retrieval**\n\n"
        "No vector embeddings or BM25.\n"
        "Claude navigates a hierarchical tree to find relevant sections."
    )

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.api_history = []
        st.session_state.display_history = []
        st.rerun()

    st.caption(f"Model: `{MODEL}`")

# ── Main chat area ────────────────────────────────────────────────────────────

st.title("🌲 Confluence Assistant (Vectorless RAG)")
st.caption(
    "Ask anything about your company's internal documentation. "
    "Claude navigates the document tree — no keyword matching, no vector similarity."
)

for msg in st.session_state.display_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tool_log"):
            with st.expander(f"🔧 {len(msg['tool_log'])} tool call(s)"):
                for entry in msg["tool_log"]:
                    st.markdown(f"**`{entry['tool']}`**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption("Input")
                        st.json(entry["input"])
                    with col2:
                        st.caption("Result preview")
                        st.text(entry["result"])

if question := st.chat_input(
    "Ask a question about your Confluence docs…",
    disabled=bool(st.session_state.mcp_error),
):
    st.session_state.display_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.status("Navigating document tree…", expanded=True) as status:
            status_slot = st.empty()
            answer, new_history, tool_log = run_agent(
                question=question,
                api_history=st.session_state.api_history,
                mcp_tools=st.session_state.mcp_tools or [],
                mcp_url=st.session_state.mcp_url,
                status_placeholder=status_slot,
            )
            status.update(label="Done", state="complete", expanded=False)

        st.markdown(answer)

        if tool_log:
            with st.expander(f"🔧 {len(tool_log)} tool call(s)"):
                for entry in tool_log:
                    st.markdown(f"**`{entry['tool']}`**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption("Input")
                        st.json(entry["input"])
                    with col2:
                        st.caption("Result preview")
                        st.text(entry["result"])

    st.session_state.api_history = new_history
    st.session_state.display_history.append(
        {"role": "assistant", "content": answer, "tool_log": tool_log}
    )
