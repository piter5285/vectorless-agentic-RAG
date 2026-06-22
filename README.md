# Confluence Vectorless Agentic RAG + MCP

This project implements vectorless agentic RAG over Confluence using Python, PageIndex, OpenAI gpt-4o-mini, Claude via Pydantic-AI, and FastMCP. An MCP server and Streamlit chatbot are implemented as core functionalities, replacing vector similarity search with LLM reasoning over a hierarchical document tree — no embeddings or reranker needed.

---

## The core idea: reasoning over structure, not similarity

Traditional RAG pipelines answer the question *"which chunk is most similar to
this query?"* using cosine distance or BM25 scores. PageIndex asks a different
question: *"given the document's table of contents, where does the answer
most likely live?"*

```
Traditional (hybrid-RAG):
  query → embed → cosine similarity → top-k chunks → rerank → answer

Vectorless (this project):
  query → read tree → reason about sections → read section by line → answer
```

The "index" is a hierarchical JSON tree — a machine-readable table of contents —
built once by an LLM analysing the document's structure. At query time, the
agent reads this tree, identifies the relevant section by its `line_num` field,
and fetches exactly those lines. No vectors, no BM25, no embeddings matrix.

---

## When to use vectorless RAG

### Strengths over hybrid retrieval

| Problem with hybrid RAG | Why vectorless avoids it |
|-------------------------|--------------------------|
| Fixed-size chunking splits sentences, code blocks, and tables across boundaries | PageIndex uses natural section boundaries from the document's own headings |
| "Similar text ≠ relevant text" — embedding similarity conflates topics | The agent reasons about section *purpose*, not surface similarity |
| In-document cross-references ("see Appendix B") are silently dropped | The agent can follow references by reading the referenced section |
| Multi-turn context is lost — each query is independent | The tree is stable; the agent can re-visit sections across conversation turns |
| Rare technical terms get diluted in dense embeddings | Tree navigation is label-based, not frequency-based |

### When hybrid RAG is still better

| Scenario | Why hybrid wins |
|----------|----------------|
| Very large corpora (> 1 M short snippets) | A flat tree becomes unwieldy; BM25 scales better |
| Unstructured content (chat logs, raw emails) | No headings to build a tree from |
| Low-latency requirements (< 200 ms) | Tree navigation requires LLM reasoning at query time |
| Simple keyword lookups | BM25 is faster and cheaper for exact-match queries |

### At a glance

| Dimension | Hybrid RAG | Vectorless RAG |
|-----------|-----------|----------------|
| Index type | BM25 + NumPy embeddings | JSON tree (no vectors) |
| Index build cost | Cheap (embeddings API) | Higher (LLM per section) |
| Query cost | Cheap (cosine + rerank) | Higher (LLM reasoning) |
| Chunking | Fixed 1500-char splits | Natural document sections |
| Cross-reference handling | Broken | Native (agent follows references) |
| Dependencies | OpenAI + Cohere + bm25s | PageIndex + OpenAI (indexing only) |
| MCP port | 8051 | 8052 |

---

## Architecture

```
[Confluence API]         [Public docs (K8s)]
       |                        |
1-fetch-confluence.py   1-fetch-k8s.py
       |                        |
       └──────────┬─────────────┘
                  ↓
          docs/<SPACE_KEY>.md
          (one Markdown file per space/section;
           all pages concatenated as ## sections)
                  ↓
          2-build-index.py
          (PageIndexClient.index() — LLM builds tree)
                  ↓
          workspace/          ← PageIndex JSON trees (no .npy, no BM25)
          indexes/meta.json   ← space_key → doc_id mapping
                  ↓
     ┌────────────┴────────────┐
     │       3 tools           │
     │  list_spaces            │  ← discover indexed knowledge areas
     │  get_space_structure    │  ← read the hierarchical tree
     │  read_section           │  ← fetch content by line range
     └────────────┬────────────┘
                  ↓
     ┌────────────┴────────────┐
     │  4-agent.py             │  pydantic-ai Claude agent
     │  5-mcp-server.py        │  FastMCP on port 8052
     │  6-chatbot.py           │  Streamlit UI
     └─────────────────────────┘
```

---

## Project structure

```
4-vectorless-agentic-RAG/
│
├── 1-fetch-confluence.py   Fetch Confluence → docs/<SPACE>.md
├── 1-fetch-k8s.py          Fetch K8s public docs → docs/K8S_*.md (no Confluence needed)
├── 2-build-index.py        Build PageIndex trees → workspace/ + indexes/meta.json
├── 3-search.py             Interactive tree explorer (no agent)
├── 4-agent.py              pydantic-ai Claude agent with structured output
├── 5-mcp-server.py         FastMCP server on port 8052
├── 6-chatbot.py            Streamlit chatbot (MCP client)
│
├── utils/
│   ├── confluence.py       Confluence REST client + html_to_markdown()
│   └── agent_tools.py      list_spaces / get_space_structure / read_section
│
├── docs/                   ← created by step 1 (one .md per space)
├── workspace/              ← created by step 2 (PageIndex internal workspace)
├── indexes/                ← created by step 2 (meta.json only — no vectors)
│
├── pyproject.toml
├── .env.example
└── readme.md
```

---

## Setup

### 1. Install PageIndex

PageIndex is not on PyPI. Install from GitHub:

```bash
pip install git+https://github.com/VectifyAI/PageIndex.git
```

Or clone and install in editable mode (recommended for development):

```bash
git clone https://github.com/VectifyAI/PageIndex.git
cd PageIndex
pip install -e .
cd ..
```

### 2. Install project dependencies

```bash
pip install -e .
# or with uv:
uv sync
```

### 3. Copy and fill in `.env`

```bash
cp .env.example .env
```

Required keys:

| Variable | Used by | Purpose |
|----------|---------|---------|
| `CONFLUENCE_BASE_URL` | steps 1, 3 | Your Atlassian instance URL |
| `CONFLUENCE_EMAIL` | steps 1, 3 | Atlassian account email |
| `CONFLUENCE_API_TOKEN` | steps 1, 3 | API token from id.atlassian.com |
| `CONFLUENCE_SPACE_KEYS` | step 1 | Comma-separated space keys to index |
| `OPENAI_API_KEY` | step 2 | PageIndex uses OpenAI to build trees |
| `ANTHROPIC_API_KEY` | steps 4–6 | Claude agent for reasoning and retrieval |

> **No `COHERE_API_KEY` needed.** Vectorless RAG has no reranking step.

---

## Running the pipeline

### Step 1 — Fetch documents

**Option A: Confluence** (requires API token)

```bash
# First run with empty CONFLUENCE_SPACE_KEYS lists available spaces:
uv run 1-fetch-confluence.py

# After setting CONFLUENCE_SPACE_KEYS=ENG,DOCS in .env:
uv run 1-fetch-confluence.py
```

Output: `docs/ENG.md`, `docs/DOCS.md`, etc. — one Markdown file per space,
containing all pages concatenated as `## Page: {title}` sections.

**Option B: Kubernetes public docs** (no API token needed, good for testing)

```bash
uv run 1-fetch-k8s.py
```

Output: `docs/K8S_CONCEPTS.md`, `docs/K8S_TASKS.md`, etc.

### Step 2 — Build PageIndex trees

```bash
uv run 2-build-index.py
```

For each `.md` in `docs/`, PageIndex calls the LLM to analyse the document
structure and produce a hierarchical JSON tree. Results are cached in
`workspace/` — subsequent runs skip already-indexed documents.

**What gets built:**
- `workspace/` — PageIndex internal tree files (managed by the library)
- `indexes/meta.json` — maps `space_key → {doc_id, space_name, page_count}`

**Cost estimate:** roughly 100–200 LLM calls to `gpt-4o-mini` per 50-page space,
which costs approximately $0.01–0.05 per space. Run once; re-running is free
for unchanged documents.

### Step 3 — Explore the tree interactively

```bash
uv run 3-search.py
```

This shows the three-step navigation that the agent uses:
1. `list_spaces()` — see what's indexed
2. `get_space_structure(space_key)` — read the hierarchical tree
3. `read_section(space_key, "120-180")` — fetch content by line range

Unlike `3-hybrid-search.py` in the hybrid-RAG project, there is no query-scoring
step. You navigate the tree manually — the same reasoning the agent does automatically.

### Step 4 — Run the agent

```bash
uv run 4-agent.py "What is our on-call escalation process?"
uv run 4-agent.py "How do we request access to production databases?"
uv run 4-agent.py "What changed in the deployment process last quarter?"
```

The agent:
1. Calls `list_spaces()` to discover indexed knowledge areas
2. Calls `get_space_structure()` to read the document tree
3. Reasons about which sections are relevant
4. Calls `read_section()` with tight line ranges (e.g. `"120-180"`)
5. Follows cross-references if needed
6. Returns a `ConfluenceAnswer` with the answer and citations

### Step 5 — Start the MCP server

```bash
uv run 5-mcp-server.py
```

Starts on `http://0.0.0.0:8052/sse`. Connect from:

**Claude Desktop** (`%APPDATA%\Claude\claude_desktop_config.json` on Windows,
`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "confluence-vectorless": {
      "url": "http://localhost:8052/sse"
    }
  }
}
```

**Claude Code** (`.mcp.json` in repo root):

```json
{
  "mcpServers": {
    "confluence-vectorless": {
      "type": "sse",
      "url": "http://localhost:8052/sse"
    }
  }
}
```

**Cursor** → Settings → MCP → Server URL: `http://localhost:8052/sse`

### Step 6 — Start the chatbot

```bash
# In one terminal:
uv run 5-mcp-server.py

# In another terminal:
uv run streamlit run 6-chatbot.py
```

---

## The three tools

These three functions are in `utils/agent_tools.py` and are shared between
the standalone agent (`4-agent.py`) and the MCP server (`5-mcp-server.py`).

### `list_spaces()`

Reads `indexes/meta.json` and returns the list of indexed knowledge areas.

```json
[
  {"space_key": "ENG", "space_name": "Engineering", "page_count": 47},
  {"space_key": "HR",  "space_name": "Human Resources", "page_count": 12}
]
```

The agent calls this first to know what's available — identical purpose to
`list_spaces()` in the hybrid-RAG project.

### `get_space_structure(space_key)`

Calls `PageIndexClient.get_document_structure(doc_id)` and returns the
full hierarchical tree as JSON. Example structure:

```json
[
  {
    "name": "Engineering Space",
    "description": "Internal engineering documentation",
    "line_num": 1,
    "sub_nodes": [
      {
        "name": "Page: Deployment Process",
        "description": "How to deploy services to production",
        "line_num": 15,
        "sub_nodes": [
          {
            "name": "Prerequisites",
            "description": "Required access and tools before deploying",
            "line_num": 22,
            "sub_nodes": []
          },
          {
            "name": "Rollback procedure",
            "description": "Steps to revert a failed deployment",
            "line_num": 48,
            "sub_nodes": []
          }
        ]
      }
    ]
  }
]
```

The agent reasons over this tree to identify the relevant `line_num` values,
then calls `read_section()` with those lines.

**This replaces `hybrid_search()` from the hybrid-RAG project.** Instead of
a ranked list of similar chunks, the agent gets a map of the entire document
and decides where to look.

### `read_section(space_key, lines)`

Calls `PageIndexClient.get_page_content(doc_id, lines)` and returns the raw
Markdown content for that line range. Examples:

```python
read_section("ENG", "48-75")    # lines 48 to 75
read_section("ENG", "42")       # single line
read_section("ENG", "10,20,35") # specific lines
```

**This replaces `get_page_full()` from the hybrid-RAG project.** Instead of
fetching an entire Confluence page by ID, the agent reads exactly the lines
it identified from the tree — typically a 30–60 line section.

---

## How PageIndex builds the tree

When you call `client.index(md_path)`, PageIndex:

1. **Parses the Markdown structure** — reads headings (`#`, `##`, `###`) to
   create an initial outline
2. **Calls the LLM for each section** — asks the model to describe what the
   section covers and whether it should be split into sub-sections
3. **Builds the recursive tree** — each node gets a `name`, `description`,
   `line_num`, and `sub_nodes` list
4. **Caches in `workspace/`** — subsequent `index()` calls on the same
   filename return the cached `doc_id` immediately

The LLM used during indexing is configured via PageIndex's `config.yaml`
(defaults to `gpt-4o-mini`). The retrieval agent (steps 4–6) uses Claude
independently — you can swap either model without affecting the other.

---

## Document format (what `docs/ENG.md` looks like)

```markdown
<!-- space: ENG | name: Engineering | pages: 47 | fetched: 2024-01-15T10:30:00Z -->

# Engineering Space

> Confluence space **ENG** — 47 pages indexed on 2024-01-15

## Page: Deployment Process
<!-- page_id: 12345 | url: https://... | last_modified: 2024-01-10T08:00:00Z -->

### Overview
All production deployments go through the CI/CD pipeline defined in...

### Prerequisites
You need write access to the `prod-deploy` GitHub environment...

### Rollback procedure
If the healthcheck fails within 5 minutes of deploy...

---

## Page: On-Call Runbook
<!-- page_id: 12346 | url: https://... | last_modified: 2024-01-12T09:00:00Z -->
...
```

Headings within each page are shifted by two levels (`h1→###`, `h2→####`)
so the page's own `## Page:` header is the root of its subtree. This lets
PageIndex build a clean three-level hierarchy:

```
# Space                  (level 1 — the space)
## Page: ...             (level 2 — individual Confluence pages)
### Section heading      (level 3 — headings within the page)
#### Subsection          (level 4)
```

---

## Comparing with hybrid-RAG

Both projects expose the same MCP interface and produce `ConfluenceAnswer`
with citations, so you can run the same question through both and compare.

| Aspect | `1-hybrid-agentic-RAG` | `4-vectorless-agentic-RAG` |
|--------|----------------------|--------------------------|
| Port | 8051 | 8052 |
| Tool 1 | `list_spaces()` | `list_spaces()` |
| Tool 2 | `hybrid_search(query)` | `get_space_structure(space_key)` |
| Tool 3 | `get_page_full(page_id)` | `read_section(space_key, lines)` |
| Output type | `ConfluenceAnswer` | `ConfluenceAnswer` |
| Citation fields | `page_id, title, url, quote` | `space_key, section_title, lines_read, quote` |
| Index files | `indexes/bm25/`, `embeddings.npy`, `meta.json` | `workspace/`, `meta.json` |
| External APIs | OpenAI (embed) + Cohere (rerank) | OpenAI (index build only) |
| Re-index cost | Free (embeddings already stored) | Free (workspace cached) |

Run them simultaneously — they use different ports and separate index
directories, so there is no conflict.

---

## Production considerations

### When to choose this approach over hybrid-RAG

**Choose vectorless when:**
- Documents are long and well-structured (technical specs, runbooks, policies)
- Users ask questions that require following references across sections
- You want to eliminate embedding API costs and Cohere dependencies
- Your content has clear heading hierarchies (PageIndex needs them to build the tree)

**Stay with hybrid-RAG when:**
- You have short, unstructured snippets (tickets, chat logs)
- Query latency matters more than retrieval quality
- Your corpus has millions of documents (PageIndex trees get unwieldy at scale)
- Users ask simple keyword lookups that BM25 handles perfectly

### Keeping the index fresh

Re-run step 1 to fetch updated pages, then step 2 to re-index changed documents.
PageIndex detects unchanged documents by filename — only re-processed files
incur LLM cost.

For continuous sync, schedule a nightly job:

```bash
uv run 1-fetch-confluence.py && uv run 2-build-index.py
```

### Scaling to many spaces

Each Confluence space becomes one `.md` file and one PageIndex document.
With many spaces, the agent's first call to `get_space_structure()` must
pick the right space. If cross-space queries are common, consider:

1. **Space metadata search** — add a fourth tool that searches space names
   and descriptions before calling `get_space_structure()`
2. **Grouped spaces** — merge related small spaces into one `.md` file at
   fetch time so fewer top-level documents need checking
3. **Hybrid first step** — use BM25 on space names and page titles only
   (cheap), then use PageIndex for the selected space's content

### Cost summary

| Step | Model | Cost |
|------|-------|------|
| Build tree (per 50-page space) | gpt-4o-mini | ~$0.01–0.05 |
| Query — list + structure | claude-sonnet-4-6 | ~$0.002 |
| Query — read section (1–2 calls) | claude-sonnet-4-6 | ~$0.005–0.015 |
| **Total per query** | | **~$0.007–0.017** |

Compare with hybrid-RAG: ~$0.003–0.008 per query (cheaper per query, but
requires Cohere API and periodic re-embedding when content changes).
