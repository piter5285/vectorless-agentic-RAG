"""
Confluence Cloud REST API client.

Uses the v1 REST API (wiki/rest/api) which is the most widely supported
path for Confluence Cloud. Authentication is email + API token via Basic Auth.

Generate an API token at: https://id.atlassian.com/manage-profile/security/api-tokens

Key difference from hybrid-RAG: we convert pages to Markdown (not plain text
and not chunks) so that PageIndex can build a hierarchical tree over the
natural document structure.
"""

import os
import re
from html import unescape

import requests
from requests.auth import HTTPBasicAuth


class ConfluenceClient:
    def __init__(self) -> None:
        base = os.environ["CONFLUENCE_BASE_URL"].rstrip("/")
        self._api = f"{base}/wiki/rest/api"
        self._base = base
        self._auth = HTTPBasicAuth(
            os.environ["CONFLUENCE_EMAIL"],
            os.environ["CONFLUENCE_API_TOKEN"],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_spaces(self) -> list[dict]:
        """Return all global spaces the token can access."""
        results, start = [], 0
        while True:
            data = self._get("/space", params={"start": start, "limit": 50, "type": "global"})
            results.extend(data["results"])
            if data.get("size", 0) < 50:
                break
            start += 50
        return results

    def iter_pages(self, space_key: str):
        """Yield raw page dicts for every page in space_key with body.storage."""
        start = 0
        while True:
            data = self._get(
                "/content",
                params={
                    "spaceKey": space_key,
                    "type": "page",
                    "expand": "body.storage,version,space",
                    "start": start,
                    "limit": 50,
                },
            )
            yield from data["results"]
            if data.get("size", 0) < 50:
                break
            start += 50

    # ------------------------------------------------------------------
    # Text processing
    # ------------------------------------------------------------------

    @staticmethod
    def html_to_markdown(html: str, heading_offset: int = 2) -> str:
        """
        Convert Confluence storage-format HTML/XHTML to Markdown.

        heading_offset: shift all headings down by this many levels so they
        nest correctly under the page's own ## heading in the space document.
        e.g. offset=2 → h1 becomes ###, h2 becomes ####, h3 becomes #####.

        Preserves:
          - Headings (shifted by heading_offset, clamped at ######)
          - Ordered and unordered lists
          - Code blocks and inline code
          - Bold and italic
          - Paragraphs and line breaks
        Strips:
          - Confluence namespace tags (ac:*, ri:*)
          - Scripts, styles, and metadata noise
        """
        # Strip Confluence-specific namespace tags (keep inner content)
        html = re.sub(r"</?ac:[^>]+>", " ", html)
        html = re.sub(r"</?ri:[^>]+>", " ", html)

        # Headings — shift by offset, clamp to h6
        for level in range(6, 0, -1):
            new_level = min(level + heading_offset, 6)
            prefix = "#" * new_level
            html = re.sub(
                rf"<h{level}(\s[^>]*)?>",
                f"\n{prefix} ",
                html,
                flags=re.IGNORECASE,
            )
            html = re.sub(rf"</h{level}>", "\n", html, flags=re.IGNORECASE)

        # Code blocks (must come before inline code)
        html = re.sub(r"<pre[^>]*>", "\n```\n", html, flags=re.IGNORECASE)
        html = re.sub(r"</pre>", "\n```\n", html, flags=re.IGNORECASE)

        # Inline code
        html = re.sub(r"<code[^>]*>", "`", html, flags=re.IGNORECASE)
        html = re.sub(r"</code>", "`", html, flags=re.IGNORECASE)

        # Bold / strong
        html = re.sub(r"<(?:strong|b)(\s[^>]*)?>", "**", html, flags=re.IGNORECASE)
        html = re.sub(r"</(?:strong|b)>", "**", html, flags=re.IGNORECASE)

        # Italic / emphasis
        html = re.sub(r"<(?:em|i)(\s[^>]*)?>", "_", html, flags=re.IGNORECASE)
        html = re.sub(r"</(?:em|i)>", "_", html, flags=re.IGNORECASE)

        # Ordered list items
        html = re.sub(r"<li[^>]*>", "\n- ", html, flags=re.IGNORECASE)

        # Block elements → newline
        html = re.sub(
            r"<(?:p|br|div|tr|th|td|ul|ol|li|blockquote)(\s[^>]*)?>",
            "\n",
            html,
            flags=re.IGNORECASE,
        )

        # Strip all remaining tags
        html = re.sub(r"<[^>]+>", "", html)

        # Decode HTML entities
        html = unescape(html)

        # Normalise whitespace
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in html.splitlines()]
        # Collapse runs of 3+ blank lines to 2 blank lines
        result = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
        return result.strip()

    @staticmethod
    def html_to_text(html: str) -> str:
        """Convert Confluence storage-format HTML to plain text (no markdown)."""
        text = re.sub(
            r"<(p|h[1-6]|li|br|div|tr|td)(\s[^>]*)?>",
            "\n",
            html,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"<[^>]+>", "", text)
        text = unescape(text)
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
        return "\n".join(ln for ln in lines if ln)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict = None) -> dict:
        resp = requests.get(
            f"{self._api}{path}",
            auth=self._auth,
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def page_url(self, space_key: str, page_id: str) -> str:
        return f"{self._base}/wiki/spaces/{space_key}/pages/{page_id}"
