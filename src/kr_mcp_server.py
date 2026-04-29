"""고려사 MCP 서버 (goryeosa)."""
from __future__ import annotations
from dataclasses import asdict
from typing import Any
from mcp.server.fastmcp import FastMCP
from kr_retrieval import search, get_article, list_volumes as _list_volumes

mcp = FastMCP("goryeosa")


@mcp.tool()
def search_kr(query: str, top_k: int = 5, filters: dict[str, Any] | None = None) -> list[dict]:
    """고려사 (高麗史) 하이브리드 검색.

    필터 예: {"volume_id": "kr_001"}, {"section_type": "世家"}, {"date_western": "0918-06-15"}, {"ganji": "병진(丙辰)"}
    """
    return [asdict(h) for h in search(query, top_k=top_k, filters=filters)]


@mcp.tool()
def get_kr_article(chunk_id: str, with_context: bool = True) -> dict | None:
    """단일 청크 + 같은 기사의 다른 슬라이스."""
    return get_article(chunk_id, with_context=with_context)


@mcp.tool()
def list_kr_volumes() -> list[dict]:
    """수록된 권 목록."""
    return _list_volumes()


if __name__ == "__main__":
    mcp.run()
