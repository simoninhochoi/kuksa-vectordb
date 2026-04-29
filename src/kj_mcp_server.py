"""고려사절요 MCP 서버 (goryeosa-jeolyo)."""
from __future__ import annotations
from dataclasses import asdict
from typing import Any
from mcp.server.fastmcp import FastMCP
from kj_retrieval import search, get_article, list_volumes as _list_volumes

mcp = FastMCP("goryeosa-jeolyo")


@mcp.tool()
def search_kj(query: str, top_k: int = 5, filters: dict[str, Any] | None = None) -> list[dict]:
    """고려사절요 (高麗史節要) 하이브리드 검색.

    필터 예: {"volume_id": "kj_001"}, {"date_western": "0918-06-15"}, {"ganji": "병진(丙辰)"}
    """
    return [asdict(h) for h in search(query, top_k=top_k, filters=filters)]


@mcp.tool()
def get_kj_article(chunk_id: str, with_context: bool = True) -> dict | None:
    """단일 청크 + 같은 기사의 다른 슬라이스."""
    return get_article(chunk_id, with_context=with_context)


@mcp.tool()
def list_kj_volumes() -> list[dict]:
    """수록된 권 목록."""
    return _list_volumes()


if __name__ == "__main__":
    mcp.run()
