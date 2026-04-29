"""비변사등록 MCP 서버 (bibyeonsa)."""
from __future__ import annotations
from dataclasses import asdict
from typing import Any
from mcp.server.fastmcp import FastMCP
from bb_retrieval import search, get_article, list_volumes as _list_volumes

mcp = FastMCP("bibyeonsa")


@mcp.tool()
def search_bb(query: str, top_k: int = 5, filters: dict[str, Any] | None = None) -> list[dict]:
    """비변사등록 (備邊司謄錄) 하이브리드 검색.

    필터 예: {"volume_id": "bb_001"}, {"king": "광해군"}, {"year_ce": "1617"}, {"month_value": "01"}
    """
    return [asdict(h) for h in search(query, top_k=top_k, filters=filters)]


@mcp.tool()
def get_bb_article(chunk_id: str, with_context: bool = True) -> dict | None:
    """단일 청크 + 같은 기사의 다른 슬라이스."""
    return get_article(chunk_id, with_context=with_context)


@mcp.tool()
def list_bb_volumes() -> list[dict]:
    """수록된 권 목록."""
    return _list_volumes()


if __name__ == "__main__":
    mcp.run()
