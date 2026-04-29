"""승정원일기 MCP 서버 (seungjeongwon)."""
from __future__ import annotations
from dataclasses import asdict
from typing import Any
from mcp.server.fastmcp import FastMCP
from sjw_retrieval import search, get_article, list_kings as _list_kings

mcp = FastMCP("seungjeongwon")


@mcp.tool()
def search_sjw(query: str, top_k: int = 5, filters: dict[str, Any] | None = None) -> list[dict]:
    """승정원일기 (承政院日記) 하이브리드 검색.

    내부적으로 두 collection (sjw=인조~정조, sjw2=순조~순종) 동시 검색 후 reranker
    로 글로벌 재순위. 응답의 source_collection 필드로 출처 확인 가능.

    필터 예: {"king": "인조"}, {"king_prefix": "A"}, {"year_ce": "1623"}, {"date_western": "1623-03-12"}
    """
    return [asdict(h) for h in search(query, top_k=top_k, filters=filters)]


@mcp.tool()
def get_sjw_article(chunk_id: str, with_context: bool = True) -> dict | None:
    """단일 청크 + 같은 기사의 다른 슬라이스."""
    return get_article(chunk_id, with_context=with_context)


@mcp.tool()
def list_sjw_kings() -> list[dict]:
    """수록된 왕 목록 (인조~순종 12명)."""
    return _list_kings()


if __name__ == "__main__":
    mcp.run()
