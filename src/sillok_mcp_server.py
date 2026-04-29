"""조선왕조실록 MCP 서버.

도구:
  - search_sillok(query, top_k=5, filters=None) → 기사 청크 리스트
  - get_sillok_article(chunk_id, with_context=True) → 단일 청크 + 같은 기사/같은 날 기사
  - list_sillok() → 실록 목록 (태조실록~고종실록)
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mcp.server.fastmcp import FastMCP

from sillok_retrieval import search, get_article, list_sillok as _list_sillok

mcp = FastMCP("sillok")


@mcp.tool()
def search_sillok(
    query: str,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
) -> list[dict]:
    """조선왕조실록 하이브리드 검색 (BGE-M3 dense+sparse + reranker).

    한자/한글 자유 사용. 엔티티 사전으로 한자↔한글 자동 확장.

    Args:
        query: 자연어 질의 (한자/한글/혼용 OK)
        top_k: 반환 청크 수 (기본 5)
        filters: payload 필터. 예:
          - {"sillok_id": "waa"}          태조실록만
          - {"king": "世宗"}               세종 연간
          - {"year_ce": "1392"}            1392년
          - {"date_western": "1392-07-17"} 특정 날짜
          - {"subject_classes": "왕실(王室)"}
    """
    hits = search(query, top_k=top_k, filters=filters)
    return [asdict(h) for h in hits]


@mcp.tool()
def get_sillok_article(chunk_id: str, with_context: bool = True) -> dict | None:
    """chunk_id로 청크 조회 + 같은 기사의 다른 슬라이스 + 같은 날의 인접 기사.

    Args:
        chunk_id: search_sillok 결과의 chunk_id
        with_context: True면 article_slices(같은 기사) + day_siblings(같은 날 기사) 포함
    """
    return get_article(chunk_id, with_context=with_context)


@mcp.tool()
def list_sillok_volumes() -> list[dict]:
    """수록된 실록 목록 (sillok_id, 왕, 제목 한자/한글)."""
    return _list_sillok()


if __name__ == "__main__":
    mcp.run()
