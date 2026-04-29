"""중국정사외국전 MCP 서버.

도구:
  - search_jung(query, top_k=5, filters=None) → 기사 청크 리스트
  - get_jung_article(chunk_id, with_context=True) → 단일 청크 + 같은 기사/같은 장 인접 기사
  - list_jung_books() → 수록 책 22권 (사기·한서·…)
  - list_jung_countries() → 외국전 대상 국가/종족 목록 (빈도순)
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mcp.server.fastmcp import FastMCP

from jung_retrieval import (
    search,
    get_article,
    list_books as _list_books,
    list_countries as _list_countries,
)

mcp = FastMCP("jung-jeongsa")


@mcp.tool()
def search_jung(
    query: str,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
) -> list[dict]:
    """중국정사외국전(22사 외국전 역주) 하이브리드 검색.

    한자/한글 자유 사용. 엔티티 사전으로 한자↔한글 자동 확장.

    Args:
        query: 자연어 질의 (한자/한글/혼용 OK). 예: "흉노의 풍속", "高句麗"
        top_k: 반환 청크 수 (기본 5)
        filters: payload 필터. 예시:
          - {"book_id": "jo_0001"}             사기만
          - {"book_name": "한서(漢書)"}        한서만
          - {"subject_country": "흉노(匈奴)"}  흉노 관련
          - {"chapter_id": "jo_0001_0110"}     卷110 匈奴列傳
    """
    hits = search(query, top_k=top_k, filters=filters)
    return [asdict(h) for h in hits]


@mcp.tool()
def get_jung_article(chunk_id: str, with_context: bool = True) -> dict | None:
    """chunk_id로 청크 조회 + 같은 기사의 다른 슬라이스 + 같은 chapter의 인접 기사 목록.

    Args:
        chunk_id: search_jung 결과의 chunk_id
        with_context: True면 article_slices(같은 기사) + chapter_siblings(같은 장의 다른 기사) 포함
    """
    return get_article(chunk_id, with_context=with_context)


@mcp.tool()
def list_jung_books() -> list[dict]:
    """수록된 22사 외국전 책 목록 (book_id, 책 이름, 저자, 시대)."""
    return _list_books()


@mcp.tool()
def list_jung_countries() -> list[dict]:
    """외국전 대상 국가/종족 목록 (빈도순). 약 690개."""
    return _list_countries()


if __name__ == "__main__":
    mcp.run()
