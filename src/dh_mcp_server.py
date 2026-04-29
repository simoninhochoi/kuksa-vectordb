"""동문휘고 MCP 서버."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mcp.server.fastmcp import FastMCP

from dh_retrieval import search, get_document, list_volumes as _list_volumes

mcp = FastMCP("dongmun-hwigo")


@mcp.tool()
def search_dongmun_hwigo(
    query: str,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
) -> list[dict]:
    """동문휘고 외교문서 하이브리드 검색 (BGE-M3 + reranker, 한자↔한글 자동 확장).

    Args:
        query: 자연어 질의 (한자/한글 OK)
        top_k: 반환 청크 수
        filters: payload 필터 예:
          - {"volume_num": "0001"}        특정 권
          - {"sender": "朝鮮國王"}          발신자
          - {"receiver": "禮部"}           수신자
          - {"date_western": "1645-08-20"} 특정 날짜
          - {"hanja_kind": "m"}            표점본만
    """
    hits = search(query, top_k=top_k, filters=filters)
    return [asdict(h) for h in hits]


@mcp.tool()
def get_dongmun_document(chunk_id: str, with_context: bool = True) -> dict | None:
    """chunk_id로 외교문서 1건 조회 + 같은 level3의 다른 슬라이스."""
    return get_document(chunk_id, with_context=with_context)


@mcp.tool()
def list_dongmun_volumes() -> list[dict]:
    """수록 권 목록 (volume_num, volume_name)."""
    return _list_volumes()


if __name__ == "__main__":
    mcp.run()
