"""한국사료총서 MCP 서버.

도구:
  - search_korean_history(query, top_k=5, filters=None) → 청크 리스트
  - get_passage(chunk_id, with_context=True) → 단일 청크 + 인접 단락
  - list_volumes() → 121권 메타 목록
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

from mcp.server.fastmcp import FastMCP

from retrieval import search, get_passage, list_volumes

mcp = FastMCP("kuksa-history")


@mcp.tool()
def search_korean_history(
    query: str,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
) -> list[dict]:
    """한국사료총서 121권 하이브리드 검색 (BGE-M3 dense+sparse + reranker).

    한자/한글 자유롭게 사용 가능. 엔티티 사전으로 한자↔한글 후보 자동 확장.

    Args:
        query: 자연어 질의 (한자/한글/혼용 OK)
        top_k: 반환 청크 수 (기본 5)
        filters: payload 필터. 예: {"volume_id": "sa_003"}, {"subject_class": "집부(集部)_별집류(別集類)"}
    """
    hits = search(query, top_k=top_k, filters=filters)
    return [asdict(h) for h in hits]


@mcp.tool()
def get_chunk_with_context(chunk_id: str, with_context: bool = True) -> dict | None:
    """청크 ID로 단일 청크 + 같은 level2 인접 단락 반환.

    Args:
        chunk_id: search 결과의 chunk_id
        with_context: True면 같은 권/장 안의 인접 청크 최대 20개 추가
    """
    return get_passage(chunk_id, with_context=with_context)


@mcp.tool()
def list_kuksa_volumes() -> list[dict]:
    """수록된 121권 메타데이터 목록 (volume_id, 제목, 저자, 시대, 분류)."""
    return list_volumes()


if __name__ == "__main__":
    mcp.run()
