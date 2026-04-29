"""원고려기사 MCP 서버 (wongoryeo)."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mcp.server.fastmcp import FastMCP
from cnwk_retrieval import search, get_article, list_emperors as _list_emperors

mcp = FastMCP("wongoryeo")


@mcp.tool()
def search_wongoryeo(query: str, top_k: int = 5, filters: dict[str, Any] | None = None) -> list[dict]:
    """원고려기사 (元代 사료에 보이는 高麗 관련 기사) 하이브리드 검색.

    필터 예: {"level1_id": "cnwk_006"} (世祖皇帝), {"date_western": "1218-99-99"}
    """
    return [asdict(h) for h in search(query, top_k=top_k, filters=filters)]


@mcp.tool()
def get_wongoryeo_article(chunk_id: str, with_context: bool = True) -> dict | None:
    """단일 청크 + 같은 기사 슬라이스 + 같은 연도(level2)의 인접 기사."""
    return get_article(chunk_id, with_context=with_context)


@mcp.tool()
def list_wongoryeo_emperors() -> list[dict]:
    """level1 단위 (序文/태조/태종/정종/헌종/세조/성종/탐라/附記)."""
    return _list_emperors()


if __name__ == "__main__":
    mcp.run()
