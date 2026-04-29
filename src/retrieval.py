"""한국사료총서 하이브리드 검색.

질의 전처리 → BGE-M3 dense+sparse → Qdrant 하이브리드 → BGE-reranker-v2-m3 재순위.

엔티티 사전(entities.json)을 이용해 한자↔한글 후보 확장.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from FlagEmbedding import BGEM3FlagModel, FlagReranker
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from hanja_util import has_hangul, has_hanja, script_of, to_hangul, duum_variants

ROOT = Path(__file__).resolve().parents[1]
QDRANT_PATH = ROOT / "data" / "qdrant_storage"
ENTITIES = ROOT / "data" / "entities.json"
COLLECTION = "korean_history"

DEVICE = "cuda"

# 검색 파라미터
DENSE_TOPK = 50
SPARSE_TOPK = 50
FUSE_TOPK = 60   # 두 결과 합산 후 reranker 입력 후보 수
DEFAULT_TOPK = 5


@dataclass
class SearchHit:
    chunk_id: str
    score: float
    volume_id: str
    volume_title_hanja: str
    volume_title_hangul: str
    author: str
    period_begin: str
    level1_title: str
    level2_title: str
    level3_title: str | None
    text: str
    text_hangul_aux: str
    entities: dict[str, list[str]] = field(default_factory=dict)


_lock = threading.Lock()
_state: dict = {}


def _ensure_loaded() -> None:
    with _lock:
        if "embed" not in _state:
            _state["embed"] = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=DEVICE)
        if "rerank" not in _state:
            _state["rerank"] = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, device=DEVICE)
        if "qdrant" not in _state:
            _state["qdrant"] = QdrantClient(path=str(QDRANT_PATH))
        if "ent" not in _state:
            with ENTITIES.open(encoding="utf-8") as f:
                _state["ent"] = json.load(f)
            # 역인덱스: 한글 → set[한자] (어떤 카테고리든)
            inv: dict[str, set[str]] = {}
            for cat, terms in _state["ent"].items():
                for hanja_s, hanguls in terms.items():
                    for hg in hanguls:
                        inv.setdefault(hg, set()).add(hanja_s)
            _state["ent_inv"] = inv


def expand_query(q: str, max_expansions: int = 30) -> str:
    """한자↔한글 후보 확장. 원본 + 가능한 다른 표기 + 음차본 병기.

    1. 한자 토큰은 그대로 + 한글 음차 병기
    2. 한글 토큰은 엔티티 사전에서 한자 후보 룩업해 추가
    """
    _ensure_loaded()
    inv: dict[str, set[str]] = _state["ent_inv"]

    parts = [q]

    # 1) 한자 음차본 추가
    if has_hanja(q):
        parts.append(to_hangul(q))

    # 2) 한글 → 한자 후보 (엔티티 사전 기반)
    if has_hangul(q):
        # 단순한 토크나이즈: 공백·구두점 분리
        import regex as re
        tokens = re.findall(r"[\p{Hangul}]{2,}", q)
        added = 0
        for tok in tokens:
            cands: set[str] = set()
            # 직접 매칭
            if tok in inv:
                cands |= inv[tok]
            # 두음변형도 시도
            for variant in duum_variants(tok):
                if variant in inv:
                    cands |= inv[variant]
            for c in sorted(cands):
                if added >= max_expansions:
                    break
                parts.append(c)
                added += 1
            if added >= max_expansions:
                break

    return " ".join(parts)


def _hybrid_search(
    client: QdrantClient,
    dense_vec: list[float],
    sparse_vec: dict[int, float],
    limit: int,
    qfilter: qm.Filter | None = None,
) -> list[qm.ScoredPoint]:
    """RRF로 dense+sparse 융합."""
    sparse = qm.SparseVector(
        indices=[int(k) for k in sparse_vec.keys()],
        values=[float(v) for v in sparse_vec.values()],
    )
    res = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            qm.Prefetch(query=dense_vec, using="dense", limit=DENSE_TOPK, filter=qfilter),
            qm.Prefetch(query=sparse, using="sparse", limit=SPARSE_TOPK, filter=qfilter),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=limit,
        query_filter=qfilter,
        with_payload=True,
    )
    return res.points


def _rerank(query: str, hits: list[qm.ScoredPoint], top_k: int) -> list[qm.ScoredPoint]:
    """BGE-reranker-v2-m3 cross-encoder 재순위."""
    if not hits:
        return []
    pairs = [[query, h.payload["text"]] for h in hits]
    scores = _state["rerank"].compute_score(pairs, normalize=True)
    if not isinstance(scores, list):
        scores = [scores]
    paired = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
    out: list[qm.ScoredPoint] = []
    for score, h in paired[:top_k]:
        h.score = float(score)
        out.append(h)
    return out


def _make_filter(filters: dict | None) -> qm.Filter | None:
    if not filters:
        return None
    must: list[qm.FieldCondition] = []
    for k, v in filters.items():
        if isinstance(v, list):
            must.append(qm.FieldCondition(key=k, match=qm.MatchAny(any=v)))
        else:
            must.append(qm.FieldCondition(key=k, match=qm.MatchValue(value=v)))
    return qm.Filter(must=must) if must else None


def search(query: str, top_k: int = DEFAULT_TOPK, filters: dict | None = None) -> list[SearchHit]:
    """질의 → 확장 → 하이브리드 → 재순위 → top_k."""
    _ensure_loaded()
    expanded = expand_query(query)

    out = _state["embed"].encode(
        [expanded],
        max_length=512,
        return_dense=True,
        return_sparse=True,
    )
    dense_vec = out["dense_vecs"][0].tolist()
    sparse_vec = out["lexical_weights"][0]

    qfilter = _make_filter(filters)
    fused = _hybrid_search(_state["qdrant"], dense_vec, sparse_vec, FUSE_TOPK, qfilter)
    reranked = _rerank(query, fused, top_k)

    return [
        SearchHit(
            chunk_id=p.payload["chunk_id"],
            score=p.score,
            volume_id=p.payload["volume_id"],
            volume_title_hanja=p.payload["volume_title_hanja"],
            volume_title_hangul=p.payload["volume_title_hangul"],
            author=p.payload["author"],
            period_begin=p.payload["period_begin"],
            level1_title=p.payload["level1_title"],
            level2_title=p.payload["level2_title"],
            level3_title=p.payload.get("level3_title"),
            text=p.payload["text"],
            text_hangul_aux=p.payload["text_hangul_aux"],
            entities=p.payload.get("entities", {}),
        )
        for p in reranked
    ]


def get_passage(chunk_id: str, with_context: bool = True) -> dict | None:
    """chunk_id로 단일 청크 + 같은 level2 인접 단락 반환."""
    _ensure_loaded()
    client: QdrantClient = _state["qdrant"]
    res = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=qm.Filter(
            must=[qm.FieldCondition(key="chunk_id", match=qm.MatchValue(value=chunk_id))]
        ),
        limit=1,
        with_payload=True,
    )
    pts, _ = res
    if not pts:
        return None
    p = pts[0]
    out = {"target": p.payload}

    if with_context:
        # 같은 volume_id + level1_title + level2_title 의 모든 청크 조회
        volume_id = p.payload["volume_id"]
        l2_title = p.payload["level2_title"]
        ctx_res = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=qm.Filter(
                must=[
                    qm.FieldCondition(key="volume_id", match=qm.MatchValue(value=volume_id)),
                ]
            ),
            limit=200,
            with_payload=True,
        )
        ctx_pts, _ = ctx_res
        sibling = [
            pt.payload for pt in ctx_pts
            if pt.payload.get("level2_title") == l2_title and pt.payload["chunk_id"] != chunk_id
        ]
        # chunk_id 기준 정렬
        sibling.sort(key=lambda x: x["chunk_id"])
        out["siblings"] = sibling[:20]
    return out


@lru_cache(maxsize=1)
def list_volumes() -> list[dict]:
    """121권 메타 목록."""
    _ensure_loaded()
    client: QdrantClient = _state["qdrant"]
    res, _ = client.scroll(
        collection_name=COLLECTION,
        limit=10_000,  # 첫 페이지에 모든 권의 첫 청크가 들어옴 직쩌
        with_payload=True,
    )
    seen: dict[str, dict] = {}
    for p in res:
        vid = p.payload["volume_id"]
        if vid not in seen:
            seen[vid] = {
                "volume_id": vid,
                "title_hanja": p.payload["volume_title_hanja"],
                "title_hangul": p.payload["volume_title_hangul"],
                "author": p.payload["author"],
                "period_begin": p.payload["period_begin"],
                "period_end": p.payload["period_end"],
                "subject_class": p.payload["subject_class"],
            }
    return sorted(seen.values(), key=lambda x: x["volume_id"])


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "이기의 토지개혁 사상"
    print(f"[query] {q}")
    print(f"[expanded] {expand_query(q)}\n")
    hits = search(q, top_k=5)
    for i, h in enumerate(hits, 1):
        print(f"--- {i}. score={h.score:.4f} ---")
        print(f"  {h.volume_title_hanja}({h.volume_title_hangul}) > {h.level2_title}  [{h.volume_id}]")
        print(f"  {h.text[:200]}")
