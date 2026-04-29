"""중국정사외국전 하이브리드 검색.

질의 전처리 → BGE-M3 dense+sparse → Qdrant 하이브리드 → BGE-reranker-v2-m3 재순위.
엔티티 사전(data/중국정사외국전/entities.json)으로 한자↔한글 후보 확장.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

from FlagEmbedding import BGEM3FlagModel, FlagReranker
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from hanja_util import has_hangul, has_hanja, to_hangul, duum_variants

ROOT = Path(__file__).resolve().parents[1]
QDRANT_PATH = ROOT / "data" / "중국정사외국전" / "qdrant_storage"
ENTITIES = ROOT / "data" / "중국정사외국전" / "entities.json"
COLLECTION = "jung_jeongsa"

DEVICE = "cuda"

DENSE_TOPK = 50
SPARSE_TOPK = 50
FUSE_TOPK = 60
DEFAULT_TOPK = 5


@dataclass
class JungHit:
    chunk_id: str
    article_id: str
    score: float
    book_id: str
    book_name: str
    book_title_hanja: str
    book_author: str
    chapter_id: str
    chapter_title: str
    article_title: str
    subject_country: str
    text: str
    text_hanmun: str
    text_korean: str
    has_footnotes_korean: int
    has_footnotes_hanmun: int
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
            inv: dict[str, set[str]] = {}
            for cat, terms in _state["ent"].items():
                for hanja_s, hanguls in terms.items():
                    for hg in hanguls:
                        inv.setdefault(hg, set()).add(hanja_s)
            _state["ent_inv"] = inv


def expand_query(q: str, max_expansions: int = 30) -> str:
    _ensure_loaded()
    inv: dict[str, set[str]] = _state["ent_inv"]

    parts = [q]
    if has_hanja(q):
        parts.append(to_hangul(q))

    if has_hangul(q):
        import regex as re
        tokens = re.findall(r"[\p{Hangul}]{2,}", q)
        added = 0
        for tok in tokens:
            cands: set[str] = set()
            if tok in inv:
                cands |= inv[tok]
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


def _to_hit(p: qm.ScoredPoint) -> JungHit:
    pl = p.payload
    return JungHit(
        chunk_id=pl["chunk_id"],
        article_id=pl["article_id"],
        score=float(p.score),
        book_id=pl["book_id"],
        book_name=pl["book_name"],
        book_title_hanja=pl["book_title_hanja"],
        book_author=pl["book_author"],
        chapter_id=pl["chapter_id"],
        chapter_title=pl["chapter_title"],
        article_title=pl["article_title"],
        subject_country=pl.get("subject_country", ""),
        text=pl["text"],
        text_hanmun=pl.get("text_hanmun", ""),
        text_korean=pl.get("text_korean", ""),
        has_footnotes_korean=pl.get("has_footnotes_korean", 0),
        has_footnotes_hanmun=pl.get("has_footnotes_hanmun", 0),
        entities=pl.get("entities", {}),
    )


def search(query: str, top_k: int = DEFAULT_TOPK, filters: dict | None = None) -> list[JungHit]:
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
    return [_to_hit(p) for p in reranked]


def get_article(chunk_id: str, with_context: bool = True) -> dict | None:
    """chunk_id로 청크 조회 + 같은 article_id 의 다른 슬라이스 + 같은 chapter 의 인접 기사."""
    _ensure_loaded()
    client: QdrantClient = _state["qdrant"]
    res, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=qm.Filter(
            must=[qm.FieldCondition(key="chunk_id", match=qm.MatchValue(value=chunk_id))]
        ),
        limit=1,
        with_payload=True,
    )
    if not res:
        return None
    p = res[0]
    out: dict = {"target": p.payload}

    if with_context:
        article_id = p.payload["article_id"]
        chapter_id = p.payload.get("chapter_id", "")
        # 같은 기사 내 다른 슬라이스
        ares, _ = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=qm.Filter(
                must=[qm.FieldCondition(key="article_id", match=qm.MatchValue(value=article_id))]
            ),
            limit=50,
            with_payload=True,
        )
        slices = sorted(
            [pt.payload for pt in ares if pt.payload["chunk_id"] != chunk_id],
            key=lambda x: x["chunk_id"],
        )
        out["article_slices"] = slices
        # 같은 장(chapter)의 인접 기사 (article_id 단위로 dedup)
        if chapter_id:
            cres, _ = client.scroll(
                collection_name=COLLECTION,
                scroll_filter=qm.Filter(
                    must=[qm.FieldCondition(key="chapter_id", match=qm.MatchValue(value=chapter_id))]
                ),
                limit=200,
                with_payload=True,
            )
            seen: set[str] = set()
            siblings: list[dict] = []
            for pt in sorted(cres, key=lambda x: x.payload["chunk_id"]):
                aid = pt.payload["article_id"]
                if aid == article_id or aid in seen:
                    continue
                seen.add(aid)
                siblings.append({
                    "article_id": aid,
                    "article_title": pt.payload.get("article_title", ""),
                    "subject_country": pt.payload.get("subject_country", ""),
                })
            out["chapter_siblings"] = siblings[:30]
    return out


@lru_cache(maxsize=1)
def list_books() -> list[dict]:
    """수록된 책 목록 (book_id 단위, 22권)."""
    _ensure_loaded()
    client: QdrantClient = _state["qdrant"]
    seen: dict[str, dict] = {}
    next_page = None
    for _ in range(50):
        res, next_page = client.scroll(
            collection_name=COLLECTION,
            limit=2000,
            offset=next_page,
            with_payload=True,
        )
        for p in res:
            bid = p.payload["book_id"]
            if bid not in seen:
                seen[bid] = {
                    "book_id": bid,
                    "book_name": p.payload["book_name"],
                    "book_title_hanja": p.payload["book_title_hanja"],
                    "book_author": p.payload.get("book_author", ""),
                    "period_label": p.payload.get("period_label", ""),
                }
        if not next_page:
            break
    return sorted(seen.values(), key=lambda x: x["book_id"])


@lru_cache(maxsize=1)
def list_countries() -> list[dict]:
    """수록된 subject_country 목록 (출현 청크 수 포함)."""
    _ensure_loaded()
    client: QdrantClient = _state["qdrant"]
    counts: dict[str, int] = {}
    next_page = None
    for _ in range(50):
        res, next_page = client.scroll(
            collection_name=COLLECTION,
            limit=2000,
            offset=next_page,
            with_payload=True,
        )
        for p in res:
            sc = p.payload.get("subject_country") or ""
            if sc:
                counts[sc] = counts.get(sc, 0) + 1
        if not next_page:
            break
    return sorted(
        [{"country": k, "chunk_count": v} for k, v in counts.items()],
        key=lambda x: -x["chunk_count"],
    )


if __name__ == "__main__":
    import sys as _sys
    q = " ".join(_sys.argv[1:]) or "흉노의 풍속과 생활"
    print(f"[query] {q}")
    print(f"[expanded] {expand_query(q)}\n")
    hits = search(q, top_k=5)
    for i, h in enumerate(hits, 1):
        print(f"--- {i}. score={h.score:.4f} ---")
        print(f"  {h.book_name} > {h.chapter_title} > {h.article_title}  [{h.chunk_id}]")
        print(f"  국가: {h.subject_country}")
        print(f"  {h.text[:200]}")
