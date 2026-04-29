"""조선왕조실록 하이브리드 검색.

질의 전처리 → BGE-M3 dense+sparse → Qdrant 하이브리드 → BGE-reranker-v2-m3 재순위.
엔티티 사전(data/조선왕조실록/entities.json)으로 한자↔한글 후보 확장.
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

from hanja_util import has_hangul, has_hanja, to_hangul, duum_variants

ROOT = Path(__file__).resolve().parents[1]
QDRANT_PATH = ROOT / "data" / "조선왕조실록" / "qdrant_storage"
ENTITIES = ROOT / "data" / "조선왕조실록" / "entities.json"
COLLECTION = "sillok"

DEVICE = "cuda"

DENSE_TOPK = 50
SPARSE_TOPK = 50
FUSE_TOPK = 60
DEFAULT_TOPK = 5


@dataclass
class SillokHit:
    chunk_id: str
    article_id: str
    score: float
    sillok_title_hanja: str
    sillok_title_hangul: str
    king: str
    volume_title: str
    year_ce: str
    day_title: str
    date_western: str
    ganji: str
    reign_year: str
    china_era: str
    article_title: str
    subject_classes: list[str]
    source_refs: list[str]
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
            inv: dict[str, set[str]] = {}
            for cat, terms in _state["ent"].items():
                for hanja_s, hanguls in terms.items():
                    for hg in hanguls:
                        inv.setdefault(hg, set()).add(hanja_s)
            _state["ent_inv"] = inv


def expand_query(q: str, max_expansions: int = 30) -> str:
    """한자↔한글 후보 확장. 한국사료총서 retrieval.expand_query와 동일 로직."""
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


def _to_hit(p: qm.ScoredPoint) -> SillokHit:
    pl = p.payload
    return SillokHit(
        chunk_id=pl["chunk_id"],
        article_id=pl["article_id"],
        score=float(p.score),
        sillok_title_hanja=pl["sillok_title_hanja"],
        sillok_title_hangul=pl["sillok_title_hangul"],
        king=pl["king"],
        volume_title=pl["volume_title"],
        year_ce=pl["year_ce"],
        day_title=pl["day_title"],
        date_western=pl["date_western"],
        ganji=pl["ganji"],
        reign_year=pl["reign_year"],
        china_era=pl["china_era"],
        article_title=pl["article_title"],
        subject_classes=pl.get("subject_classes", []),
        source_refs=pl.get("source_refs", []),
        text=pl["text"],
        text_hangul_aux=pl["text_hangul_aux"],
        entities=pl.get("entities", {}),
    )


def search(query: str, top_k: int = DEFAULT_TOPK, filters: dict | None = None) -> list[SillokHit]:
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
    """chunk_id로 단일 청크 + 같은 article_id의 sibling 슬라이스 반환."""
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
        day_title = p.payload.get("day_title", "")
        volume_id = p.payload.get("volume_id", "")
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
        # 같은 날(day_title)의 인접 기사
        if day_title and volume_id:
            dres, _ = client.scroll(
                collection_name=COLLECTION,
                scroll_filter=qm.Filter(
                    must=[
                        qm.FieldCondition(key="volume_id", match=qm.MatchValue(value=volume_id)),
                        qm.FieldCondition(key="day_title", match=qm.MatchValue(value=day_title)),
                    ]
                ),
                limit=50,
                with_payload=True,
            )
            # 다른 article_id만, chunk_id별 dedup하지 말고 article_id 첫 청크만
            seen_aids: set[str] = set()
            siblings: list[dict] = []
            for pt in sorted(dres, key=lambda x: x.payload["chunk_id"]):
                aid = pt.payload["article_id"]
                if aid == article_id or aid in seen_aids:
                    continue
                seen_aids.add(aid)
                siblings.append(pt.payload)
            out["day_siblings"] = siblings[:20]
    return out


@lru_cache(maxsize=1)
def list_sillok() -> list[dict]:
    """수록된 실록 목록 (sillok_id 단위로 요약)."""
    _ensure_loaded()
    client: QdrantClient = _state["qdrant"]
    # 실록 당 수천~수만 청크. 첫 스크롤로 다 잡히지는 않지만
    # 초기에 등장하는 청크로 고유 sillok_id를 수집.
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
            sid = p.payload["sillok_id"]
            if sid not in seen:
                seen[sid] = {
                    "sillok_id": sid,
                    "king": p.payload["king"],
                    "title_hanja": p.payload["sillok_title_hanja"],
                    "title_hangul": p.payload["sillok_title_hangul"],
                }
        if not next_page:
            break
    return sorted(seen.values(), key=lambda x: x["sillok_id"])


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "태조가 왕위에 오르다"
    print(f"[query] {q}")
    print(f"[expanded] {expand_query(q)}\n")
    hits = search(q, top_k=5)
    for i, h in enumerate(hits, 1):
        print(f"--- {i}. score={h.score:.4f} ---")
        print(f"  {h.sillok_title_hanja}({h.sillok_title_hangul}) > {h.day_title}  [{h.chunk_id}]")
        print(f"  제목: {h.article_title}")
        print(f"  {h.text[:200]}")
