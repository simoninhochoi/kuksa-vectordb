"""승정원일기 하이브리드 검색 (Docker Qdrant 서버 단일 collection).

데이터: server collection `sjw_full` 에 200만 points
  - server_id 0~1,258,763 = 인조~정조 (sjw 마이그)
  - server_id 1,258,764~1,909,771 = 순조~고종 전반 (sjw2 마이그)
  - server_id 1,909,772~2,051,476 = 고종 후반~순종 (sjw3 마이그)
  - 순조 partial 3,831 points 는 마이그레이션 단계에서 필터링 제외됨

Docker container `qdrant_sjw` (port 6333) 에 의존. 시작 명령:
  docker start qdrant_sjw

이전 local mode 3-collection 분할 아키텍처는 폐기 (cold start 30분+ 문제 해결).
"""
from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

from FlagEmbedding import BGEM3FlagModel, FlagReranker
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from hanja_util import has_hanja, has_hangul, to_hangul, duum_variants

ROOT = Path(__file__).resolve().parents[1]

SERVER_HOST = "localhost"
SERVER_PORT = 6333
COLLECTION = "sjw_full"
ENTITIES = ROOT / "data" / "승정원일기" / "entities.json"

DEVICE = "cuda"
DENSE_TOPK = 50
SPARSE_TOPK = 50
FUSE_TOPK = 60
DEFAULT_TOPK = 5


@dataclass
class SjwHit:
    chunk_id: str
    article_id: str
    score: float
    king: str
    king_prefix: str
    year_id: str
    year_label: str
    year_ce: str
    day_id: str
    day_title: str
    date_western: str
    ganji: str
    reign_year: str
    article_type: str
    article_title: str
    text: str
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
            _state["qdrant"] = QdrantClient(host=SERVER_HOST, port=SERVER_PORT, timeout=60)
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
            if tok in inv: cands |= inv[tok]
            for v in duum_variants(tok):
                if v in inv: cands |= inv[v]
            for c in sorted(cands):
                if added >= max_expansions: break
                parts.append(c); added += 1
            if added >= max_expansions: break
    return " ".join(parts)


def _make_filter(filters: dict | None) -> qm.Filter | None:
    if not filters: return None
    must = []
    for k, v in filters.items():
        if isinstance(v, list):
            must.append(qm.FieldCondition(key=k, match=qm.MatchAny(any=v)))
        else:
            must.append(qm.FieldCondition(key=k, match=qm.MatchValue(value=v)))
    return qm.Filter(must=must) if must else None


def search(query: str, top_k: int = DEFAULT_TOPK, filters: dict | None = None) -> list[SjwHit]:
    _ensure_loaded()
    expanded = expand_query(query)
    out = _state["embed"].encode([expanded], max_length=512, return_dense=True, return_sparse=True)
    dvec = out["dense_vecs"][0].tolist()
    svec = out["lexical_weights"][0]
    sparse = qm.SparseVector(indices=[int(k) for k in svec.keys()],
                              values=[float(v) for v in svec.values()])
    qfilter = _make_filter(filters)
    res = _state["qdrant"].query_points(
        collection_name=COLLECTION,
        prefetch=[
            qm.Prefetch(query=dvec, using="dense", limit=DENSE_TOPK, filter=qfilter),
            qm.Prefetch(query=sparse, using="sparse", limit=SPARSE_TOPK, filter=qfilter),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=FUSE_TOPK, query_filter=qfilter, with_payload=True,
    )
    hits = res.points
    if not hits:
        return []

    pairs = [[query, h.payload["text"]] for h in hits]
    scores = _state["rerank"].compute_score(pairs, normalize=True)
    if not isinstance(scores, list):
        scores = [scores]
    paired = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
    out_hits: list[SjwHit] = []
    for sc, p in paired[:top_k]:
        pl = p.payload
        out_hits.append(SjwHit(
            chunk_id=pl["chunk_id"], article_id=pl["article_id"], score=float(sc),
            king=pl.get("king", ""), king_prefix=pl.get("king_prefix", ""),
            year_id=pl.get("year_id", ""), year_label=pl.get("year_label", ""),
            year_ce=pl.get("year_ce", ""),
            day_id=pl.get("day_id", ""), day_title=pl.get("day_title", ""),
            date_western=pl["date_western"], ganji=pl.get("ganji", ""),
            reign_year=pl.get("reign_year", ""),
            article_type=pl.get("article_type", ""),
            article_title=pl["article_title"],
            text=pl["text"], entities=pl.get("entities", {}),
        ))
    return out_hits


def get_article(chunk_id: str, with_context: bool = True) -> dict | None:
    _ensure_loaded()
    client: QdrantClient = _state["qdrant"]
    res, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=qm.Filter(must=[qm.FieldCondition(
            key="chunk_id", match=qm.MatchValue(value=chunk_id))]),
        limit=1, with_payload=True,
    )
    if not res:
        return None
    p = res[0]
    out: dict = {"target": p.payload}
    if with_context:
        aid = p.payload["article_id"]
        ares, _ = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=qm.Filter(must=[qm.FieldCondition(
                key="article_id", match=qm.MatchValue(value=aid))]),
            limit=20, with_payload=True,
        )
        out["article_slices"] = sorted(
            [pt.payload for pt in ares if pt.payload["chunk_id"] != chunk_id],
            key=lambda x: x["chunk_id"],
        )
    return out


@lru_cache(maxsize=1)
def list_kings() -> list[dict]:
    """수록된 왕 목록 (인조~순종 12명)."""
    _ensure_loaded()
    client: QdrantClient = _state["qdrant"]
    seen: dict[str, dict] = {}
    next_page = None
    for _ in range(500):
        res, next_page = client.scroll(COLLECTION, limit=5000, offset=next_page, with_payload=True)
        for p in res:
            kp = p.payload.get("king_prefix", "")
            if kp and kp not in seen:
                seen[kp] = {"king_prefix": kp, "king": p.payload.get("king", "")}
        if not next_page: break
    return sorted(seen.values(), key=lambda x: x["king_prefix"])


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "인조가 청나라 사신을 접견하다"
    print(f"[query] {q}")
    for i, h in enumerate(search(q, top_k=5), 1):
        print(f"--- {i}. score={h.score:.4f} ---")
        print(f"  {h.king} {h.year_ce} > {h.day_title} > {h.article_title}  [{h.chunk_id}]")
        print(f"  {h.text[:200]}")
