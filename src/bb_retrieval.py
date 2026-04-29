"""비변사등록 하이브리드 검색."""
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
QDRANT_PATH = ROOT / "data" / "비변사등록" / "qdrant_storage"
ENTITIES = ROOT / "data" / "비변사등록" / "entities.json"
COLLECTION = "bibyeonsa_deungrok"

DEVICE = "cuda"
DENSE_TOPK = 50
SPARSE_TOPK = 50
FUSE_TOPK = 60
DEFAULT_TOPK = 5


@dataclass
class BbHit:
    chunk_id: str
    article_id: str
    score: float
    volume_id: str
    year_ce: str
    ganji: str
    king: str
    reign_year: str
    month_value: str
    article_title: str
    date_western: str
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


def search(query: str, top_k: int = DEFAULT_TOPK, filters: dict | None = None) -> list[BbHit]:
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
    out_hits = []
    for sc, p in paired[:top_k]:
        pl = p.payload
        out_hits.append(BbHit(
            chunk_id=pl["chunk_id"], article_id=pl["article_id"], score=float(sc),
            volume_id=pl["volume_id"], year_ce=pl.get("year_ce", ""),
            ganji=pl.get("ganji", ""), king=pl.get("king", ""),
            reign_year=pl.get("reign_year", ""), month_value=pl.get("month_value", ""),
            article_title=pl["article_title"],
            date_western=pl["date_western"],
            text=pl["text"], entities=pl.get("entities", {}),
        ))
    return out_hits


def get_article(chunk_id: str, with_context: bool = True) -> dict | None:
    _ensure_loaded()
    client: QdrantClient = _state["qdrant"]
    res, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=qm.Filter(must=[qm.FieldCondition(key="chunk_id", match=qm.MatchValue(value=chunk_id))]),
        limit=1, with_payload=True,
    )
    if not res: return None
    p = res[0]
    out = {"target": p.payload}
    if with_context:
        aid = p.payload["article_id"]
        ares, _ = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=qm.Filter(must=[qm.FieldCondition(key="article_id", match=qm.MatchValue(value=aid))]),
            limit=20, with_payload=True,
        )
        out["article_slices"] = sorted(
            [pt.payload for pt in ares if pt.payload["chunk_id"] != chunk_id],
            key=lambda x: x["chunk_id"],
        )
    return out


@lru_cache(maxsize=1)
def list_volumes() -> list[dict]:
    _ensure_loaded()
    client: QdrantClient = _state["qdrant"]
    seen: dict[str, dict] = {}
    next_page = None
    for _ in range(50):
        res, next_page = client.scroll(COLLECTION, limit=2000, offset=next_page, with_payload=True)
        for p in res:
            v = p.payload["volume_id"]
            if v not in seen:
                seen[v] = {"volume_id": v, "volume_label": p.payload.get("volume_label", "")}
        if not next_page: break
    return sorted(seen.values(), key=lambda x: x["volume_id"])


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "비변사 좌목"
    for i, h in enumerate(search(q, top_k=5), 1):
        print(f"--- {i}. score={h.score:.4f} ---")
        print(f"  [{h.king} {h.reign_year}년 {h.year_ce}] {h.month_value}월 > {h.article_title}  [{h.chunk_id}]")
        print(f"  {h.text[:200]}")
