"""동문휘고 하이브리드 검색."""
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
QDRANT_PATH = ROOT / "data" / "동문휘고" / "qdrant_storage"
ENTITIES = ROOT / "data" / "동문휘고" / "entities.json"
COLLECTION = "dongmun_hwigo"

DEVICE = "cuda"
DENSE_TOPK = 50
SPARSE_TOPK = 50
FUSE_TOPK = 60
DEFAULT_TOPK = 5


@dataclass
class DhHit:
    chunk_id: str
    level3_id: str
    score: float
    volume_num: str
    volume_name: str
    title_hanja: str
    title_hangul: str
    sender: str
    receiver: str
    date_western: str
    date_text: str
    text: str
    text_hangul_aux: str
    text_translation_korean: str
    hanja_kind: str
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
                for hj, hgs in terms.items():
                    for hg in hgs:
                        inv.setdefault(hg, set()).add(hj)
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
            for v in duum_variants(tok):
                if v in inv:
                    cands |= inv[v]
            for c in sorted(cands):
                if added >= max_expansions:
                    break
                parts.append(c)
                added += 1
            if added >= max_expansions:
                break
    return " ".join(parts)


def _hybrid(client, dense_vec, sparse_vec, limit, qfilter):
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
        limit=limit, query_filter=qfilter, with_payload=True,
    )
    return res.points


def _rerank(query, hits, top_k):
    if not hits:
        return []
    pairs = [[query, h.payload["text"]] for h in hits]
    scores = _state["rerank"].compute_score(pairs, normalize=True)
    if not isinstance(scores, list):
        scores = [scores]
    paired = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
    out = []
    for s, h in paired[:top_k]:
        h.score = float(s)
        out.append(h)
    return out


def _make_filter(filters):
    if not filters:
        return None
    must = []
    for k, v in filters.items():
        if isinstance(v, list):
            must.append(qm.FieldCondition(key=k, match=qm.MatchAny(any=v)))
        else:
            must.append(qm.FieldCondition(key=k, match=qm.MatchValue(value=v)))
    return qm.Filter(must=must) if must else None


def _to_hit(p):
    pl = p.payload
    return DhHit(
        chunk_id=pl["chunk_id"],
        level3_id=pl["level3_id"],
        score=float(p.score),
        volume_num=pl["volume_num"],
        volume_name=pl["volume_name"],
        title_hanja=pl["title_hanja"],
        title_hangul=pl["title_hangul"],
        sender=pl["sender"],
        receiver=pl["receiver"],
        date_western=pl["date_western"],
        date_text=pl["date_text"],
        text=pl["text"],
        text_hangul_aux=pl["text_hangul_aux"],
        text_translation_korean=pl.get("text_translation_korean", ""),
        hanja_kind=pl.get("hanja_kind", ""),
        entities=pl.get("entities", {}),
    )


def search(query, top_k=DEFAULT_TOPK, filters=None):
    _ensure_loaded()
    expanded = expand_query(query)
    out = _state["embed"].encode(
        [expanded], max_length=512, return_dense=True, return_sparse=True,
    )
    dense_vec = out["dense_vecs"][0].tolist()
    sparse_vec = out["lexical_weights"][0]
    qfilter = _make_filter(filters)
    fused = _hybrid(_state["qdrant"], dense_vec, sparse_vec, FUSE_TOPK, qfilter)
    reranked = _rerank(query, fused, top_k)
    return [_to_hit(p) for p in reranked]


def get_document(chunk_id, with_context=True):
    """chunk_id로 단일 + 같은 level3 슬라이스 + 같은 권 인접 문서."""
    _ensure_loaded()
    client = _state["qdrant"]
    res, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=qm.Filter(must=[qm.FieldCondition(
            key="chunk_id", match=qm.MatchValue(value=chunk_id))]),
        limit=1, with_payload=True,
    )
    if not res:
        return None
    p = res[0]
    out = {"target": p.payload}
    if with_context:
        l3id = p.payload["level3_id"]
        ares, _ = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=qm.Filter(must=[qm.FieldCondition(
                key="level3_id", match=qm.MatchValue(value=l3id))]),
            limit=20, with_payload=True,
        )
        slices = sorted(
            [pt.payload for pt in ares if pt.payload["chunk_id"] != chunk_id],
            key=lambda x: x["chunk_id"],
        )
        out["document_slices"] = slices
    return out


@lru_cache(maxsize=1)
def list_volumes():
    _ensure_loaded()
    client = _state["qdrant"]
    seen: dict[str, dict] = {}
    next_page = None
    for _ in range(30):
        res, next_page = client.scroll(
            collection_name=COLLECTION, limit=2000, offset=next_page, with_payload=True,
        )
        for p in res:
            v = p.payload["volume_num"]
            if v not in seen:
                seen[v] = {
                    "volume_num": v,
                    "volume_name": p.payload["volume_name"],
                }
        if not next_page:
            break
    return sorted(seen.values(), key=lambda x: x["volume_num"])


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "세자 책봉"
    print(f"[query] {q}")
    print(f"[expanded] {expand_query(q)}\n")
    hits = search(q, top_k=5)
    for i, h in enumerate(hits, 1):
        print(f"--- {i}. score={h.score:.4f} [{h.chunk_id}] ---")
        print(f"  {h.volume_name}  date={h.date_western}")
        print(f"  {h.sender} → {h.receiver}")
        print(f"  제목: {h.title_hangul}")
        print(f"  본문: {h.text[:200]}")
