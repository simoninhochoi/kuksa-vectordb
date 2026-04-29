"""data/동문휘고/chunks.jsonl → Qdrant (dense + sparse 하이브리드).

collection: 'dongmun_hwigo'.
임베딩 입력: 제목·sender·receiver 컨텍스트 + 한자 본문 + 한글 음차 + (있을 시) 한국어 번역.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "동문휘고" / "chunks.jsonl"
QDRANT_PATH = ROOT / "data" / "동문휘고" / "qdrant_storage"
COLLECTION = "dongmun_hwigo"

DENSE_DIM = 1024
ENCODE_BATCH = 32
UPSERT_BATCH = 256
MAX_LENGTH = 512
THROTTLE = float(os.environ.get("INDEXER_THROTTLE", "0"))


def load_chunks() -> list[dict]:
    if not CHUNKS.exists():
        print(f"[err] chunks not found: {CHUNKS}", file=sys.stderr)
        sys.exit(1)
    with CHUNKS.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def make_embedding_text(c: dict) -> str:
    """제목·발수신자 컨텍스트 + 본문 + 한글 음차 + (있을 시) 한국어 번역."""
    parts: list[str] = []
    title_bits = [c.get("title_hangul", ""), c.get("title_hanja", "")]
    s = c.get("sender", "")
    r = c.get("receiver", "")
    sr = []
    if s:
        sr.append(f"발신:{s}")
    if r:
        sr.append(f"수신:{r}")
    ctx = [b for b in title_bits if b] + sr
    if ctx:
        parts.append(f"[{' | '.join(ctx)}]")
    parts.append(c["text"])
    aux = c.get("text_hangul_aux", "")
    if aux and aux != c["text"]:
        parts.append(f"[한글: {aux}]")
    kor = c.get("text_translation_korean", "")
    if kor:
        parts.append(f"[번역: {kor}]")
    return " ".join(parts)


def setup_collection(client: QdrantClient, resume: bool = False) -> int:
    if client.collection_exists(COLLECTION):
        if resume:
            n = client.get_collection(COLLECTION).points_count
            safe_start = max(0, n - UPSERT_BATCH)
            safe_start = (safe_start // ENCODE_BATCH) * ENCODE_BATCH
            print(f"[info] resume: {n} points; restart from {safe_start}")
            return safe_start
        print(f"[info] collection {COLLECTION} exists; recreating")
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": qm.VectorParams(size=DENSE_DIM, distance=qm.Distance.COSINE)},
        sparse_vectors_config={"sparse": qm.SparseVectorParams()},
    )
    for field, schema in [
        ("volume_num", qm.PayloadSchemaType.KEYWORD),
        ("level3_id", qm.PayloadSchemaType.KEYWORD),
        ("hanja_kind", qm.PayloadSchemaType.KEYWORD),
        ("date_western", qm.PayloadSchemaType.KEYWORD),
        ("sender", qm.PayloadSchemaType.KEYWORD),
        ("receiver", qm.PayloadSchemaType.KEYWORD),
        ("entities.person", qm.PayloadSchemaType.KEYWORD),
        ("entities.place", qm.PayloadSchemaType.KEYWORD),
    ]:
        try:
            client.create_payload_index(
                collection_name=COLLECTION, field_name=field, field_schema=schema,
            )
        except Exception as e:
            print(f"[warn] payload index {field}: {e}")
    return 0


def _prevent_sleep() -> None:
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x1 | 0x2)
        print("[info] sleep prevention enabled")
    except Exception as e:
        print(f"[warn] sleep prevention: {e}")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    _prevent_sleep()
    chunks = load_chunks()
    print(f"[info] {len(chunks)} chunks to index")

    print("[info] loading BGE-M3")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda")

    QDRANT_PATH.parent.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))
    start = setup_collection(client, resume=args.resume)

    pbar = tqdm(total=len(chunks), desc="embedding+upsert", initial=start)
    pending: list[qm.PointStruct] = []

    def flush() -> None:
        if pending:
            client.upsert(collection_name=COLLECTION, points=pending)
            pending.clear()

    if THROTTLE > 0:
        print(f"[info] throttle={THROTTLE}")

    for batch_start in range(start, len(chunks), ENCODE_BATCH):
        t0 = time.perf_counter()
        batch = chunks[batch_start : batch_start + ENCODE_BATCH]
        texts = [make_embedding_text(c) for c in batch]

        out = model.encode(
            texts, batch_size=ENCODE_BATCH, max_length=MAX_LENGTH,
            return_dense=True, return_sparse=True, return_colbert_vecs=False,
        )
        dense_vecs = out["dense_vecs"]
        sparse_vecs = out["lexical_weights"]

        for i, c in enumerate(batch):
            sw = sparse_vecs[i]
            indices = [int(k) for k in sw.keys()]
            values = [float(v) for v in sw.values()]
            payload = {
                "chunk_id": c["chunk_id"],
                "level3_id": c["level3_id"],
                "volume_num": c["volume_num"],
                "volume_name": c["volume_name"],
                "hanja_kind": c["hanja_kind"],
                "title_hanja": c["title_hanja"],
                "title_hangul": c["title_hangul"],
                "sender": c["sender"],
                "receiver": c["receiver"],
                "date_western": c["date_western"],
                "date_is_leap": c["date_is_leap"],
                "date_text": c["date_text"],
                "text": c["text"],
                "text_hangul_aux": c["text_hangul_aux"],
                "text_translation_korean": c["text_translation_korean"],
                "entities": c.get("entities", {}),
            }
            pending.append(qm.PointStruct(
                id=batch_start + i,
                vector={
                    "dense": dense_vecs[i].tolist(),
                    "sparse": qm.SparseVector(indices=indices, values=values),
                },
                payload=payload,
            ))

        if len(pending) >= UPSERT_BATCH:
            flush()
        pbar.update(len(batch))

        if THROTTLE > 0:
            time.sleep((time.perf_counter() - t0) * THROTTLE)

    flush()
    pbar.close()
    info = client.get_collection(COLLECTION)
    print(f"[done] collection points: {info.points_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
