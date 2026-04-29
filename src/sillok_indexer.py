"""data/조선왕조실록/chunks.jsonl → Qdrant (dense + sparse 하이브리드).

BGE-M3로 dense(1024-dim) + sparse 동시 산출.
Qdrant 로컬 파일 모드 (data/조선왕조실록/qdrant_storage). 한국사료총서와 별도 저장소.
임베딩 입력은 "제목 컨텍스트 + 본문(한자) + [한글 음차]" 병기.
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
CHUNKS = ROOT / "data" / "조선왕조실록" / "chunks.jsonl"
QDRANT_PATH = ROOT / "data" / "조선왕조실록" / "qdrant_storage"
COLLECTION = "sillok"

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
    """임베딩 입력: 왕·기사 제목 컨텍스트 + 본문 + 한글 음차."""
    parts: list[str] = []
    # 제목 컨텍스트 (한글 기사제목 + 왕)
    king = c.get("king", "")
    atitle = c.get("article_title", "")
    day = c.get("day_title", "")
    ctx_bits = [b for b in [king, day, atitle] if b]
    if ctx_bits:
        parts.append(f"[{' | '.join(ctx_bits)}]")
    parts.append(c["text"])
    aux = c.get("text_hangul_aux", "")
    if aux and aux != c["text"]:
        parts.append(f"[한글: {aux}]")
    return " ".join(parts)


def setup_collection(client: QdrantClient, resume: bool = False) -> int:
    if client.collection_exists(COLLECTION):
        if resume:
            n = client.get_collection(COLLECTION).points_count
            safe_start = max(0, n - UPSERT_BATCH)
            safe_start = (safe_start // ENCODE_BATCH) * ENCODE_BATCH
            print(f"[info] resume: collection has {n} points; restarting from chunk {safe_start}")
            return safe_start
        print(f"[info] collection {COLLECTION} exists; recreating")
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": qm.VectorParams(size=DENSE_DIM, distance=qm.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": qm.SparseVectorParams(),
        },
    )
    for field, schema in [
        ("sillok_id", qm.PayloadSchemaType.KEYWORD),
        ("king", qm.PayloadSchemaType.KEYWORD),
        ("volume_id", qm.PayloadSchemaType.KEYWORD),
        ("year_ce", qm.PayloadSchemaType.KEYWORD),
        ("date_western", qm.PayloadSchemaType.KEYWORD),
        ("subject_classes", qm.PayloadSchemaType.KEYWORD),
        ("entities.person", qm.PayloadSchemaType.KEYWORD),
        ("entities.place", qm.PayloadSchemaType.KEYWORD),
        ("entities.title", qm.PayloadSchemaType.KEYWORD),
    ]:
        try:
            client.create_payload_index(
                collection_name=COLLECTION,
                field_name=field,
                field_schema=schema,
            )
        except Exception as e:
            print(f"[warn] payload index {field}: {e}")
    return 0


def _prevent_sleep() -> None:
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_DISPLAY_REQUIRED = 0x00000002
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )
        print("[info] sleep prevention enabled")
    except Exception as e:
        print(f"[warn] sleep prevention failed: {e}")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="기존 컬렉션 유지하고 중단된 지점부터 이어서 인덱싱")
    args = ap.parse_args()

    _prevent_sleep()
    chunks = load_chunks()
    print(f"[info] {len(chunks)} chunks to index")

    print("[info] loading BGE-M3")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda")

    QDRANT_PATH.parent.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))
    start_chunk = setup_collection(client, resume=args.resume)

    pbar = tqdm(total=len(chunks), desc="embedding+upsert", initial=start_chunk)
    pending: list[qm.PointStruct] = []

    def flush() -> None:
        if pending:
            client.upsert(collection_name=COLLECTION, points=pending)
            pending.clear()

    if THROTTLE > 0:
        print(f"[info] throttle={THROTTLE}")

    for batch_start in range(start_chunk, len(chunks), ENCODE_BATCH):
        t0 = time.perf_counter()
        batch = chunks[batch_start : batch_start + ENCODE_BATCH]
        texts = [make_embedding_text(c) for c in batch]

        out = model.encode(
            texts,
            batch_size=ENCODE_BATCH,
            max_length=MAX_LENGTH,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vecs = out["dense_vecs"]
        sparse_vecs = out["lexical_weights"]

        for i, c in enumerate(batch):
            sw = sparse_vecs[i]
            indices = [int(k) for k in sw.keys()]
            values = [float(v) for v in sw.values()]
            payload = {
                "chunk_id": c["chunk_id"],
                "article_id": c["article_id"],
                "sillok_id": c["sillok_id"],
                "sillok_title_hanja": c["sillok_title_hanja"],
                "sillok_title_hangul": c["sillok_title_hangul"],
                "king": c["king"],
                "volume_id": c["volume_id"],
                "volume_title": c["volume_title"],
                "year_ce": c["year_ce"],
                "month_title": c["month_title"],
                "day_title": c["day_title"],
                "date_western": c["date_western"],
                "date_is_leap": c["date_is_leap"],
                "ganji": c["ganji"],
                "reign_year": c["reign_year"],
                "china_era": c["china_era"],
                "article_title": c["article_title"],
                "subject_classes": c["subject_classes"],
                "source_refs": c["source_refs"],
                "text": c["text"],
                "text_hangul_aux": c["text_hangul_aux"],
                "entities": c.get("entities", {}),
            }
            pending.append(
                qm.PointStruct(
                    id=batch_start + i,
                    vector={
                        "dense": dense_vecs[i].tolist(),
                        "sparse": qm.SparseVector(indices=indices, values=values),
                    },
                    payload=payload,
                )
            )

        if len(pending) >= UPSERT_BATCH:
            flush()
        pbar.update(len(batch))

        if THROTTLE > 0:
            elapsed = time.perf_counter() - t0
            time.sleep(elapsed * THROTTLE)

    flush()
    pbar.close()

    info = client.get_collection(COLLECTION)
    print(f"[done] collection points: {info.points_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
