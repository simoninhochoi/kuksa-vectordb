"""chunks.jsonl → Qdrant (dense + sparse 하이브리드).

BGE-M3로 dense(1024-dim) + sparse(lexical weights) 동시 산출.
Qdrant는 로컬 파일 모드 (data/qdrant_storage). Docker 불필요.

임베딩 입력은 한자 본문 + 한글 음차 병기:
  "原文 [한글: 한글음차]"
이렇게 두 표기를 한 벡터에 묶어 회수율을 끌어올림.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "chunks.jsonl"
QDRANT_PATH = ROOT / "data" / "qdrant_storage"
COLLECTION = "korean_history"

DENSE_DIM = 1024
ENCODE_BATCH = 32
UPSERT_BATCH = 256  # 임베딩 N batch를 모아 한 번에 upsert (Qdrant 디스크 쓰기 절감)
MAX_LENGTH = 512    # chunks p99 ~800자 ≈ 500 token; 1024는 과함
# GPU 발열 조절: batch당 (작업시간 × THROTTLE) 만큼 sleep. 0 = no throttle, 1.0 ≈ 50% utilization
THROTTLE = float(os.environ.get("INDEXER_THROTTLE", "0"))


def load_chunks() -> list[dict]:
    if not CHUNKS.exists():
        print(f"[err] chunks not found: {CHUNKS}", file=sys.stderr)
        sys.exit(1)
    with CHUNKS.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def make_embedding_text(c: dict) -> str:
    """임베딩 입력: 본문 + 한글 음차 병기 + (선택) level 제목."""
    parts = []
    # 제목 컨텍스트 (회수에 도움)
    title_ctx = " > ".join(t for t in [c.get("volume_title_hanja", ""), c.get("level2_title", "")] if t)
    if title_ctx:
        parts.append(f"[{title_ctx}]")
    parts.append(c["text"])
    aux = c.get("text_hangul_aux", "")
    if aux and aux != c["text"]:
        parts.append(f"[한글: {aux}]")
    return " ".join(parts)


def setup_collection(client: QdrantClient, resume: bool = False) -> int:
    """컬렉션 준비. resume=True면 기존 컬렉션 보존하고 다음 시작 chunk index 반환."""
    if client.collection_exists(COLLECTION):
        if resume:
            n = client.get_collection(COLLECTION).points_count
            # 마지막 미flush batch(<=UPSERT_BATCH)가 손실됐을 수 있으므로 안전 마진
            safe_start = max(0, n - UPSERT_BATCH)
            # ENCODE_BATCH 경계로 정렬
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
    # payload index for fast filtering
    for field, schema in [
        ("volume_id", qm.PayloadSchemaType.KEYWORD),
        ("subject_class", qm.PayloadSchemaType.KEYWORD),
        ("period_begin", qm.PayloadSchemaType.KEYWORD),
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
    """Windows: 인덱싱 중 시스템·디스플레이 절전 방지."""
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

    print("[info] loading BGE-M3 (first time will download ~2GB)")
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
        print(f"[info] throttle={THROTTLE} (sleep {THROTTLE}× work-time per batch for GPU cooling)")

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
            pending.append(
                qm.PointStruct(
                    id=batch_start + i,
                    vector={
                        "dense": dense_vecs[i].tolist(),
                        "sparse": qm.SparseVector(indices=indices, values=values),
                    },
                    payload={
                        "chunk_id": c["chunk_id"],
                        "volume_id": c["volume_id"],
                        "volume_title_hanja": c["volume_title_hanja"],
                        "volume_title_hangul": c["volume_title_hangul"],
                        "series_volume": c.get("series_volume"),
                        "author": c["author"],
                        "period_begin": c["period_begin"],
                        "period_end": c["period_end"],
                        "subject_class": c["subject_class"],
                        "level1_title": c["level1_title"],
                        "level2_title": c["level2_title"],
                        "level3_title": c.get("level3_title"),
                        "text": c["text"],
                        "text_hangul_aux": c["text_hangul_aux"],
                        "entities": c.get("entities", {}),
                    },
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
