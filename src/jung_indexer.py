"""data/중국정사외국전/chunks.jsonl → Qdrant (dense + sparse 하이브리드).

BGE-M3로 dense(1024) + sparse 동시 산출.
Qdrant 로컬 파일 모드 (data/중국정사외국전/qdrant_storage). collection: jung_jeongsa.
임베딩 입력은 청크의 결합 텍스트(제목+국역+원문)를 그대로 사용.
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
CHUNKS = ROOT / "data" / "중국정사외국전" / "chunks.jsonl"
QDRANT_PATH = ROOT / "data" / "중국정사외국전" / "qdrant_storage"
COLLECTION = "jung_jeongsa"

DENSE_DIM = 1024
ENCODE_BATCH = 32
UPSERT_BATCH = 256
MAX_LENGTH = 512
THROTTLE = float(os.environ.get("INDEXER_THROTTLE", "0"))


def load_chunks() -> list[dict]:
    if not CHUNKS.exists():
        print(f"[err] chunks not found: {CHUNKS} (run jung_chunker.py first)", file=sys.stderr)
        sys.exit(1)
    with CHUNKS.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def make_embedding_text(c: dict) -> str:
    """청크 text는 이미 [제목 컨텍스트] [국역] ... [원문] ... 형태.
    한글 음차본을 보강해 한자 ↔ 한글 검색 매칭률을 올린다.
    """
    base = c["text"]
    aux = c.get("text_hangul_aux", "")
    if aux and aux != base:
        return base + "\n[한글: " + aux + "]"
    return base


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
        ("book_id", qm.PayloadSchemaType.KEYWORD),
        ("book_name", qm.PayloadSchemaType.KEYWORD),
        ("chapter_id", qm.PayloadSchemaType.KEYWORD),
        ("article_id", qm.PayloadSchemaType.KEYWORD),
        ("subject_country", qm.PayloadSchemaType.KEYWORD),
        ("entities.person", qm.PayloadSchemaType.KEYWORD),
        ("entities.place", qm.PayloadSchemaType.KEYWORD),
        ("entities.nation", qm.PayloadSchemaType.KEYWORD),
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
                "book_id": c["book_id"],
                "book_name": c["book_name"],
                "book_title_hanja": c["book_title_hanja"],
                "book_title_alt": c["book_title_alt"],
                "book_author": c["book_author"],
                "book_dynasty": c["book_dynasty"],
                "period_label": c["period_label"],
                "chapter_id": c["chapter_id"],
                "chapter_title": c["chapter_title"],
                "article_title": c["article_title"],
                "subject_country": c["subject_country"],
                "text": c["text"],
                "text_hanmun": c["text_hanmun"],
                "text_korean": c["text_korean"],
                "text_hangul_aux": c["text_hangul_aux"],
                "has_footnotes_korean": c["has_footnotes_korean"],
                "has_footnotes_hanmun": c["has_footnotes_hanmun"],
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
