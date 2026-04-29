"""승정원일기 chunks.jsonl → Qdrant (collection: sjw)."""
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
CHUNKS = ROOT / "data" / "승정원일기" / "chunks.jsonl"
QDRANT_PATH = ROOT / "data" / "승정원일기" / "qdrant_storage"
COLLECTION = "sjw"

DENSE_DIM = 1024
ENCODE_BATCH = 32
UPSERT_BATCH = 256
MAX_LENGTH = 512
THROTTLE = float(os.environ.get("INDEXER_THROTTLE", "0"))


def load_chunks() -> list[dict]:
    if not CHUNKS.exists():
        print(f"[err] {CHUNKS}", file=sys.stderr); sys.exit(1)
    with CHUNKS.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def make_embedding_text(c: dict) -> str:
    base = c["text"]
    aux = c.get("text_hangul_aux", "")
    if aux and aux != base:
        return base + "\n[한글: " + aux + "]"
    return base


def setup_collection(client: QdrantClient, resume: bool = False) -> int:
    if client.collection_exists(COLLECTION):
        if resume:
            n = client.get_collection(COLLECTION).points_count
            safe = max(0, n - UPSERT_BATCH)
            safe = (safe // ENCODE_BATCH) * ENCODE_BATCH
            print(f"[info] resume from {safe}")
            return safe
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        # on_disk=True: 200만 청크 × 1024-dim × float64 ≈ 17 GiB → memmap on disk
        # (Qdrant local file 모드는 기본적으로 dense 벡터를 in-memory numpy로 보관)
        vectors_config={"dense": qm.VectorParams(size=DENSE_DIM, distance=qm.Distance.COSINE, on_disk=True)},
        sparse_vectors_config={"sparse": qm.SparseVectorParams()},
    )
    for field, schema in [
        ("year_id", qm.PayloadSchemaType.KEYWORD),
        ("king", qm.PayloadSchemaType.KEYWORD),
        ("king_prefix", qm.PayloadSchemaType.KEYWORD),
        ("year_ce", qm.PayloadSchemaType.KEYWORD),
        ("day_id", qm.PayloadSchemaType.KEYWORD),
        ("article_id", qm.PayloadSchemaType.KEYWORD),
        ("date_western", qm.PayloadSchemaType.KEYWORD),
        ("ganji", qm.PayloadSchemaType.KEYWORD),
        ("entities.person", qm.PayloadSchemaType.KEYWORD),
        ("entities.place", qm.PayloadSchemaType.KEYWORD),
        ("entities.title", qm.PayloadSchemaType.KEYWORD),
    ]:
        try:
            client.create_payload_index(COLLECTION, field, schema)
        except Exception as e:
            print(f"[warn] payload index {field}: {e}")
    return 0


def _prevent_sleep() -> None:
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)
    except Exception:
        pass


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    _prevent_sleep()
    chunks = load_chunks()
    print(f"[info] {len(chunks)} chunks")
    print("[info] loading BGE-M3")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda")
    QDRANT_PATH.parent.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))
    start = setup_collection(client, resume=args.resume)
    pbar = tqdm(total=len(chunks), desc="embed", initial=start)
    pending: list[qm.PointStruct] = []

    def flush() -> None:
        if pending:
            client.upsert(COLLECTION, points=pending); pending.clear()

    for bs in range(start, len(chunks), ENCODE_BATCH):
        t0 = time.perf_counter()
        batch = chunks[bs:bs+ENCODE_BATCH]
        texts = [make_embedding_text(c) for c in batch]
        out = model.encode(texts, batch_size=ENCODE_BATCH, max_length=MAX_LENGTH,
                            return_dense=True, return_sparse=True, return_colbert_vecs=False)
        dvecs, svecs = out["dense_vecs"], out["lexical_weights"]
        for i, c in enumerate(batch):
            sw = svecs[i]
            payload = {k: c[k] for k in c}
            pending.append(qm.PointStruct(
                id=bs + i,
                vector={"dense": dvecs[i].tolist(),
                         "sparse": qm.SparseVector(indices=[int(k) for k in sw.keys()],
                                                    values=[float(v) for v in sw.values()])},
                payload=payload,
            ))
        if len(pending) >= UPSERT_BATCH:
            flush()
        pbar.update(len(batch))
        if THROTTLE > 0:
            time.sleep((time.perf_counter() - t0) * THROTTLE)
    flush(); pbar.close()
    info = client.get_collection(COLLECTION)
    print(f"[done] points: {info.points_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
