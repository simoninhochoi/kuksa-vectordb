"""승정원일기 3 collection → Docker Qdrant 서버 (SQLite 직접 추출).

qdrant_client local mode 가 16GB sjw collection 여는데 1시간+ hang 됨.
우회: SQLite의 `points (id TEXT, point BLOB)` 를 직접 query, BLOB 은 pickled
qdrant_client.PointStruct 객체 → 그대로 server 에 upsert.

ID 재매핑:
  - sjw  : local id 0~1,258,763 → server id 0~1,258,763 (순조 partial >=1,258,764 skip)
  - sjw2 : local id 0~651,007 → server id 1,258,764~1,909,771
  - sjw3 : local id 0~141,704 → server id 1,909,772~2,051,476
  결과 server collection sjw_full 의 ID = chunks.jsonl 의 line 번호.
"""
from __future__ import annotations

import sqlite3
import pickle
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

# (name, sqlite path, server_id_offset, local_id_max_exclusive)
LOCAL_DBS = [
    ("sjw",
     ROOT / "data" / "승정원일기" / "qdrant_storage" / "collection" / "sjw" / "storage.sqlite",
     0, 1_258_764),
    ("sjw2",
     ROOT / "data" / "승정원일기" / "qdrant_storage_sjw2" / "collection" / "sjw2" / "storage.sqlite",
     1_258_764, None),
    ("sjw3",
     ROOT / "data" / "승정원일기" / "qdrant_storage_sjw3" / "collection" / "sjw3" / "storage.sqlite",
     1_909_772, None),
]

SERVER_HOST = "localhost"
SERVER_PORT = 6333
SERVER_COLLECTION = "sjw_full"

DENSE_DIM = 1024
UPSERT_BATCH = 200
PROGRESS_EVERY = 5000


def setup_server_collection(server: QdrantClient, fresh: bool = False) -> None:
    if server.collection_exists(SERVER_COLLECTION):
        if fresh:
            print(f"[info] 기존 server collection 삭제 후 재생성")
            server.delete_collection(SERVER_COLLECTION)
        else:
            n = server.get_collection(SERVER_COLLECTION).points_count
            print(f"[info] server collection 이미 존재 ({n:,} points). 동일 ID 는 덮어씀.")
            return
    print(f"[info] 서버 collection {SERVER_COLLECTION} 생성 (on_disk=True for memmap)")
    # 200만 points × 1024-dim × 4 bytes = 8 GB
    # Docker Desktop RAM 한도 7.5 GB 이하라 메모리 모드는 OOM/freeze 위험
    # → vectors + HNSW 모두 on_disk (memmap), 일부만 OS 캐시 사용
    server.create_collection(
        collection_name=SERVER_COLLECTION,
        vectors_config={
            "dense": qm.VectorParams(
                size=DENSE_DIM,
                distance=qm.Distance.COSINE,
                on_disk=True,    # vectors memmap
                hnsw_config=qm.HnswConfigDiff(on_disk=True),  # 인덱스도 디스크
            ),
        },
        sparse_vectors_config={"sparse": qm.SparseVectorParams()},
        on_disk_payload=True,
        # 인덱싱 자체 임계값 — 첫 N points 동안은 in-memory build, 그 이상은 segment 분할
        optimizers_config=qm.OptimizersConfigDiff(
            indexing_threshold=20000,
            memmap_threshold=20000,    # 20k points 이상 segment 는 memmap 으로
            default_segment_number=4,
        ),
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
            server.create_payload_index(SERVER_COLLECTION, field, schema)
        except Exception as e:
            print(f"[warn] payload_index {field}: {e}")


def migrate_one(name: str, sqlite_path: Path, server_id_offset: int,
                 local_id_max: int | None, server: QdrantClient) -> tuple[int, int]:
    """SQLite → server 직접 마이그. (migrated, skipped) 반환."""
    if not sqlite_path.exists():
        print(f"[skip] {sqlite_path} 없음")
        return 0, 0

    print(f"\n=== {name} 마이그레이션 시작 ({sqlite_path.parent.parent.name}) ===")
    conn = sqlite3.connect(str(sqlite_path))
    conn.execute("PRAGMA query_only = ON")  # 읽기 전용
    cur = conn.cursor()
    n_total = cur.execute("SELECT COUNT(*) FROM points").fetchone()[0]
    print(f"[info] {name}: {n_total:,} rows")

    pending: list[qm.PointStruct] = []
    migrated = 0
    skipped = 0
    pbar = tqdm(total=n_total, desc=name, smoothing=0.05)

    # streaming cursor — load row by row
    for row in cur.execute("SELECT id, point FROM points"):
        _id_blob, point_blob = row
        try:
            p: qm.PointStruct = pickle.loads(point_blob)
        except Exception as e:
            skipped += 1
            pbar.update(1)
            continue

        local_id = p.id if isinstance(p.id, int) else None
        if local_id is None:
            skipped += 1
            pbar.update(1)
            continue

        # ID 범위 필터 (sjw 의 순조 partial 제외)
        if local_id_max is not None and local_id >= local_id_max:
            skipped += 1
            pbar.update(1)
            continue

        new_id = server_id_offset + local_id
        # 새 PointStruct (ID 재매핑)
        pending.append(qm.PointStruct(
            id=new_id,
            vector=p.vector,   # {'dense': [...], 'sparse': SparseVector(...)}
            payload=p.payload,
        ))

        if len(pending) >= UPSERT_BATCH:
            try:
                server.upsert(SERVER_COLLECTION, points=pending, wait=False)
            except Exception as e:
                print(f"\n[err] upsert: {e}")
                # 재시도 1회
                time.sleep(2)
                server.upsert(SERVER_COLLECTION, points=pending, wait=False)
            migrated += len(pending)
            pending.clear()

        pbar.update(1)

    if pending:
        server.upsert(SERVER_COLLECTION, points=pending, wait=False)
        migrated += len(pending)
        pending.clear()

    pbar.close()
    conn.close()
    print(f"[info] {name}: migrated={migrated:,}, skipped={skipped:,}, "
          f"server_id 범위={server_id_offset}~{server_id_offset + (local_id_max or n_total) - 1}")
    return migrated, skipped


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="server collection 삭제 후 재생성")
    args = ap.parse_args()

    print(f"[info] Docker server 연결 ({SERVER_HOST}:{SERVER_PORT})")
    server = QdrantClient(host=SERVER_HOST, port=SERVER_PORT, timeout=120)
    server.get_collections()  # ping

    setup_server_collection(server, fresh=args.fresh)

    total_migrated = 0
    total_skipped = 0
    for name, path, offset, max_id in LOCAL_DBS:
        m, s = migrate_one(name, path, offset, max_id, server)
        total_migrated += m
        total_skipped += s

    # 최종 검증
    info = server.get_collection(SERVER_COLLECTION)
    print(f"\n[done] migrated={total_migrated:,}, skipped={total_skipped:,}, "
          f"server points={info.points_count:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
