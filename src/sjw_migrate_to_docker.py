"""승정원일기 3 collection (sjw + sjw2 + sjw3) → Docker Qdrant 서버로 마이그레이션.

기존 local mode collection 들의 vector + payload 를 scroll 로 읽어 Docker 서버
collection 'sjw_full' 에 upsert. BGE-M3 재계산 안 함 (큰 시간 절약).

Pre-requisite:
  docker run -d -p 6333:6333 -v <abs>/data/승정원일기/qdrant_server:/qdrant/storage \\
    --name qdrant_sjw qdrant/qdrant
"""
from __future__ import annotations

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

LOCAL_PATHS = [
    ("sjw",  ROOT / "data" / "승정원일기" / "qdrant_storage"),
    ("sjw2", ROOT / "data" / "승정원일기" / "qdrant_storage_sjw2"),
    ("sjw3", ROOT / "data" / "승정원일기" / "qdrant_storage_sjw3"),
]
SERVER_HOST = "localhost"
SERVER_PORT = 6333
SERVER_COLLECTION = "sjw_full"

DENSE_DIM = 1024
SCROLL_BATCH = 100
UPSERT_BATCH = 100

# sjw collection 의 순조 partial 4k 제외
ALLOWED_KINGS_FOR_SJW1 = {"A", "B", "C", "D", "E", "F", "G"}


def setup_server_collection(server: QdrantClient) -> None:
    if server.collection_exists(SERVER_COLLECTION):
        n = server.get_collection(SERVER_COLLECTION).points_count
        print(f"[info] server collection 이미 존재 ({n:,} points). 그대로 이어 받음.")
        return
    print(f"[info] 서버에 {SERVER_COLLECTION} 컬렉션 생성")
    server.create_collection(
        collection_name=SERVER_COLLECTION,
        vectors_config={"dense": qm.VectorParams(size=DENSE_DIM, distance=qm.Distance.COSINE)},
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
            server.create_payload_index(SERVER_COLLECTION, field, schema)
        except Exception as e:
            print(f"[warn] payload_index {field}: {e}")


def migrate_collection(name: str, local_path: Path, server: QdrantClient,
                        next_id_start: int) -> int:
    """local collection → server upsert. next_id 반환 (다음 collection 의 시작 ID)."""
    print(f"\n=== {name} 마이그레이션 시작 ({local_path.name}) ===")
    if not local_path.exists():
        print(f"[skip] {local_path} 없음")
        return next_id_start

    print(f"[info] local client 열기...")
    t0 = time.perf_counter()
    local = QdrantClient(path=str(local_path))
    print(f"[info] local client 열림 ({time.perf_counter()-t0:.1f}s)")

    n_total = local.get_collection(name).points_count
    print(f"[info] {name}: {n_total:,} points")

    # scroll → upsert
    pbar = tqdm(total=n_total, desc=name)
    next_page = None
    cur_id = next_id_start
    pending: list[qm.PointStruct] = []
    skipped = 0

    while True:
        try:
            res, next_page = local.scroll(
                collection_name=name,
                limit=SCROLL_BATCH,
                offset=next_page,
                with_payload=True,
                with_vectors=True,
            )
        except Exception as e:
            print(f"\n[err] scroll 실패: {e}")
            break

        if not res:
            break

        for p in res:
            pl = p.payload
            # sjw collection 의 순조 partial 제외
            if name == "sjw" and pl.get("king_prefix") not in ALLOWED_KINGS_FOR_SJW1:
                skipped += 1
                pbar.update(1)
                continue

            # vector 형태 통일
            vec = p.vector
            if not isinstance(vec, dict):
                continue
            dense = vec.get("dense")
            sparse = vec.get("sparse")
            if dense is None:
                continue
            sparse_vec = None
            if sparse is not None:
                # qdrant local 의 SparseVector 객체 → server 형식으로
                if hasattr(sparse, "indices") and hasattr(sparse, "values"):
                    sparse_vec = qm.SparseVector(
                        indices=list(sparse.indices),
                        values=list(sparse.values),
                    )

            point_vec = {"dense": list(dense)}
            if sparse_vec is not None:
                point_vec["sparse"] = sparse_vec

            pending.append(qm.PointStruct(id=cur_id, vector=point_vec, payload=pl))
            cur_id += 1
            pbar.update(1)

            if len(pending) >= UPSERT_BATCH:
                server.upsert(SERVER_COLLECTION, points=pending)
                pending.clear()

        if not next_page:
            break

    if pending:
        server.upsert(SERVER_COLLECTION, points=pending)
        pending.clear()
    pbar.close()
    if skipped:
        print(f"[info] {name}: {skipped} 점 (필터 제외)")
    print(f"[info] {name}: ID 범위 {next_id_start} ~ {cur_id-1}")
    # local client 닫기 (메모리 해제)
    try:
        local.close()
    except Exception:
        pass
    return cur_id


def main() -> int:
    print(f"[info] Docker 서버 연결 ({SERVER_HOST}:{SERVER_PORT})")
    server = QdrantClient(host=SERVER_HOST, port=SERVER_PORT)
    try:
        server.get_collections()
    except Exception as e:
        print(f"[err] 서버 연결 실패: {e}")
        return 1

    setup_server_collection(server)

    cur_id = 0
    if server.collection_exists(SERVER_COLLECTION):
        cur_id = server.get_collection(SERVER_COLLECTION).points_count
        if cur_id > 0:
            print(f"[info] 이미 {cur_id:,} points 있음 — 처음부터 다시 시작하려면 server collection 삭제 후 재실행")

    for name, path in LOCAL_PATHS:
        cur_id = migrate_collection(name, path, server, cur_id)

    final = server.get_collection(SERVER_COLLECTION).points_count
    print(f"\n[done] 서버 {SERVER_COLLECTION}: {final:,} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
