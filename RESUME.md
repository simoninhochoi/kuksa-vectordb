# 승정원일기 인덱싱 — Docker Qdrant 서버 단일 collection (최종)

## 최종 아키텍처

**single Docker Qdrant collection `sjw_full`** (port 6333, container `qdrant_sjw`).
- 2,051,477 points (chunks.jsonl 정확 일치)
- vectors `on_disk=True` (memmap), HNSW `on_disk=True`
- payload `on_disk_payload=True`
- container memory limit 6 GB (호스트 15.4 GB 보호)
- storage: `data/승정원일기/qdrant_server/` (~5-6 GB)

## Docker 컨테이너 시작/종료/상태

```bash
# 시작 (기존 컨테이너)
docker start qdrant_sjw

# 상태 확인
docker ps
docker stats qdrant_sjw --no-stream
curl -s http://localhost:6333/collections/sjw_full

# 종료
docker stop qdrant_sjw

# 메모리 사용 모니터
docker stats qdrant_sjw
```

OS 재부팅 후 자동 시작하려면 Docker Desktop 설정에서 "Start Docker Desktop when you sign in" 활성화 + 컨테이너 restart policy 추가:
```bash
docker update --restart unless-stopped qdrant_sjw
```

## 이전 시도 히스토리 (참고용)

### 1차 시도 — qdrant_client local mode 단일 collection
- 200만 청크 시도. 1.26M 시점에 `numpy._ArrayMemoryError: 28.9 GiB` 크래시.
- 원인: qdrant local mode는 dense vector를 in-memory float64 numpy 배열로 보관, upsert 시 doubling.

### 2차 시도 — local mode 3-way split (sjw + sjw2 + sjw3)
- 인덱싱 자체는 성공했지만 retrieval **cold start가 30분+** (각 collection 16/8/1.5 GB SQLite 메모리 로드)
- 실제 검색 사용 불가능

### 3차 시도 — Docker server (in-memory vectors)
- migrate 진행 중 호스트 RAM 압박 → swap 폭주 → **OS freeze 발생** (사용자 강제 재부팅)
- 원인: Docker Desktop 한도 7.5 GB, 200만 vectors는 8+ GB 필요

### 4차 시도 — Docker server + on_disk vectors ✅ 성공
- collection 생성 시 `on_disk=True` (vectors + HNSW), `on_disk_payload=True`, `memmap_threshold=20000`, container `--memory 6g`
- SQLite 직접 추출 마이그레이션 (qdrant_client local mode bypass) — `src/sjw_migrate_sqlite_direct.py`
- 약 1시간 30분 소요. MEM 항상 1-3 GB 이내 유지 (이전 7.5 GB 초과 freeze 대비)

## 검색 동작 검증 (smoke test)

| 시기 | 질의 | 최고 score | 시기 일치 |
|---|---|---:|:---:|
| 인조 (1623) | "인조가 청나라 사신을 접견하다" | 0.951 | ✅ 인조 1623 사신접견 |
| 철종 (1851~1862) | "철종이 동학을 우려하다" | 0.153 | ✅ 철종 시기 (동학은 직접 언급 적음) |
| 고종 갑오개혁 (1894) | "갑오개혁 군국기무처" | 0.122 | ✅ **고종 31년(1894) 군국기무처 설치** 정확히 hit |

cold start 응답 즉시 (이전 local 3-way split 시 30분+ 대비).

## 주요 코드 변경

- `src/sjw_retrieval.py`: 3-way local merge → **Docker server 단일 collection**으로 전면 재작성
- `src/sjw_migrate_sqlite_direct.py`: SQLite blob 직접 unpickle → server 마이그레이션 (qdrant_client local mode bypass)
- `src/sjw_indexer.py`, `sjw2_indexer.py`, `sjw3_indexer.py`: 더 이상 사용 안 함 (참고용 보존)

## local mode storage (백업 가능)

기존 local collection 들은 그대로 보존:
- `data/승정원일기/qdrant_storage/` (sjw, 16.7 GB)
- `data/승정원일기/qdrant_storage_sjw2/` (sjw2, 7.9 GB)
- `data/승정원일기/qdrant_storage_sjw3/` (sjw3, 1.5 GB)

총 26 GB. Docker server에 모두 마이그레이션 완료됐으므로 디스크 회수 원하면 삭제 가능. 단 Docker storage는 백업 안 됐으니 적어도 한 번은 유지 권장. 또는 외부 디스크에 백업 후 삭제.

## MCP 서버 (변경 없음)

`.mcp.json`의 `seungjeongwon` entry 그대로 — `sjw_mcp_server.py`가 변경된 `sjw_retrieval.py` 사용. Claude Code 재시작 시 MCP 서버 자동 활성, 즉시 검색 가능.

**전제 조건**: Docker container `qdrant_sjw` 실행 중이어야 함. 컨테이너 죽어 있으면 검색 시 `ConnectionError`. `docker start qdrant_sjw`로 시작.

## 다른 데이터셋 (참고)

| 데이터셋 | 청크 | 모드 | MCP server |
|---|---:|---|---|
| 원고려기사 | 196 | local | wongoryeo |
| 고려사절요 | 11,365 | local | goryeosa-jeolyo |
| 고려사 | 31,885 | local | goryeosa |
| 비변사등록 | 103,028 | local | bibyeonsa |
| **승정원일기** | **2,051,477** | **Docker** | seungjeongwon |
