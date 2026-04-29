# 국편 벡터 DB 프로젝트 — 운영 가이드

국사편찬위원회 한문 사료를 BGE-M3 하이브리드 벡터 DB + MCP 서버로 노출하는 단일 프로젝트. **여러 데이터셋(한국사료총서·조선왕조실록·…)을 같은 venv·같은 청킹·임베딩·검색 파이프라인으로 처리**하되, 데이터·컬렉션·MCP 서버는 데이터셋별로 분리한다.

## 디렉터리 컨벤션

```
국편 db/
├── data/                              # 한국사료총서 (legacy default — 그대로 유지)
│   ├── raw/                           # 추출 XML
│   ├── chunks.jsonl
│   ├── entities.json
│   └── qdrant_storage/                # collection: korean_history
├── data/<데이터셋명>/                  # 신규 데이터셋(예: 조선왕조실록, 비변사등록…)
│   ├── raw/
│   ├── chunks.jsonl
│   ├── entities.json
│   └── qdrant_storage/                # collection: <데이터셋명 영문>
├── src/
│   ├── chunker.py / entity_dict.py / indexer.py / retrieval.py / mcp_server.py
│   │   └── 한국사료총서 전용 (RAW=data/raw, COLLECTION=korean_history)
│   ├── hanja_util.py                  # 공유 유틸 (한자↔한글 음차, 두음법칙)
│   ├── chunker.py 내부 _para_text_and_entities, _split_long, _merge_entities
│   │   └── 신규 chunker가 import 해서 재사용
│   ├── sillok_*.py                    # 조선왕조실록 (병렬 모듈, 데이터 격리)
│   ├── keep_awake.py                  # 장시간 작업 중 Windows 절전 차단
│   └── <new>_*.py                     # 새 데이터셋도 이 prefix 패턴
├── .mcp.json                          # 데이터셋별 MCP 서버 entry 추가
└── pyproject.toml                     # 공유 venv (Python 3.12, torch 2.6.0+cu124)
```

**원칙**: 기존 데이터셋의 모듈·데이터·컬렉션은 절대 건드리지 않는다. 신규 데이터셋은 `data/<이름>/` + `src/<prefix>_*.py` 평행 추가.

## 6단계 파이프라인 (신규 데이터셋 추가 표준 절차)

### 0. 사전 검토 (10분)

XML 샘플을 풀어 다음을 확인:

- **루트 태그**: `<level1>` / `<level2>` / `<item>` 중 무엇으로 시작하는가. 파일별로 다를 수 있음 (실록은 `_000`만 level1, 나머지는 level2-root).
- **DTD**: `history.dtd` 사용이면 `<index type="이름|지명|관직|서명|관서|단체|사건|국명|연호|학교|회사조합|기타">한자</index>` 마크업 그대로 활용 가능. 다른 DTD면 entity 추출 규칙 재설계.
- **기사(=청크) 단위**: `<biblioData type="T">`가 콘텐츠 노드. `type="L"`은 디렉터리.
- **계층**: 어느 level까지 디렉터리이고 어느 level부터가 본문 기사인가. 사료총서는 level1-3, 실록은 level1-5(層).
- **메타 필드**: 저자·연대·분류·날짜 어디에 어떻게 들어 있는가. `dateOccured type="서기"` 등 attribute 존재 여부.

### 1. 추출 (`<prefix>_extract.py`)

`사료총서/실록`처럼 `사료총서_extract.py` 또는 `sillok_extract.py` 패턴. zip 경로 지정 후 `data/<이름>/raw/`에 풀기. `sillok_extract.py`를 그대로 복사·경로 수정.

### 2. 청킹 (`<prefix>_chunker.py`)

**🔴 핵심 회피 사항 (실록에서 학습)**: 파일별로 루트 태그가 다른 데이터셋의 경우, **level2+ 루트 파일에서는 level1 메타(전체 작품 제목·왕·저자 등)가 비어 있다**. 다음 둘 중 하나로 시드 필요:
1. `_build_<prefix>_meta_map()` 함수로 level1-root 파일들을 미리 스캔해 `id → (king/author, title_hanja, title_hangul)` 매핑을 빌드
2. `chunk_file()` 진입 시 root depth가 ≥2이면 `root.get("id").split("_")[0]`로 prefix 추출해 ctx 시드

`sillok_chunker.py`의 `_build_sillok_meta_map`·`chunk_file` 패턴을 그대로 차용.

청크 헬퍼는 `chunker.py`에서 import:
```python
from chunker import MIN_CHARS, MAX_CHARS, _para_text_and_entities, _merge_entities, _split_long
```

청크 스키마는 데이터셋별로 자유롭게 확장 가능하지만, **검색 필터로 쓸 필드는 `indexer`의 `payload_index` 리스트에 반드시 등록**.

### 3. 엔티티 사전 (`<prefix>_entity_dict.py`)

`history.dtd` 사용 데이터셋이면 `sillok_entity_dict.py`를 거의 그대로 복제. 다음만 조정:
- RAW 경로
- glob 패턴 (`2nd_*.xml` / `sa_*.xml` 등)
- 본문 괄호 병기를 어디까지 신뢰할지 (사료총서는 biblioData/creator만, 실록은 paragraph도)

### 4. 인덱싱 (`<prefix>_indexer.py`)

`sillok_indexer.py`를 복제 후 다음 수정:
- `CHUNKS`, `QDRANT_PATH`, `COLLECTION` 상수
- `payload_index` 필드 목록 (필터링에 쓸 필드)
- `make_embedding_text()` — 임베딩 입력에 어떤 컨텍스트(작품명·연대·기사제목)를 prefix할지

**환경변수**:
- `INDEXER_THROTTLE=1.0` → batch 작업시간만큼 sleep, GPU 50% duty cycle (RTX 3070 Laptop 발열 조절). 0.5면 33%, 0이면 throttle 없음.
- 인덱서는 자체 `_prevent_sleep()`으로 Windows 절전 차단.
- 중단되면 `--resume`으로 마지막 안전 batch부터 재개.

성능 기준 (RTX 3070 Laptop, fp16):
- throttle 0: ~36 it/s (batch 32 chunks)
- throttle 1.0: ~18 it/s, 발열 안정. 407k 청크 = 6h 45m.

### 5. (필요시) Payload 패치

청크 스키마를 변경하거나 chunker 버그 수정 후 재인덱싱이 부담스러우면, **재임베딩 없이 payload만 갱신** 가능. 단, **chunk 순서가 보존되어야** point ID(=line N) 매핑이 유효하다.

```python
# sillok_patch_payload.py 패턴: sillok_id 단위로 묶어 set_payload 일괄 호출
client.set_payload(collection_name=..., payload={...}, points=[ids], wait=True)
```

**⚠️ Qdrant local file mode set_payload는 매우 느리다 (407k 갱신에 30분+).** 가능한 한 인덱싱 단계에서 페이로드를 정확히 만드는 것이 낫다.

### 6. 검색 + MCP 서버

`<prefix>_retrieval.py` (`sillok_retrieval.py` 복제) + `<prefix>_mcp_server.py`. 핵심:
- `expand_query()` — 한자↔한글 후보 확장 (엔티티 사전 inv index 사용)
- dense+sparse RRF → reranker → top_k
- MCP 도구 3종: `search_<name>` / `get_<name>_article` / `list_<name>_volumes`

`.mcp.json`에 entry 추가:
```json
"<name>": {
  "command": "uv",
  "args": ["run", "--directory", "c:/Users/inhoc/Projects/국편 db", "python", "src/<prefix>_mcp_server.py"],
  "env": {"PYTHONIOENCODING": "utf-8"}
}
```

Claude Code 재시작하면 자동 활성. Claude Desktop은 별도 등록 필요.

## 공통 pitfalls / 해결책

### Windows CP949 stdout에 한자 인코딩 크래시
모든 신규 스크립트 상단에:
```python
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass
```
MCP 서버는 `.mcp.json`의 `PYTHONIOENCODING=utf-8`로 처리.

### 장시간 작업 중 시스템 절전
인덱서는 자체 `_prevent_sleep()` 호출. patch·검색·기타 임의 스크립트 실행 시:
```bash
nohup uv run python src/keep_awake.py > /dev/null 2>&1 &
# 작업 끝나면 taskkill //PID <pid> //F
```
`ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED` 유지.

### `tr -u` 옵션 없음
Git Bash for Windows의 `tr`은 `-u` (unbuffered) 미지원. Monitor 명령에서 사용 금지. 대신 `python -u` (unbuffered Python)로 처리하거나 buffering 감수.

### Print buffering이 파이프 통하면 0 byte 출력
bash → tr → grep → tail 파이프라인은 process 종료 시까지 출력 안 보임. 진행 상황 모니터링 필요하면 직접 파일에 redirect 후 `tail -f`로 읽기. `print(..., flush=True)` 또는 `python -u` 권장.

### Qdrant local file mode 단일 writer 제약
인덱서가 도는 동안 같은 storage path를 다른 프로세스가 열면 락 충돌. MCP 서버는 인덱서 종료 후 기동.

### 🔴 대용량 데이터셋 (1M+ 청크) — Qdrant **Docker 서버** 사용 필수
qdrant-client **local file mode** 는 dense 벡터를 in-memory float64 numpy 배열로 보관하며 upsert 시 capacity 부족하면 `np.resize(arr, (idx*2+1, dim))` 로 doubling. 100만 청크(1024-dim float64) 시점에 약 16 GiB 할당 시도 → OOM. `VectorParams(on_disk=True)` 플래그도 doubling array 자체를 막지는 못함 (저장 위치만 영향).

또한 단일 collection 100만+ points 의 **cold start (retrieval 첫 호출)도 사실상 불가**: qdrant_client local mode 가 collection 열 때 dense vectors 를 메모리로 로드하려다 30분+ hang. 검색 자체가 사용 불가.

**기준**: 단일 collection 청크 수
- ~50만 미만: local file mode 안전 (kuksa·sillok·jung_jeongsa·dongmun·kj·kr·bb 등 모두 정상)
- ~80만~100만: 경계, 조심
- **100만 이상: Docker 서버 모드 필수**

#### Docker 서버 사용법 (sjw 케이스 표준 패턴)

**컨테이너 생성** — `--memory` 호스트 RAM 보호 + on_disk 옵션:
```bash
docker run -d \
  --name qdrant_<dataset> \
  --memory 6g --memory-swap 6g \
  -p 6333:6333 -p 6334:6334 \
  -v "$(pwd -W)/data/<dataset>/qdrant_server:/qdrant/storage" \
  qdrant/qdrant
```

**collection 생성 — `on_disk=True` 필수** (vectors + HNSW + payload 모두 디스크):
```python
client.create_collection(
    collection_name=COLLECTION,
    vectors_config={"dense": qm.VectorParams(
        size=1024, distance=qm.Distance.COSINE,
        on_disk=True,
        hnsw_config=qm.HnswConfigDiff(on_disk=True),
    )},
    sparse_vectors_config={"sparse": qm.SparseVectorParams()},
    on_disk_payload=True,
    optimizers_config=qm.OptimizersConfigDiff(
        indexing_threshold=20000,
        memmap_threshold=20000,
        default_segment_number=4,
    ),
)
```

**indexer/retrieval 코드**:
```python
client = QdrantClient(host="localhost", port=6333, timeout=60)
```

**자동 시작** — OS 재부팅 후 자동 기동:
```bash
docker update --restart unless-stopped qdrant_<dataset>
```
+ Docker Desktop 설정 "Start when you sign in" 활성화.

#### Local mode → Docker 마이그레이션 (vector 재계산 없이)

이미 local 로 인덱싱한 데이터를 **재embedding 없이** Docker server 로 옮기려면:
- qdrant_client local mode 의 `client.scroll()` 은 collection 여는 데 hang (위 한계)
- → SQLite blob 직접 추출: storage.sqlite 의 `points (id TEXT, point BLOB)` 의 BLOB 은 pickled `qdrant_client.PointStruct`. `pickle.loads(blob)` 로 그대로 복원 → server 에 upsert.
- 200만 points 약 1시간 30분 (호스트 RAM 1-3 GB 이내 안정).
- 표준 구현: `src/sjw_migrate_sqlite_direct.py` 참조.

#### 호스트 RAM 한계

Docker Desktop on Windows: WSL2 한도가 호스트 RAM 의 ~50% (.wslconfig 로 조정 가능). 16 GB 호스트면 7.5 GB 한도 기본. 호스트 free RAM 1-2 GB 만 남으면 swap 폭주 → OS freeze 위험. 컨테이너 `--memory` 명시 + collection on_disk 둘 다 필요.

### `expand_query` 비대화 (한글 질의)
한글 토큰 → 한자 후보 확장이 너무 많으면 reranker가 느려진다. `max_expansions` 기본 30. 필요시 줄이기.

### Chunk 순서가 보존되지 않으면 patch 불가
`sorted(RAW.glob(...))`을 따르고, 동일 파일 내에서는 DOM iteration 순서 그대로. chunker 변경 시 chunk_id at line N이 동일한지 sample check 필수 (백업본과 비교).

### XML 인코딩 — UTF-16 vs UTF-8 혼재
국편 데이터셋(한국사료총서·실록)은 UTF-8, 동북아역사재단 데이터셋(동문휘고·중국정사외국전)은 **UTF-16**. lxml은 자동 처리하지만 `cat`·`head`·`grep`으로 직접 읽으면 깨짐 — 파싱은 lxml로, 인코딩 확인은 `head -c 50 file.xml` 후 `<?xml encoding=...?>` 선언 보고 판단.

### 평행 버전 (d/m/k) 데이터셋 처리
동북아역사재단 데이터셋은 같은 권에 대해 한자원본(`*.d_*`) / 한자표점본(`*.m_*`) / 한국어번역본(`*.k_*`) 평행 버전이 존재할 수 있음. 청킹 시 **m > d 우선으로 한자 본문**을 사용하고 **k는 첫 슬라이스의 `text_translation_korean` 필드**에 부가. 한 권에 m이 없으면 d만 사용. 같은 level3 ID suffix로 매핑.

### 청킹 단위 사전 테스트 패턴
신규 데이터셋의 XML 구조를 처음 보면 청킹 단위가 비자명함. 정식 chunker 작성 전에 **`<dataset>_test_chunking.py`** 류의 비교 스크립트로 두세 전략을 같이 돌려 길이 분포·메타 풍부도·문자 분포를 비교하고 결정. 동문휘고에서는 `level3 단위 (외교문서 1건)` vs `paragraph 단위`를 비교했고, level3가 압도적 — 이 패턴을 신규 데이터셋에도 적용.

### 신규 데이터셋의 DTD가 다르면 index type 매핑 재확인
- `history.dtd` (사료총서·실록): 이름·지명·관직·서명·관서·단체·사건·국명·연호·학교·회사조합·기타 12종
- `nahf.dtd` (동문휘고·중국정사외국전): **이름·지명만 2종**. entity_dict의 TYPE_MAP을 데이터셋별로 조정.

### 외교문서·연표 메타 — 발수신자·날짜
동문휘고 같은 외교문서는 `<creator><sender>`·`<creator><receiver>` + `<date><dateSend date="YYYY-MM-DDL0">` 구조. 일반 사료의 `<author>`·`<dateOccured>`와 다름. **신규 데이터셋의 `front/biblioData` 자식들을 inspect2.py 류로 먼저 확인**하고 chunker 메타 추출 로직을 맞춰야 함.

## 데이터셋별 컬렉션 — 현황

| 데이터셋 | sid prefix | data path | collection | MCP server | 청크 수 |
|---|---|---|---|---|---|
| 한국사료총서 121권 | `sa_*` | `data/` | `korean_history` | `kuksa-history` | 299,423 |
| 조선왕조실록 (2nd_*) | `2nd_*` (sid: waa~wya) | `data/조선왕조실록/` | `sillok` | `sillok` | 407,576 |
| 중국정사외국전 22사 | `jo.d/jo.k` (UTF-16) | `data/중국정사외국전/` | `jung_jeongsa` | `jung-jeongsa` | 18,169 |
| 동문휘고 229권 | `dh.d/dh.m/dh.k` (UTF-16) | `data/동문휘고/` | `dongmun_hwigo` | `dongmun-hwigo` | 17,367 |
| 원고려기사 | `cnwk_*` | `data/원고려기사/` | `wongoryeo_gisa` | `wongoryeo` | 196 |
| 고려사절요 36권 | `kj_*` | `data/고려사절요/` | `goryeosa_jeolyo` | `goryeosa-jeolyo` | 11,365 |
| 고려사 139권 | `kr_*` | `data/고려사/` | `goryeosa` | `goryeosa` | 31,885 |
| 비변사등록 274권 | `bb_*` | `data/비변사등록/` | `bibyeonsa_deungrok` | `bibyeonsa` | 103,028 |
| 승정원일기 ★ Docker server | `2nd_*.sjw.y` | `data/승정원일기/qdrant_server/` (container) | `sjw_full` (Docker) | `seungjeongwon` | 2,051,477 |

★ 승정원일기는 200만 청크 단일 collection 100만 한계 회피를 위해 **Docker Qdrant 서버** 모드 채택 (위 「대용량 데이터셋」 섹션 참조). container `qdrant_sjw` (port 6333), on_disk vectors + HNSW. local mode 인덱싱 → SQLite 직접 추출 마이그레이션 완료. local storage (`qdrant_storage`, `qdrant_storage_sjw2`, `qdrant_storage_sjw3` 합 26 GB)는 백업으로 보존.

## venv·의존성 운영

`pyproject.toml`이 단일 권위. 신규 데이터셋이 새 의존성을 요구하면 여기에 추가하고 `uv lock --upgrade`. **transformers는 4.56.x 고정** (FlagEmbedding 1.3.4 호환 윈도우), **torch는 `2.6.0+cu124`** (CPU 빌드로 침식 방지).

## 메모리 (Claude 자동기억)

- `~/.claude/projects/c--Users-inhoc-Projects/memory/project_kuksa_vectordb.md` — 한국사료총서 운영 메모
- `~/.claude/projects/c--Users-inhoc-Projects/memory/project_sillok_vectordb.md` — 조선왕조실록 운영 메모

신규 데이터셋 추가 시 동일 디렉터리에 `project_<name>_vectordb.md`를 작성하고 `MEMORY.md` 인덱스에 한 줄 추가.
