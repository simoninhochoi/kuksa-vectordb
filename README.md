# 국편 한국사 벡터 DB 모음

국사편찬위원회·동북아역사재단·한국고전번역원의 한국사 사료(漢文 원전 + 한국어 번역)를 BGE-M3 dense+sparse 하이브리드 + reranker로 검색하는 로컬 벡터 DB와 MCP 서버 모음. **모든 사료를 같은 venv·같은 엔진·하나의 `.mcp.json`** 에서 통합 운용.

## 수록 데이터셋

| 데이터셋 | MCP 서버 | 청크 수 | 데이터 출처 |
|---|---|---|---|
| 한국사료총서 121권 | `kuksa-history` | 299,423 | 국사편찬위원회 |
| 조선왕조실록 (太祖~哲宗) | `sillok` | 407,576 | 국사편찬위원회 |
| 중국정사외국전 22사 | `jung-jeongsa` | ~18,000 | 동북아역사재단 |
| 동문휘고 229권 | `dongmun-hwigo` | 17,367 | 동북아역사재단 |
| 비변사등록 274권 | `bibyeonsa` | ~103,000 | 국사편찬위원회 |
| 고려사 139편 | `goryeosa` | 31,885 | 국사편찬위원회 |
| 고려사절요 36권 | `goryeosa-jeolyo` | 11,365 | 국사편찬위원회 |
| 원고려기사 | `wongoryeo` | 196 | 동북아역사재단 |
| 승정원일기 297 XML (12 왕대) | `seungjeongwon` | ~2,000,000 | 한국고전번역원 |

## 빠른 시작

### 1. 코드·환경 준비

```bash
git clone https://github.com/simoninhochoi/kuksa-vectordb.git
cd kuksa-vectordb
uv sync                          # Python 3.12, BGE-M3, Qdrant 등 설치
```

### 2. 데이터 다운로드 (Hugging Face)

데이터는 별도 private dataset으로 호스팅. 토큰 발급 후 (`huggingface.co/settings/tokens`):

```bash
huggingface-cli login
python scripts/download_data.py dongmun_hwigo    # 단일 데이터셋
python scripts/download_data.py --all            # 모두 (~50GB, 시간 오래 걸림)
python scripts/download_data.py --list           # 사용 가능한 데이터셋 목록
```

### 3. MCP 서버 활성

`.mcp.json`이 이미 9개 서버를 모두 등록해 놓음. Claude Code 또는 Claude Desktop에서:

```bash
# Claude Code: 프로젝트 디렉토리에서 시작하면 자동 활성
cd kuksa-vectordb
claude

# Claude Desktop: ~/.config/claude/claude_desktop_config.json 에 .mcp.json 내용 복사
```

각 데이터셋의 MCP 도구는 `search_<name>` / `get_<name>_article` / `list_<name>_volumes` 패턴 (예: `search_sillok`, `get_dongmun_document`).

### 4. 승정원일기만 — Docker Qdrant 추가 설정

승정원일기는 200만 청크 규모라 Qdrant local file mode가 비효율적. **Docker Qdrant 서버**를 별도 가동:

```bash
# 압축 풀린 데이터 디렉토리를 마운트해 Docker Qdrant 시작
docker run -d \
  --name qdrant_sjw \
  -p 6333:6333 \
  -v "$(pwd)/data/승정원일기/qdrant_server:/qdrant/storage" \
  --memory=6g \
  --restart unless-stopped \
  qdrant/qdrant

# 상태 확인
curl -s http://localhost:6333/collections/sjw_full | python -m json.tool
# 점수 ≈ 2,051,477 points 면 정상

# 향후 (재부팅 후 등)
docker start qdrant_sjw
```

`sjw_mcp_server`는 `localhost:6333`에 자동 연결. 다른 데이터셋은 Docker 불필요.

## 시스템 요구사항

- Windows / macOS / Linux. 대용량 디스크 (전체 ~60 GB 권장)
- **NVIDIA GPU 권장** (RTX 3070 이상). CPU 모드도 가능하지만 ~50배 느림
- Python 3.12 (3.13 미지원 — FlagEmbedding 호환성)
- 인덱스를 직접 다시 만들려면 16GB+ VRAM 권장

## 디렉토리 구조

```
kuksa-vectordb/
├── README.md
├── CLAUDE.md                        # 프로젝트 운영·확장 가이드
├── pyproject.toml                   # uv lock 포함
├── .mcp.json                        # 9개 MCP 서버 등록
├── scripts/
│   └── download_data.py             # HF Hub 다운로더
└── src/
    ├── chunker.py / entity_dict.py / indexer.py / retrieval.py / mcp_server.py    # 한국사료총서
    ├── sillok_*.py                  # 조선왕조실록
    ├── dh_*.py                      # 동문휘고
    ├── bb_*.py                      # 비변사등록
    ├── kr_*.py                      # 고려사
    ├── kj_*.py                      # 고려사절요
    ├── cnwk_*.py                    # 원고려기사
    ├── jo_*.py                      # 중국정사외국전
    ├── sjw_*.py                     # 승정원일기
    ├── hanja_util.py                # 공유: 한자↔한글 음차, 두음법칙
    └── keep_awake.py                # Windows 절전 차단
```

## 새 데이터셋 추가

[CLAUDE.md](CLAUDE.md) 참조. 6단계 파이프라인 (extract → chunk → entity_dict → index → retrieval → mcp_server)을 따르고 운영 권한·인코딩·청킹 단위에 대한 학습 사항을 정리해 놓았습니다.

## 라이선스 / 출처 표시

본 레포 코드: MIT 라이선스 ([LICENSE](LICENSE) 참조).

수록 데이터의 1차 사료 저작권:
- 국사편찬위원회 (한국사료총서·조선왕조실록·비변사등록·고려사·고려사절요): **공공누리 제4유형**(출처표시·비영리·변경금지)
- 동북아역사재단 (동문휘고·중국정사외국전·원고려기사): **공공누리 제4유형**
- 한국고전번역원 (승정원일기): 해당 기관 정책에 따름

본 데이터셋(청크·임베딩·Qdrant 인덱스)은 위 1차 사료를 학술 검색 목적으로 가공한 파생물입니다. **재배포·상업적 이용 시에는 반드시 원 저작권사 동의를 별도 확인**하세요. 사료 원문(raw XML)은 본 패키지에 포함되어 있지 않으며 각 기관 사이트에서 직접 받아야 합니다:
- [국사편찬위원회 한국사데이터베이스](https://db.history.go.kr/)
- [동북아역사재단 한국사데이터베이스](https://contents.nahf.or.kr/)
- [한국고전번역원 한국고전종합DB](https://db.itkc.or.kr/)

## 인용

```
@misc{kuksa-vectordb-2026,
  author = {Choi, Inho},
  title  = {Kuksa Korean History Vector DB Collection},
  year   = {2026},
  url    = {https://github.com/simoninhochoi/kuksa-vectordb}
}
```
