# 배포 체크리스트

## 1. GitHub 레포 생성 + 푸시

`gh` CLI는 이미 설치됨. 미인증 상태면 먼저:
```bash
gh auth login
# 옵션: GitHub.com → HTTPS → Login with web browser
```

레포 생성 + 푸시 (현재 디렉토리에서):
```bash
cd "c:/Users/inhoc/Projects/국편 db"

# public이면 누구나 코드 클론 가능, private면 friend collaborator 초대 필요
gh repo create simoninhochoi/kuksa-vectordb \
  --public \
  --source=. \
  --description "BGE-M3 hybrid vector DBs for Korean historical sources (조선왕조실록, 승정원일기, 동문휘고, 사료총서 외)" \
  --push
```

`--private`로 만들고 싶으면 `--public` 자리만 바꾸면 됨.

푸시 후 동시에 친구들 collaborator 초대 (private인 경우):
```bash
gh repo edit --add-collaborator <친구_github_id>
# 여러 명이면 반복
```

## 2. Hugging Face Datasets — private 업로드

### 토큰 발급 + 로그인
1. https://huggingface.co/settings/tokens 에서 **write** 권한 토큰 발급
2. 로그인:
```bash
pip install huggingface_hub
huggingface-cli login
# 토큰 붙여넣기
```

### 데이터셋 repo 생성 (private)
```bash
huggingface-cli repo create kuksa-vectordbs \
  --type dataset \
  --private
# → huggingface.co/datasets/cjoseph509/kuksa-vectordbs (private)
```

### tar.gz 업로드 — 작은 것부터
빌드 끝난 tarball들을 dist/에서 업로드:
```bash
cd "c:/Users/inhoc/Projects/국편 db"

# 작은 것 5개 일괄 (한 번에)
for f in wongoryeo goryeosa_jeolyo dongmun_hwigo goryeosa jung_jeongsa; do
  echo "==== uploading $f ===="
  huggingface-cli upload cjoseph509/kuksa-vectordbs \
    "dist/${f}.tar.gz" "${f}.tar.gz" \
    --repo-type dataset
done

# 중간/큰 것 (각각 시간 더 걸림)
huggingface-cli upload cjoseph509/kuksa-vectordbs dist/bibyeonsa.tar.gz bibyeonsa.tar.gz --repo-type dataset
huggingface-cli upload cjoseph509/kuksa-vectordbs dist/kuksa_history.tar.gz kuksa_history.tar.gz --repo-type dataset
huggingface-cli upload cjoseph509/kuksa-vectordbs dist/sillok.tar.gz sillok.tar.gz --repo-type dataset

# 승정원일기 (가장 큰, 16GB+ 가능)
huggingface-cli upload cjoseph509/kuksa-vectordbs dist/seungjeongwon.tar.gz seungjeongwon.tar.gz --repo-type dataset
```

업로드 도중 끊기면 같은 명령 재실행 — `huggingface-cli upload`는 chunk-resume 지원.

### Dataset README 작성
```bash
cat > /tmp/dataset_README.md <<'EOF'
# Kuksa Korean History Vector DBs

Pre-built BGE-M3 dense+sparse hybrid vector DBs for 9 Korean historical
source collections. Use with the code at:
https://github.com/simoninhochoi/kuksa-vectordb

## Datasets

| File | Size | Records | Source |
|---|---|---|---|
| `wongoryeo.tar.gz` | ~3MB | 196 chunks | 동북아역사재단 |
| `goryeosa_jeolyo.tar.gz` | ~130MB | 11k | 국편위 |
| `dongmun_hwigo.tar.gz` | ~200MB | 17k | 동북아역사재단 |
| `goryeosa.tar.gz` | ~350MB | 32k | 국편위 |
| `jung_jeongsa.tar.gz` | ~400MB | 18k | 동북아역사재단 |
| `bibyeonsa.tar.gz` | ~1.2GB | 103k | 국편위 |
| `kuksa_history.tar.gz` | ~3GB | 299k | 국편위 |
| `sillok.tar.gz` | ~4GB | 408k | 국편위 |
| `seungjeongwon.tar.gz` | ~12-15GB | 2M | 한국고전번역원 |

## Usage
```python
python scripts/download_data.py dongmun_hwigo
```
See main repo README for details.

## License
Primary sources: 공공누리 4유형 (each issuer policy applies).
Vector DB derivative: friend distribution only (private).
EOF

huggingface-cli upload cjoseph509/kuksa-vectordbs /tmp/dataset_README.md README.md --repo-type dataset
```

### 친구 초대 (private dataset)
HF 웹: https://huggingface.co/datasets/cjoseph509/kuksa-vectordbs/settings → "Members" → 친구 HF 계정 추가

## 3. 친구 측 사용법 (검증 절차)

친구가 코드 + 데이터 받는 풀 시퀀스:
```bash
# 코드
git clone https://github.com/simoninhochoi/kuksa-vectordb.git
cd kuksa-vectordb

# 환경
uv sync

# 데이터 (HF 토큰 필요)
huggingface-cli login

# 작은 것 먼저 시험
python scripts/download_data.py dongmun_hwigo

# 검색 테스트
uv run python src/dh_retrieval.py "건륭 만수절"

# 큰 것 (필요시)
python scripts/download_data.py sillok
python scripts/download_data.py seungjeongwon  # ★ 12-15GB
```

## 4. 라이선스 주의

- **public github 코드**: MIT, 누구나 OK
- **private HF dataset**: friend 한정, 학술 비영리만. 외부에 재배포 금지 명시.
- 라이선스 위반 우려시 GitHub 코드 repo도 private로 → 친구만 collaborator 초대.
