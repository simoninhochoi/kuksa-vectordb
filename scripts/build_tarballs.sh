#!/usr/bin/env bash
# 모든 데이터셋 tar.gz를 dist/ 에 생성. 작은 것부터.
# 각 tar는 'data/' 상대 경로 보존 → 친구가 프로젝트 루트에서 풀면 됨.
#
# 큰 파일(승정원·실록·사료총서)은 시간 오래 걸림. 백그라운드 권장:
#   nohup bash scripts/build_tarballs.sh > dist/build.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p dist
LOG=dist/build.log

run() {
  local name="$1"; shift
  local out="dist/${name}.tar.gz"
  echo "[$(date +%H:%M:%S)] building $name → $out" | tee -a "$LOG"
  if [ -f "$out" ]; then
    echo "  already exists, skip ($(du -h "$out" | cut -f1))" | tee -a "$LOG"
    return 0
  fi
  tar -czf "$out" "$@" && \
    echo "  done ($(du -h "$out" | cut -f1))" | tee -a "$LOG"
}

# ───── 작은 것부터 ─────
run wongoryeo \
    data/원고려기사/raw \
    data/원고려기사/chunks.jsonl \
    data/원고려기사/entities.json \
    data/원고려기사/qdrant_storage

run goryeosa_jeolyo \
    data/고려사절요/raw \
    data/고려사절요/chunks.jsonl \
    data/고려사절요/entities.json \
    data/고려사절요/qdrant_storage

run dongmun_hwigo \
    data/동문휘고/raw \
    data/동문휘고/chunks.jsonl \
    data/동문휘고/entities.json \
    data/동문휘고/qdrant_storage

run goryeosa \
    data/고려사/raw \
    data/고려사/chunks.jsonl \
    data/고려사/entities.json \
    data/고려사/qdrant_storage

run jung_jeongsa \
    data/중국정사외국전/raw \
    data/중국정사외국전/chunks.jsonl \
    data/중국정사외국전/entities.json \
    data/중국정사외국전/qdrant_storage

# ───── 중간 ─────
run bibyeonsa \
    data/비변사등록/raw \
    data/비변사등록/chunks.jsonl \
    data/비변사등록/entities.json \
    data/비변사등록/qdrant_storage

# 한국사료총서 — 특수: data/ 직속 (서브폴더 없음)
run kuksa_history \
    data/raw \
    data/chunks.jsonl \
    data/entities.json \
    data/qdrant_storage

# ───── 큰 것 ─────
run sillok \
    data/조선왕조실록/raw \
    data/조선왕조실록/chunks.jsonl \
    data/조선왕조실록/entities.json \
    data/조선왕조실록/qdrant_storage

# 승정원일기 — Docker용 qdrant_server (qdrant_storage* 폐기 실험본은 제외)
run seungjeongwon \
    data/승정원일기/raw \
    data/승정원일기/chunks.jsonl \
    data/승정원일기/entities.json \
    data/승정원일기/qdrant_server

echo "[$(date +%H:%M:%S)] all builds complete" | tee -a "$LOG"
ls -lh dist/*.tar.gz | tee -a "$LOG"
