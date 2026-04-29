#!/usr/bin/env bash
# Qdrant 컨테이너 기동. 데이터는 ./data/qdrant_storage 에 영속.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STORAGE="$ROOT/data/qdrant_storage"
mkdir -p "$STORAGE"

CONTAINER=kuksa-qdrant

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  echo "[info] $CONTAINER already running"
  exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  echo "[info] starting existing $CONTAINER"
  docker start "$CONTAINER"
  exit 0
fi

echo "[info] creating $CONTAINER (storage: $STORAGE)"
docker run -d \
  --name "$CONTAINER" \
  -p 6333:6333 -p 6334:6334 \
  -v "$STORAGE:/qdrant/storage" \
  qdrant/qdrant:latest
