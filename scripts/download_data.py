"""Hugging Face Hub에서 사전 빌드된 벡터 DB tar.gz를 받아 풀어 둠.

사용:
  python scripts/download_data.py dongmun_hwigo
  python scripts/download_data.py kuksa_history sillok      # 여러 개
  python scripts/download_data.py --all
  python scripts/download_data.py --list

Private dataset 이므로 토큰 필요:
  huggingface-cli login   # 또는 HF_TOKEN 환경변수
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

# 데이터셋 ID → (HF 파일명, 추출 후 표시 경로, 사람이 읽을 라벨)
# 모든 tar.gz는 'data/' 상대 경로를 보존하므로 프로젝트 루트에서 풀면 됨.
DATASETS: dict[str, tuple[str, str, str]] = {
    "kuksa_history":      ("kuksa_history.tar.gz",   "data/qdrant_storage + chunks.jsonl",       "한국사료총서 121권 (~3GB) — file mode"),
    "sillok":             ("sillok.tar.gz",          "data/조선왕조실록/",                        "조선왕조실록 (~4GB 압축) — file mode"),
    "jung_jeongsa":       ("jung_jeongsa.tar.gz",    "data/중국정사외국전/",                      "중국정사외국전 22사 (~400MB) — file mode"),
    "dongmun_hwigo":      ("dongmun_hwigo.tar.gz",   "data/동문휘고/",                            "동문휘고 229권 (~200MB) — file mode"),
    "bibyeonsa":          ("bibyeonsa.tar.gz",       "data/비변사등록/",                          "비변사등록 (~1.2GB) — file mode"),
    "goryeosa":           ("goryeosa.tar.gz",        "data/고려사/",                              "고려사 (~350MB) — file mode"),
    "goryeosa_jeolyo":    ("goryeosa_jeolyo.tar.gz", "data/고려사절요/",                          "고려사절요 (~130MB) — file mode"),
    "wongoryeo":          ("wongoryeo.tar.gz",       "data/원고려기사/",                          "원고려기사 (~3MB) — file mode"),
    "seungjeongwon":      ("seungjeongwon.tar.gz",   "data/승정원일기/qdrant_server/",            "승정원일기 12 왕대 (~12GB 압축) ★ Docker Qdrant 필요"),
}

REPO_ID = "simoninhochoi/kuksa-vectordbs"
ROOT = Path(__file__).resolve().parents[1]


def download_one(name: str) -> None:
    if name not in DATASETS:
        print(f"[err] unknown dataset: {name}. Use --list to see available.", file=sys.stderr)
        sys.exit(2)
    fn, target_dir, label = DATASETS[name]
    print(f"[download] {name}  ({label})", flush=True)

    from huggingface_hub import hf_hub_download
    archive = hf_hub_download(
        repo_id=REPO_ID,
        filename=fn,
        repo_type="dataset",
    )
    print(f"  archive: {archive}", flush=True)

    print(f"  extracting to {ROOT / target_dir} (and parent paths inside tar) ...", flush=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(ROOT)
    print(f"[done] {name}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", help="dataset id (kuksa_history, sillok, dongmun_hwigo, ...)")
    ap.add_argument("--all", action="store_true", help="모든 데이터셋 다운로드")
    ap.add_argument("--list", action="store_true", help="사용 가능한 데이터셋 목록 출력")
    args = ap.parse_args()

    if args.list:
        print("사용 가능한 데이터셋:\n")
        for k, (fn, td, label) in DATASETS.items():
            print(f"  {k:20s} → {label}")
            print(f"  {'':20s}   tar: {fn}, target: {td}")
        return 0

    targets = list(DATASETS.keys()) if args.all else args.datasets
    if not targets:
        ap.print_help()
        return 1

    for name in targets:
        download_one(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
