"""원고려기사 zip → data/원고려기사/raw/ 추출."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "교육부 국사편찬위원회_한국사데이터베이스 정보_원고려기사_20241015.zip"
OUT = ROOT / "data" / "원고려기사" / "raw"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not ZIP.exists():
        print(f"[err] zip not found: {ZIP}", file=sys.stderr)
        return 1
    with zipfile.ZipFile(ZIP) as zf:
        members = [m for m in zf.namelist() if m.endswith(".xml") or m.endswith(".dtd")]
        print(f"[info] {len(members)} files → {OUT}")
        for m in members:
            zf.extract(m, OUT)
    print(f"[done] extracted {len(list(OUT.glob('cnwk_*.xml')))} xml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
