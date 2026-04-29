"""조선왕조실록 zip → data/조선왕조실록/raw/ 추출."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "교육부 국사편찬위원회_조선왕조실록 정보_실록원문_20221103.zip"
OUT = ROOT / "data" / "조선왕조실록" / "raw"


def main() -> int:
    if not ZIP.exists():
        print(f"[err] zip not found: {ZIP}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP) as zf:
        members = zf.namelist()
        xml_members = [m for m in members if m.endswith(".xml")]
        dtd_members = [m for m in members if m.endswith(".dtd")]
        print(f"[info] {len(xml_members)} xml + {len(dtd_members)} dtd → {OUT}")
        for m in xml_members + dtd_members:
            zf.extract(m, OUT)

    extracted = sorted(OUT.glob("2nd_*.xml"))
    print(f"[done] extracted {len(extracted)} xml files")
    if extracted:
        print(f"[range] {extracted[0].name} ... {extracted[-1].name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
