"""중국정사외국전 zip → data/중국정사외국전/raw/ 추출 + UTF-16 → UTF-8 변환.

원본 XML들은 UTF-16 BE BOM(\\xfe\\xff)로 시작한다. 후속 파이프라인이
파일을 그대로 lxml.parse(path)에 넘기면 BOM 처리/디코딩이 까다롭다.
한 번 추출하면서 동시에 UTF-8로 재저장한다 (XML 선언의 encoding 속성도 갱신).
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "동북아역사재단_중국정사외국전DB_20220830.zip"
OUT = ROOT / "data" / "중국정사외국전" / "raw"


def _to_utf8(path: Path) -> None:
    """UTF-16(BOM 자동 감지) → UTF-8 inplace 변환. 이미 UTF-8이면 no-op."""
    raw = path.read_bytes()
    if raw[:2] in (b"\xfe\xff", b"\xff\xfe"):
        text = raw.decode("utf-16")
    elif raw[:3] == b"\xef\xbb\xbf":
        text = raw[3:].decode("utf-8")
    else:
        # 이미 평이한 UTF-8로 가정
        try:
            text = raw.decode("utf-8")
            # XML 선언 갱신 필요 없으면 종료
            if 'encoding="UTF-16"' not in text[:200]:
                return
        except UnicodeDecodeError:
            text = raw.decode("utf-16", errors="replace")
    # XML 선언의 encoding 속성을 UTF-8로 교체 (있는 경우만)
    if text.startswith("<?xml"):
        end = text.find("?>")
        if end > 0:
            decl = text[: end + 2]
            decl = decl.replace('encoding="UTF-16"', 'encoding="UTF-8"')
            decl = decl.replace("encoding='UTF-16'", "encoding='UTF-8'")
            text = decl + text[end + 2 :]
    path.write_bytes(text.encode("utf-8"))


def main() -> int:
    if not ZIP.exists():
        # zip이 이미 풀려 있을 수도 있다. 그러면 raw/ 만 변환.
        if OUT.exists() and any(OUT.glob("jo.*.xml")):
            print(f"[info] zip not found, but raw exists: {OUT}")
        else:
            print(f"[err] zip not found: {ZIP}", file=sys.stderr)
            return 1
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP) as zf:
            members = [m for m in zf.namelist() if m.endswith(".xml")]
            print(f"[info] {len(members)} xml → {OUT}")
            for m in members:
                zf.extract(m, OUT)

    xml_files = sorted(OUT.glob("jo.*.xml"))
    print(f"[info] converting {len(xml_files)} files to UTF-8")
    for xf in xml_files:
        _to_utf8(xf)

    extracted = sorted(OUT.glob("jo.*.xml"))
    print(f"[done] {len(extracted)} xml files (UTF-8)")
    if extracted:
        d_files = [f for f in extracted if f.name.startswith("jo.d_")]
        k_files = [f for f in extracted if f.name.startswith("jo.k_")]
        print(f"[stats] jo.d (한문 원문) = {len(d_files)}, jo.k (국역) = {len(k_files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
