"""원고려기사 엔티티 사전. data/원고려기사/entities.json."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

import regex as re
from lxml import etree
from tqdm import tqdm

from hanja_util import has_hanja, has_hangul, to_hangul, duum_variants

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "원고려기사" / "raw"
OUT = ROOT / "data" / "원고려기사" / "entities.json"

TYPE_MAP = {
    "이름": "person", "지명": "place", "관직": "position",
    "서명": "title", "관서": "office", "단체": "group",
    "사건": "event", "국명": "nation", "연호": "era",
    "학교": "school", "회사조합": "company", "기타": "other",
}

PAREN_HANGUL_HANJA = re.compile(r"([\p{Hangul}]{1,8})\s*\(\s*([\p{Han}]{1,8})\s*\)")
PAREN_HANJA_HANGUL = re.compile(r"([\p{Han}]{1,8})\s*\(\s*([\p{Hangul}]{1,8})\s*\)")


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", "", s or "").strip()


def build() -> dict[str, dict[str, list[str]]]:
    table: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    parser = etree.XMLParser(load_dtd=False, resolve_entities=False, recover=True)
    xml_files = sorted(RAW.glob("cnwk_*.xml"))
    for xf in tqdm(xml_files, desc="parse"):
        try:
            tree = etree.parse(str(xf), parser)
        except etree.XMLSyntaxError as e:
            print(f"[warn] {xf.name}: {e}", file=sys.stderr)
            continue
        for el in tree.iter("index"):
            t = el.get("type")
            if t not in TYPE_MAP:
                continue
            txt = _norm("".join(el.itertext()))
            if txt and has_hanja(txt):
                table[TYPE_MAP[t]][txt]
        for title_el in tree.iter("title"):
            main_el = title_el.find("mainTitle")
            alt_el = title_el.find("alternative")
            if main_el is None or alt_el is None:
                continue
            m = _norm("".join(main_el.itertext()))
            a = _norm("".join(alt_el.itertext()))
            if has_hanja(m) and has_hangul(a):
                table["title"][m].add(a)
        for p in tree.iter("paragraph"):
            text = "".join(p.itertext())
            for mm in PAREN_HANGUL_HANJA.finditer(text):
                table["other"][mm.group(2)].add(mm.group(1))
            for mm in PAREN_HANJA_HANGUL.finditer(text):
                table["other"][mm.group(1)].add(mm.group(2))

    for cat, terms in table.items():
        for hanja_s in list(terms.keys()):
            if not terms[hanja_s]:
                hg = to_hangul(hanja_s)
                if has_hangul(hg):
                    terms[hanja_s].update(duum_variants(hg))
            else:
                more: set[str] = set()
                for hg in list(terms[hanja_s]):
                    more |= duum_variants(hg)
                terms[hanja_s] |= more
    return {cat: {k: sorted(v) for k, v in d.items()} for cat, d in table.items()}


def main() -> int:
    table = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {OUT}")
    for cat in sorted(table.keys()):
        print(f"  {cat:10s} {len(table[cat]):>5d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
