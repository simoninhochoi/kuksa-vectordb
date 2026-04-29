"""동문휘고 XML → 엔티티 사전 (data/동문휘고/entities.json).

수집 소스:
  1. <index type="이름|지명">한자</index>  (nahf.dtd)
  2. dh.k_*.xml의 [한자(한글)] / [한글(한자)] 패턴 — 번역본의 인명·지명 병기
  3. <title>의 mainTitle(한글)·alternative(한자) 짝짓기 — 외교문서 제목

출력 카테고리: person / place / title / other
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import regex as re
from lxml import etree
from tqdm import tqdm

from hanja_util import has_hanja, has_hangul, to_hangul, duum_variants

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "동문휘고" / "raw"
OUT = ROOT / "data" / "동문휘고" / "entities.json"

TYPE_MAP = {"이름": "person", "지명": "place"}

PAREN_HANGUL_HANJA = re.compile(r"([\p{Hangul}]{1,8})\s*\(\s*([\p{Han}]{1,8})\s*\)")
PAREN_HANJA_HANGUL = re.compile(r"([\p{Han}]{1,8})\s*\(\s*([\p{Hangul}]{1,8})\s*\)")


def _norm(s: str | None) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", "", s).strip()


def _walk_indexes(tree) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for el in tree.iter("index"):
        t = el.get("type")
        if t not in TYPE_MAP:
            continue
        txt = _norm("".join(el.itertext()))
        if not txt or not has_hanja(txt):
            continue
        out.append((TYPE_MAP[t], txt))
    return out


def _walk_paren_pairs(tree) -> list[tuple[str, str]]:
    """한국어 번역본 본문의 '한자(한글)' / '한글(한자)' 인명 표기 추출."""
    out: list[tuple[str, str]] = []
    for p in tree.iter("paragraph"):
        text = "".join(p.itertext())
        for m in PAREN_HANGUL_HANJA.finditer(text):
            hg, hj = m.group(1), m.group(2)
            out.append((hj, hg))
        for m in PAREN_HANJA_HANGUL.finditer(text):
            hj, hg = m.group(1), m.group(2)
            out.append((hj, hg))
    return out


def _walk_title_pairs(tree) -> list[tuple[str, str]]:
    """level3의 <title>: alternative(한자) ↔ mainTitle(한글) 짝."""
    out: list[tuple[str, str]] = []
    for title_el in tree.iter("title"):
        main = title_el.find("mainTitle")
        alt = title_el.find("alternative")
        if main is None or alt is None:
            continue
        main_txt = _norm("".join(main.itertext()))
        alt_txt = _norm("".join(alt.itertext()))
        if has_hanja(alt_txt) and has_hangul(main_txt):
            out.append((alt_txt, main_txt))
    return out


def build() -> dict[str, dict[str, list[str]]]:
    table: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    files = sorted(RAW.glob("dh.*.xml"))
    if not files:
        print(f"[err] no XML in {RAW}", file=sys.stderr)
        sys.exit(1)
    parser = etree.XMLParser(load_dtd=False, resolve_entities=False, recover=True)
    for f in tqdm(files, desc="parsing XML"):
        try:
            tree = etree.parse(str(f), parser)
        except etree.XMLSyntaxError:
            continue
        for cat, hj in _walk_indexes(tree):
            table[cat][hj]
        for hj, hg in _walk_title_pairs(tree):
            table["title"][hj].add(hg)
        # k 번역본 본문에서 괄호 병기
        if f.name.startswith("dh.k_"):
            for hj, hg in _walk_paren_pairs(tree):
                table["person"][hj].add(hg)  # 모호하지만 인명·지명이 다수

    # hanja 음차 + 두음변형 보강
    for cat, terms in table.items():
        for hj in list(terms.keys()):
            if not terms[hj]:
                hg = to_hangul(hj)
                if has_hangul(hg):
                    terms[hj].update(duum_variants(hg))
            else:
                more: set[str] = set()
                for hg in list(terms[hj]):
                    more |= duum_variants(hg)
                terms[hj] |= more

    return {cat: {k: sorted(v) for k, v in d.items()} for cat, d in table.items()}


def main() -> int:
    table = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {OUT}")
    print("[stats]")
    for cat in sorted(table.keys()):
        n = len(table[cat])
        print(f"  {cat}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
