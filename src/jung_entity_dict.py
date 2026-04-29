"""중국정사외국전 XML에서 엔티티 사전 추출.

수집 소스:
  1. <index type="이름|지명|서명|...">한자</index>  (jo.d / jo.k 양쪽)
  2. level1 biblioData/title 의 mainTitle ↔ alternative
     (예: 史記 ↔ 譯註 中國 正史 外國傳 1 史記外國傳 譯註)
  3. 본문에 자주 나오는 '한글(漢字)' / '漢字(한글)' 괄호 병기
     (jo.k 국역에서 풍부함: '흉노(匈奴)', '신라(新羅)' 등)
  4. subjectClass schema="국가" 의 "흉노(匈奴)" 형식 → 한자/한글 매핑

출력: data/중국정사외국전/entities.json
"""
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
RAW = ROOT / "data" / "중국정사외국전" / "raw"
OUT = ROOT / "data" / "중국정사외국전" / "entities.json"

TYPE_MAP = {
    "이름": "person",
    "지명": "place",
    "서명": "title",
    "관직": "position",
    "관서": "office",
    "단체": "group",
    "사건": "event",
    "국명": "nation",
    "연호": "era",
    "기타": "other",
}

PAREN_HANGUL_HANJA = re.compile(r"([\p{Hangul}]{1,8})\s*\(\s*([\p{Han}]{1,8})\s*\)")
PAREN_HANJA_HANGUL = re.compile(r"([\p{Han}]{1,8})\s*\(\s*([\p{Hangul}]{1,8})\s*\)")


def _norm(s: str | None) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", "", s).strip()


def _walk_indexes(tree: etree._ElementTree) -> list[tuple[str, str]]:
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


def _walk_title_pairs(tree: etree._ElementTree) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for title_el in tree.iter("title"):
        main_el = title_el.find("mainTitle")
        alt_el = title_el.find("alternative")
        if main_el is None or alt_el is None:
            continue
        m = _norm("".join(main_el.itertext()))
        a = _norm("".join(alt_el.itertext()))
        if has_hanja(m) and has_hangul(a):
            out.append((m, a))
    return out


def _walk_paren_pairs_in_paragraphs(tree: etree._ElementTree) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for p in tree.iter("paragraph"):
        text = "".join(p.itertext())
        for m in PAREN_HANGUL_HANJA.finditer(text):
            hangul, hanja_s = m.group(1), m.group(2)
            out.append((hanja_s, hangul))
        for m in PAREN_HANJA_HANGUL.finditer(text):
            hanja_s, hangul = m.group(1), m.group(2)
            out.append((hanja_s, hangul))
    return out


def _walk_subject_country_pairs(tree: etree._ElementTree) -> list[tuple[str, str]]:
    """subjectClass schema='국가' 텍스트는 보통 '흉노(匈奴)' 형식."""
    out: list[tuple[str, str]] = []
    for sc in tree.iter("subjectClass"):
        if sc.get("schema") != "국가":
            continue
        txt = "".join(sc.itertext()).strip()
        m = re.match(r"^([\p{Hangul}\s]+)\s*\(\s*([\p{Han}]+)\s*\)$", txt)
        if m:
            hangul = re.sub(r"\s+", "", m.group(1))
            hanja_s = m.group(2)
            out.append((hanja_s, hangul))
    return out


def build() -> dict[str, dict[str, list[str]]]:
    table: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    xml_files = sorted(RAW.glob("jo.*.xml"))
    if not xml_files:
        print(f"[err] no XML in {RAW}", file=sys.stderr)
        sys.exit(1)

    parser = etree.XMLParser(load_dtd=False, resolve_entities=False, recover=True)

    for xf in tqdm(xml_files, desc="parsing XML"):
        try:
            tree = etree.parse(str(xf), parser)
        except etree.XMLSyntaxError as e:
            print(f"[warn] {xf.name}: {e}", file=sys.stderr)
            continue

        # 1) <index>
        for cat, hanja_s in _walk_indexes(tree):
            table[cat][hanja_s]  # ensure key exists

        # 2) mainTitle / alternative
        for hanja_s, hangul in _walk_title_pairs(tree):
            table["title"][hanja_s].add(hangul)

        # 3) 괄호 병기 — 국가/인명/지명 어디 카테고리인지 모호 → "other"에 넣고,
        #    동일 한자가 index에서 person/place로 등록돼 있으면 거기에도 보강.
        for hanja_s, hangul in _walk_paren_pairs_in_paragraphs(tree):
            table["other"][hanja_s].add(hangul)
            for cat in ("person", "place", "nation"):
                if hanja_s in table[cat]:
                    table[cat][hanja_s].add(hangul)

        # 4) subjectClass schema="국가" 의 "흉노(匈奴)" 형식 → 국가 카테고리
        for hanja_s, hangul in _walk_subject_country_pairs(tree):
            table["nation"][hanja_s].add(hangul)

    # 5) hanja 라이브러리 음차 + 두음법칙 폴백
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
    print("[stats]")
    for cat in sorted(table.keys()):
        n = len(table[cat])
        sample_keys = list(table[cat].keys())[:3]
        sample = ", ".join(f"{k}→{table[cat][k]}" for k in sample_keys)
        print(f"  {cat:10s} {n:>6d}  e.g. {sample}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
