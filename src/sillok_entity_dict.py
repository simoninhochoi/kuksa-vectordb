"""조선왕조실록 XML에서 엔티티 사전 추출.

수집 소스:
  1. <index type="이름|지명|관직|서명|관서|단체|사건|국명|연호|학교|회사조합|기타">한자</index>
  2. level1 <biblioData><title> 의 mainTitle(한자) ↔ alternative(한글) 짝짓기
     (예: 太祖實錄 ↔ 태조실록)
  3. 기사 본문에 나오는 '한글(漢字)' / '漢字(한글)' 괄호 병기
     (실록 본문에는 드물지만 편수관 명단 등에 존재)

출력: data/조선왕조실록/entities.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

# Windows CP949 stdout에 한자 출력 시 크래시 방지
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
RAW = ROOT / "data" / "조선왕조실록" / "raw"
OUT = ROOT / "data" / "조선왕조실록" / "entities.json"

TYPE_MAP = {
    "이름": "person",
    "지명": "place",
    "관직": "position",
    "서명": "title",
    "관서": "office",
    "단체": "group",
    "사건": "event",
    "국명": "nation",
    "연호": "era",
    "학교": "school",
    "회사조합": "company",
    "기타": "other",
}

PAREN_HANGUL_HANJA = re.compile(r"([\p{Hangul}]{1,8})\s*\(\s*([\p{Han}]{1,8})\s*\)")
PAREN_HANJA_HANGUL = re.compile(r"([\p{Han}]{1,8})\s*\(\s*([\p{Hangul}]{1,8})\s*\)")


def _norm_text(s: str | None) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", "", s).strip()


def _walk_indexes(tree: etree._ElementTree) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for el in tree.iter("index"):
        t = el.get("type")
        if t not in TYPE_MAP:
            continue
        txt = _norm_text("".join(el.itertext()))
        if not txt or not has_hanja(txt):
            continue
        out.append((TYPE_MAP[t], txt))
    return out


def _walk_title_pairs(tree: etree._ElementTree) -> list[tuple[str, str]]:
    """<title>의 mainTitle ↔ alternative 짝."""
    out: list[tuple[str, str]] = []
    for title_el in tree.iter("title"):
        main_el = title_el.find("mainTitle")
        alt_el = title_el.find("alternative")
        if main_el is None or alt_el is None:
            continue
        main_txt = _norm_text("".join(main_el.itertext()))
        alt_txt = _norm_text("".join(alt_el.itertext()))
        if has_hanja(main_txt) and has_hangul(alt_txt):
            out.append((main_txt, alt_txt))
    return out


def _walk_paren_pairs_in_paragraphs(tree: etree._ElementTree) -> list[tuple[str, str]]:
    """paragraph 본문에서 '한자(한글)' / '한글(한자)' 패턴 스캔.
    실록 기사에는 드물지만 부록·편수관 명단 등에 간혹 등장.
    """
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


def build() -> dict[str, dict[str, list[str]]]:
    table: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    xml_files = sorted(RAW.glob("2nd_*.xml"))
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

        # 1) <index> 태그
        for cat, hanja_s in _walk_indexes(tree):
            table[cat][hanja_s]  # ensure key exists

        # 2) mainTitle / alternative (실록명: 太祖實錄 ↔ 태조실록)
        for hanja_s, hangul in _walk_title_pairs(tree):
            table["title"][hanja_s].add(hangul)

        # 3) 본문 괄호 병기 (카테고리 불명 → 임시로 "other"에 넣지 말고
        #    person/place에 각각 후보로 추가하는 것은 오염 위험 → title 범주에 귀속)
        for hanja_s, hangul in _walk_paren_pairs_in_paragraphs(tree):
            # 실록 본문에서 나온 괄호 병기는 모호하므로 "other" 범주
            table["other"][hanja_s].add(hangul)

    # 4) hanja 라이브러리 음차 + 두음법칙 폴백
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
