"""한국사료총서 121권에서 엔티티 사전 추출.

수집 소스:
  1. <index type="이름|지명|관직|서명|관서|단체|사건|국명|연호|학교|회사조합|기타">한자</index>
  2. biblioData/creator/* 의 '이름(漢字)' 및 '漢字(한글)' 괄호 병기
  3. <alternative lang="kor">한글</alternative> ↔ 인접 <mainTitle>한자</mainTitle>

출력: data/entities.json
{
  "person": {"李沂": ["이기", "리기"], ...},
  "place":  {...},
  "title":  {...},
  ...
}
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

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "entities.json"

# DTD index type → 카테고리
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

# '한글(漢字)' 또는 '漢字(한글)' 패턴
PAREN_HANGUL_HANJA = re.compile(r"([\p{Hangul}]{1,8})\s*\(\s*([\p{Han}]{1,8})\s*\)")
PAREN_HANJA_HANGUL = re.compile(r"([\p{Han}]{1,8})\s*\(\s*([\p{Hangul}]{1,8})\s*\)")


def _norm_text(s: str | None) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", "", s).strip()


def _walk_indexes(tree: etree._ElementTree) -> list[tuple[str, str]]:
    """[(type, hanja_term), ...]"""
    out: list[tuple[str, str]] = []
    for el in tree.iter("index"):
        t = el.get("type")
        if t not in TYPE_MAP:
            continue
        # itertext: index 안에 nested가 있을 수 있으나 보통 plain text
        txt = _norm_text("".join(el.itertext()))
        if not txt or not has_hanja(txt):
            continue
        out.append((TYPE_MAP[t], txt))
    return out


def _walk_paren_pairs(tree: etree._ElementTree) -> list[tuple[str, str, str]]:
    """biblioData 안의 '한글(漢字)' / '漢字(한글)' → [(category, hanja, hangul)]
    category는 author/editor/etc → person, publisher → office 등으로 매핑.
    여기선 모두 person으로 일단 넣고, 추후 정밀 매핑 필요시 확장.
    """
    out: list[tuple[str, str, str]] = []
    for biblio in tree.iter("biblioData"):
        # creator 하위만 우선 (저자·편집자·발수신 등)
        for creator_el in biblio.iter():
            if creator_el.tag not in {
                "author", "editor", "sender", "receiver",
                "reporter", "binder", "translator", "contributor",
                "publisher",
            }:
                continue
            text = "".join(creator_el.itertext())
            cat = "office" if creator_el.tag == "publisher" else "person"
            for m in PAREN_HANGUL_HANJA.finditer(text):
                hangul, hanja_s = m.group(1), m.group(2)
                out.append((cat, hanja_s, hangul))
            for m in PAREN_HANJA_HANGUL.finditer(text):
                hanja_s, hangul = m.group(1), m.group(2)
                out.append((cat, hanja_s, hangul))
    return out


def _walk_title_pairs(tree: etree._ElementTree) -> list[tuple[str, str]]:
    """<title> 안의 mainTitle(한자) ↔ alternative lang="kor" 짝짓기."""
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


def build() -> dict[str, dict[str, list[str]]]:
    # category → hanja → set[hangul reading]
    table: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    xml_files = sorted(RAW.glob("sa_*.xml"))
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
            table[cat][hanja_s]  # ensure key exists; readings filled later

        # 2) biblioData 괄호 병기 (정답지 — corpus-extracted reading)
        for cat, hanja_s, hangul in _walk_paren_pairs(tree):
            table[cat][hanja_s].add(hangul)

        # 3) mainTitle / alternative
        for hanja_s, hangul in _walk_title_pairs(tree):
            table["title"][hanja_s].add(hangul)

    # 4) 폴백: hanja 라이브러리 음차 + 두음법칙 변형
    for cat, terms in table.items():
        for hanja_s in list(terms.keys()):
            if not terms[hanja_s]:  # corpus에서 한글 미발견
                hg = to_hangul(hanja_s)
                if has_hangul(hg):
                    terms[hanja_s].update(duum_variants(hg))
            else:
                # 정답지가 있어도 두음변형은 추가
                more: set[str] = set()
                for hg in list(terms[hanja_s]):
                    more |= duum_variants(hg)
                terms[hanja_s] |= more

    # set → sorted list
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
