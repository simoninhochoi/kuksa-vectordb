"""승정원일기 (承政院日記) → 기사 단위 청크.

XML 297개. ID prefix(SJW-X)별 왕:
  A=인조 B=효종 C=현종 D=숙종 E=경종 F=영조 G=정조
  H=순조 I=헌종 J=철종 K=고종 L=순종

계층: level1=왕(SJW-X) → level2=연 → level3=월 → level4=일 → level5=개별 기사 (leaf)
2nd_A01만 level1 root(item)이고 나머지는 level2 root → king 메타 외부 매핑 필요.

leaf article = level5 (또는 드물게 level4 with content). 한 leaf 1건 = 1 청크.

산출물: data/승정원일기/chunks.jsonl
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

import regex as re
from lxml import etree
from tqdm import tqdm

from chunker import MAX_CHARS, _para_text_and_entities, _merge_entities, _split_long
from hanja_util import to_hangul

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "승정원일기" / "raw"
OUT = ROOT / "data" / "승정원일기" / "chunks.jsonl"

KING_MAP = {
    "A": "인조", "B": "효종", "C": "현종", "D": "숙종", "E": "경종",
    "F": "영조", "G": "정조", "H": "순조", "I": "헌종", "J": "철종",
    "K": "고종", "L": "순종",
}


@dataclass
class SjwChunk:
    chunk_id: str
    article_id: str
    book_id: str            # "sjw"
    king_prefix: str        # "A".."L"
    king: str               # 인조 등
    year_id: str            # SJW-A01 (level2)
    year_label: str         # "승정원일기 인조01년"
    year_ce: str            # 1623
    month_id: str           # SJW-A01030
    day_id: str             # SJW-A01030120
    day_title: str          # "승정원일기 인조01년 03월 12일"
    date_western: str       # 1623-03-12
    ganji: str              # 임인
    reign_year: str         # 인조 01-03-12
    article_type: str       # 좌목/기사 등
    article_title: str
    text: str
    text_hangul_aux: str
    entities: dict[str, list[str]] = field(default_factory=dict)


def _txt(el):
    if el is None: return ""
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _parse_dates(bib):
    """biblioData/date → (date_western, ganji, reign_year, year_ce)."""
    if bib is None:
        return "", "", "", ""
    date_el = bib.find("date")
    if date_el is None:
        return "", "", "", ""
    ymd = ""; ganji = ""; reign = ""; year_ce = ""
    for d in date_el.findall("dateOccured"):
        t = d.get("type")
        raw = (d.get("date") or "").strip()
        if t == "서기":
            m = re.match(r"^(\d{4}(?:-\d{2}(?:-\d{2})?)?)(?:L[01])?$", raw)
            if m:
                v = m.group(1)
                if len(v) == 4: year_ce = v
                else: ymd = v; year_ce = v[:4]
        elif t == "간지":
            ganji = _txt(d)
        elif t == "재위연도":
            reign = _txt(d)
    return ymd, ganji, reign, year_ce


def chunk_file(xml_path: Path) -> list[SjwChunk]:
    parser = etree.XMLParser(load_dtd=False, resolve_entities=False, recover=True)
    tree = etree.parse(str(xml_path), parser)
    root = tree.getroot()
    out: list[SjwChunk] = []

    # root이 level1 (item) 또는 level2일 수 있음
    if root.tag == "item":
        l1_nodes = root.findall("level1")
    elif root.tag == "level1":
        l1_nodes = [root]
    else:
        # root이 level2임 → 가상 level1 처리
        l1_nodes = [None]

    def _process_level2(l2, king_prefix, king):
        year_id = l2.get("id", "")
        front = l2.find("front")
        bib = front.find("biblioData") if front is not None else None
        title = bib.find("title") if bib is not None else None
        year_label = _txt(title.find("mainTitle")) if title is not None else ""
        _, _, _, year_ce = _parse_dates(bib)

        for l3 in l2.findall("level3"):
            month_id = l3.get("id", "")
            for l4 in l3.findall("level4"):
                day_id = l4.get("id", "")
                fr = l4.find("front")
                bib4 = fr.find("biblioData") if fr is not None else None
                t4 = bib4.find("title") if bib4 is not None else None
                day_title = _txt(t4.find("mainTitle")) if t4 is not None else ""
                ymd, ganji, reign, _ = _parse_dates(bib4)

                # level5 leaves (or fallback: level4 with content)
                l5s = l4.findall("level5")
                if not l5s:
                    # level4 자체가 leaf로 paragraph 보유?
                    txt_el = l4.find("text")
                    if txt_el is not None:
                        for p in txt_el.iter("paragraph"):
                            # 의미 없는 노드는 제외 (content가 빈 경우)
                            pass
                    continue
                for l5 in l5s:
                    aid = l5.get("id", "")
                    fr5 = l5.find("front")
                    bib5 = fr5.find("biblioData") if fr5 is not None else None
                    t5 = bib5.find("title") if bib5 is not None else None
                    article_title = _txt(t5.find("mainTitle")) if t5 is not None else ""
                    article_type = l5.get("type", "")
                    text_el = l5.find("text")
                    if text_el is None: continue
                    parts = []; ents = {}
                    for content in text_el.iter("content"):
                        for p in content.findall("paragraph"):
                            t, e = _para_text_and_entities(p)
                            if t:
                                parts.append(t); _merge_entities(ents, e)
                    if not parts: continue
                    joined = " ".join(parts)
                    ctx_bits = [b for b in (king, day_title, article_title) if b]
                    header = "[" + " | ".join(ctx_bits) + "]" if ctx_bits else ""
                    combined = (header + "\n" + joined) if header else joined
                    pieces = _split_long(combined)
                    for j, piece in enumerate(pieces):
                        cid = aid if len(pieces) == 1 else f"{aid}_s{j}"
                        out.append(SjwChunk(
                            chunk_id=cid, article_id=aid, book_id="sjw",
                            king_prefix=king_prefix, king=king,
                            year_id=year_id, year_label=year_label, year_ce=year_ce,
                            month_id=month_id,
                            day_id=day_id, day_title=day_title,
                            date_western=ymd, ganji=ganji, reign_year=reign,
                            article_type=article_type, article_title=article_title,
                            text=piece, text_hangul_aux=to_hangul(piece),
                            entities=ents if j == 0 else {},
                        ))

    # 루트 형태별 처리
    if root.tag == "item":
        for l1 in l1_nodes:
            l1_id = l1.get("id", "")
            king_prefix = l1_id.split("-")[1] if "-" in l1_id else ""
            king = KING_MAP.get(king_prefix, "")
            for l2 in l1.findall("level2"):
                _process_level2(l2, king_prefix, king)
    elif root.tag == "level1":
        l1_id = root.get("id", "")
        king_prefix = l1_id.split("-")[1] if "-" in l1_id else ""
        king = KING_MAP.get(king_prefix, "")
        for l2 in root.findall("level2"):
            _process_level2(l2, king_prefix, king)
    elif root.tag == "level2":
        # root이 level2 → king은 id에서 추출
        l2_id = root.get("id", "")  # SJW-A01
        # SJW-A01 → A
        m = re.match(r"^SJW-([A-Z])\d+$", l2_id)
        king_prefix = m.group(1) if m else ""
        king = KING_MAP.get(king_prefix, "")
        _process_level2(root, king_prefix, king)
    return out


def main() -> int:
    xml_files = sorted(RAW.glob("2nd_*.xml"))
    if not xml_files:
        print(f"[err] no 2nd_*.xml in {RAW}", file=sys.stderr); return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    total = 0; long_articles = 0; seen = set()
    with OUT.open("w", encoding="utf-8") as f:
        for xf in tqdm(xml_files, desc="chunk"):
            try:
                chunks = chunk_file(xf)
            except Exception as e:
                print(f"[warn] {xf.name}: {e}", file=sys.stderr); continue
            for c in chunks:
                if c.chunk_id != c.article_id and c.article_id not in seen:
                    long_articles += 1; seen.add(c.article_id)
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            total += len(chunks)
    print(f"[done] wrote {total} chunks → {OUT}")
    print(f"[info] {long_articles} long articles split (>{MAX_CHARS})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
