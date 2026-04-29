"""비변사등록 (備邊司謄錄) → 기사 단위 청크.

XML 274개 (bb_001 ~ bb_NNN). root는 item, 그 안에 level1=권 단위.
계층: level1=권 → level2=년(왕명·재위년도·간지·서기 attrs) → level3=월
       → level4=개별 기사 (좌목·안건 등, leaf with <text>)

산출물: data/비변사등록/chunks.jsonl
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
RAW = ROOT / "data" / "비변사등록" / "raw"
OUT = ROOT / "data" / "비변사등록" / "chunks.jsonl"


@dataclass
class BbChunk:
    chunk_id: str
    article_id: str
    book_id: str            # "bb"
    volume_id: str          # bb_NNN
    year_ce: str            # 1617
    ganji: str              # 정사
    king: str               # 광해군
    reign_year: str         # 9
    month_value: str        # 01-12 or "윤NN"
    article_title: str
    date_western: str
    text: str
    text_hangul_aux: str
    entities: dict[str, list[str]] = field(default_factory=dict)


def _txt(el):
    if el is None:
        return ""
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _parse_dates(bib):
    if bib is None:
        return ""
    date_el = bib.find("date")
    if date_el is None:
        return ""
    do = date_el.find("dateOccured")
    if do is None:
        return ""
    raw = (do.get("date") or "").strip()
    m = re.match(r"^(\d{4}(?:-\d{2}(?:-\d{2})?)?)(?:L[01])?$", raw)
    if m:
        return m.group(1)
    return raw


def chunk_file(xml_path):
    parser = etree.XMLParser(load_dtd=False, resolve_entities=False, recover=True)
    tree = etree.parse(str(xml_path), parser)
    root = tree.getroot()
    out = []

    for l1 in root.iter("level1"):
        vid = l1.get("id", "")
        for l2 in l1.findall("level2"):
            year_ce = l2.get("value", "")
            ganji = l2.get("간지", "")
            king = l2.get("왕명", "")
            reign = l2.get("재위년도", "")
            for l3 in l2.findall("level3"):
                month_value = l3.get("value", "")
                for l4 in l3.findall("level4"):
                    if l4.find("text") is None:
                        continue
                    aid = l4.get("id", "")
                    front = l4.find("front")
                    bib = front.find("biblioData") if front is not None else None
                    title = bib.find("title") if bib is not None else None
                    article_title = _txt(title.find("mainTitle")) if title is not None else ""
                    date_western = _parse_dates(bib)

                    parts = []; ents = {}
                    for content in l4.find("text").iter("content"):
                        for p in content.findall("paragraph"):
                            t, e = _para_text_and_entities(p)
                            if t:
                                parts.append(t)
                                _merge_entities(ents, e)
                    if not parts:
                        continue
                    joined = " ".join(parts)

                    ctx_bits = [b for b in (king, f"{year_ce} {ganji}".strip(),
                                              f"{month_value}월" if month_value else "",
                                              article_title) if b]
                    header = "[" + " | ".join(ctx_bits) + "]" if ctx_bits else ""
                    combined = (header + "\n" + joined) if header else joined
                    pieces = _split_long(combined)
                    for j, piece in enumerate(pieces):
                        cid = aid if len(pieces) == 1 else f"{aid}_s{j}"
                        out.append(BbChunk(
                            chunk_id=cid, article_id=aid, book_id="bb",
                            volume_id=vid, year_ce=year_ce, ganji=ganji,
                            king=king, reign_year=reign,
                            month_value=month_value, article_title=article_title,
                            date_western=date_western,
                            text=piece, text_hangul_aux=to_hangul(piece),
                            entities=ents if j == 0 else {},
                        ))
    return out


def main():
    xml_files = sorted(RAW.glob("bb_*.xml"))
    if not xml_files:
        print(f"[err] no bb_*.xml in {RAW}", file=sys.stderr); return 1
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
