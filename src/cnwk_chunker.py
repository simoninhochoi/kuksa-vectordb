"""원고려기사 (元高麗紀事) → 기사 단위 청크.

XML 1개 (cnwk_001.xml) 안에 level1×9 (序文/태조/태종/정종/헌종/세조/성종/탐라/附記),
각 level1 → level2 (연도) → level3 (개별 기사, leaf with <text>).
청크 단위: level3 = 한 기사 1건. 본문이 짧아 분할 거의 없음.

산출물: data/원고려기사/chunks.jsonl
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
RAW = ROOT / "data" / "원고려기사" / "raw"
OUT = ROOT / "data" / "원고려기사" / "chunks.jsonl"


@dataclass
class CnwkChunk:
    chunk_id: str
    article_id: str          # level3 id (예: cnwk_001_0010_0010)
    book_id: str             # "cnwk" 고정
    level1_id: str           # cnwk_001~009
    level1_title: str        # "[太祖皇帝]" 등
    level2_id: str
    level2_label: str        # "태조(太祖) 13년(1218년)"
    article_title: str       # mainTitle
    date_western: str        # YYYY-MM-DD or YYYY-MM-99 등
    date_label: str          # dateOccured 텍스트
    text: str
    text_hangul_aux: str
    entities: dict[str, list[str]] = field(default_factory=dict)


def _txt(el: etree._Element | None) -> str:
    if el is None:
        return ""
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _level_main_title(level_el: etree._Element) -> str:
    front = level_el.find("front")
    if front is None:
        return ""
    bib = front.find("biblioData")
    if bib is None:
        return ""
    title = bib.find("title")
    if title is None:
        return ""
    return _txt(title.find("mainTitle"))


def _parse_date(bib: etree._Element) -> tuple[str, str]:
    """biblioData/date/dateOccured → (YYYY-MM-DD or partial, label)."""
    date_el = bib.find("date") if bib is not None else None
    if date_el is None:
        return "", ""
    do = date_el.find("dateOccured")
    if do is None:
        return "", ""
    raw = (do.get("date") or "").strip()
    label = _txt(do)
    # "1218-99-99L0" → "1218-99-99"
    m = re.match(r"^(\d{4}(?:-\d{2}(?:-\d{2})?)?)(?:L[01])?$", raw)
    if m:
        return m.group(1), label
    return raw, label


def chunk_file(xml_path: Path) -> list[CnwkChunk]:
    parser = etree.XMLParser(load_dtd=False, resolve_entities=False, recover=True)
    tree = etree.parse(str(xml_path), parser)
    root = tree.getroot()
    out: list[CnwkChunk] = []

    for l1 in root.iter("level1"):
        l1_id = l1.get("id", "")
        l1_title = _level_main_title(l1)
        for l2 in l1.findall("level2"):
            l2_id = l2.get("id", "")
            l2_label = l2.get("value") or _level_main_title(l2)
            for l3 in l2.findall("level3"):
                aid = l3.get("id", "")
                front = l3.find("front")
                bib = front.find("biblioData") if front is not None else None
                title = bib.find("title") if bib is not None else None
                article_title = _txt(title.find("mainTitle")) if title is not None else ""
                date_western, date_label = _parse_date(bib) if bib is not None else ("", "")

                text_el = l3.find("text")
                if text_el is None:
                    continue
                # paragraph 평문화 + 엔티티 수집
                ents: dict[str, list[str]] = {}
                parts: list[str] = []
                for content in text_el.iter("content"):
                    for p in content.findall("paragraph"):
                        t, e = _para_text_and_entities(p)
                        if t:
                            parts.append(t)
                            _merge_entities(ents, e)
                if not parts:
                    continue
                joined = " ".join(parts)

                # 임베딩 컨텍스트 prefix
                ctx_bits = [b for b in (l1_title, l2_label, article_title) if b]
                header = "[" + " | ".join(ctx_bits) + "]" if ctx_bits else ""
                combined = (header + "\n" + joined) if header else joined

                pieces = _split_long(combined)
                for j, piece in enumerate(pieces):
                    cid = aid if len(pieces) == 1 else f"{aid}_s{j}"
                    out.append(
                        CnwkChunk(
                            chunk_id=cid,
                            article_id=aid,
                            book_id="cnwk",
                            level1_id=l1_id,
                            level1_title=l1_title,
                            level2_id=l2_id,
                            level2_label=l2_label,
                            article_title=article_title,
                            date_western=date_western,
                            date_label=date_label,
                            text=piece,
                            text_hangul_aux=to_hangul(piece),
                            entities=ents if j == 0 else {},
                        )
                    )
    return out


def main() -> int:
    xml_files = sorted(RAW.glob("cnwk_*.xml"))
    if not xml_files:
        print(f"[err] no cnwk_*.xml in {RAW}", file=sys.stderr)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    long_articles = 0
    seen: set[str] = set()
    with OUT.open("w", encoding="utf-8") as f:
        for xf in tqdm(xml_files, desc="chunking"):
            chunks = chunk_file(xf)
            for c in chunks:
                if c.chunk_id != c.article_id and c.article_id not in seen:
                    long_articles += 1
                    seen.add(c.article_id)
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            total += len(chunks)
    print(f"[done] wrote {total} chunks → {OUT}")
    print(f"[info] {long_articles} articles split (>{MAX_CHARS})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
