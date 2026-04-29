"""고려사 (高麗史) → 기사 단위 청크.

XML 139개 (kr_000=메타, kr_NNN=권). 권 root는 level1 with type='世家'/'志'/'列傳'.
Leaf 깊이가 변동적: level3 (총서) 또는 level4 (일별 기사).
generic recursive walker — `<text>` 보유 노드 = leaf article.

산출물: data/고려사/chunks.jsonl
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
RAW = ROOT / "data" / "고려사" / "raw"
OUT = ROOT / "data" / "고려사" / "chunks.jsonl"


@dataclass
class KrChunk:
    chunk_id: str
    article_id: str
    book_id: str            # "kr"
    volume_id: str          # level1 id (kr_NNN)
    section_type: str       # 世家/志/列傳/序文/고려세계 등 (level1 type)
    volume_label: str       # 권 mainTitle 또는 type
    breadcrumbs: list[str]
    article_title: str
    date_western: str
    date_label: str
    ganji: str
    text: str
    text_hangul_aux: str
    entities: dict[str, list[str]] = field(default_factory=dict)


def _txt(el):
    if el is None:
        return ""
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _node_label(node):
    if (v := node.get("value")):
        return v
    front = node.find("front")
    if front is not None:
        bib = front.find("biblioData")
        if bib is not None:
            title = bib.find("title")
            if title is not None:
                t = _txt(title.find("mainTitle"))
                if t: return t
    return node.get("type", "")


def _parse_dates(bib):
    if bib is None:
        return "", "", ""
    date_el = bib.find("date")
    if date_el is None:
        return "", "", ""
    ymd = ""; label = ""; ganji = ""
    for d in date_el.findall("dateOccured"):
        t = d.get("type")
        raw = (d.get("date") or "").strip()
        if t == "양":
            m = re.match(r"^(\d{4}(?:-\d{2}(?:-\d{2})?)?)(?:L[01])?$", raw)
            if m: ymd = m.group(1); label = _txt(d) or label
        elif t == "음" and not ymd:
            m = re.match(r"^(\d{4}(?:-\d{2}(?:-\d{2})?)?)(?:L[01])?$", raw)
            if m: ymd = m.group(1); label = _txt(d) or label
        elif t == "일간지":
            ganji = _txt(d)
    return ymd, label, ganji


def _emit_article(node, ancestors, book_id, volume_id, section_type, volume_label):
    aid = node.get("id", "")
    front = node.find("front")
    bib = front.find("biblioData") if front is not None else None
    title = bib.find("title") if bib is not None else None
    article_title = _txt(title.find("mainTitle")) if title is not None else ""
    ymd, dlabel, ganji = _parse_dates(bib)
    breadcrumbs = []
    for a in ancestors:
        lbl = _node_label(a)
        if lbl: breadcrumbs.append(lbl)
    text_el = node.find("text")
    if text_el is None:
        return []
    parts = []; ents = {}
    for content in text_el.iter("content"):
        for p in content.findall("paragraph"):
            t, e = _para_text_and_entities(p)
            if t:
                parts.append(t); _merge_entities(ents, e)
    if not parts:
        return []
    joined = " ".join(parts)
    ctx = list(breadcrumbs)
    if article_title: ctx.append(article_title)
    header = "[" + " | ".join(ctx) + "]" if ctx else ""
    combined = (header + "\n" + joined) if header else joined
    pieces = _split_long(combined)
    out = []
    for j, piece in enumerate(pieces):
        cid = aid if len(pieces) == 1 else f"{aid}_s{j}"
        out.append(KrChunk(
            chunk_id=cid, article_id=aid, book_id=book_id,
            volume_id=volume_id, section_type=section_type,
            volume_label=volume_label, breadcrumbs=breadcrumbs,
            article_title=article_title,
            date_western=ymd, date_label=dlabel, ganji=ganji,
            text=piece, text_hangul_aux=to_hangul(piece),
            entities=ents if j == 0 else {},
        ))
    return out


def _walk(node, ancestors, volume_id, section_type, volume_label, book_id):
    out = []
    if node.find("text") is not None:
        out.extend(_emit_article(node, ancestors, book_id, volume_id, section_type, volume_label))
    for child in node:
        if not isinstance(child.tag, str):
            continue
        tag = etree.QName(child).localname
        if tag.startswith("level"):
            out.extend(_walk(child, ancestors + [node], volume_id, section_type, volume_label, book_id))
    return out


def chunk_file(xml_path):
    parser = etree.XMLParser(load_dtd=False, resolve_entities=False, recover=True)
    tree = etree.parse(str(xml_path), parser)
    root = tree.getroot()
    book_id = "kr"
    if root.tag == "item":
        chunks = []
        for l1 in root.findall("level1"):
            vid = l1.get("id", "")
            stype = l1.get("type", "")
            vlabel = _node_label(l1) or stype
            chunks.extend(_walk(l1, [], vid, stype, vlabel, book_id))
        return chunks
    else:
        vid = root.get("id", "")
        stype = root.get("type", "")
        vlabel = _node_label(root) or stype
        return _walk(root, [], vid, stype, vlabel, book_id)


def main():
    xml_files = sorted(RAW.glob("kr_*.xml"))
    if not xml_files:
        print(f"[err] no kr_*.xml in {RAW}", file=sys.stderr); return 1
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
