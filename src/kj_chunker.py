"""고려사절요 (高麗史節要) → 기사 단위 청크.

XML 36개 (kj_000=메타, kj_001~kj_035=권). 권 root는 level1.
계층: level1=권 → level2=대분류(卷N) → level3=중분류 또는 leaf(태조총서 등)
       → level4=년 → level5=일/기사 (leaf with <text>)
leaf 깊이가 변동적이라 generic recursive walker로 처리 — `<text>` 보유한
가장 깊은 노드를 article로 간주.

산출물: data/고려사절요/chunks.jsonl
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
RAW = ROOT / "data" / "고려사절요" / "raw"
OUT = ROOT / "data" / "고려사절요" / "chunks.jsonl"


@dataclass
class KjChunk:
    chunk_id: str
    article_id: str
    book_id: str            # "kj"
    volume_id: str          # level1 id (kj_NNN)
    volume_label: str       # 권 mainTitle 또는 type
    breadcrumbs: list[str]  # 조상 노드들의 라벨들 (level2~ 부터)
    article_title: str
    date_western: str
    date_label: str
    ganji: str              # 일간지
    text: str
    text_hangul_aux: str
    entities: dict[str, list[str]] = field(default_factory=dict)


def _txt(el: etree._Element | None) -> str:
    if el is None:
        return ""
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _node_label(node: etree._Element) -> str:
    """level node → 표시용 레이블. value 우선, 없으면 mainTitle, 없으면 type."""
    if (v := node.get("value")):
        return v
    front = node.find("front")
    if front is not None:
        bib = front.find("biblioData")
        if bib is not None:
            title = bib.find("title")
            if title is not None:
                t = _txt(title.find("mainTitle"))
                if t:
                    return t
    if (typ := node.get("type")):
        return typ
    return ""


def _parse_dates(bib: etree._Element | None) -> tuple[str, str, str]:
    """biblioData/date → (date_western_YYYY-MM-DD, date_label, ganji)."""
    if bib is None:
        return "", "", ""
    date_el = bib.find("date")
    if date_el is None:
        return "", "", ""
    ymd = ""
    label = ""
    ganji = ""
    for d in date_el.findall("dateOccured"):
        t = d.get("type")
        raw = (d.get("date") or "").strip()
        if t == "양":
            m = re.match(r"^(\d{4}(?:-\d{2}(?:-\d{2})?)?)(?:L[01])?$", raw)
            if m:
                ymd = m.group(1)
                label = _txt(d) or label
        elif t == "음" and not ymd:
            m = re.match(r"^(\d{4}(?:-\d{2}(?:-\d{2})?)?)(?:L[01])?$", raw)
            if m:
                ymd = m.group(1)
                label = _txt(d) or label
        elif t == "일간지":
            ganji = _txt(d)
    return ymd, label, ganji


def _emit_article(node: etree._Element, ancestors: list[etree._Element], book_id: str,
                   volume_id: str, volume_label: str) -> list[KjChunk]:
    aid = node.get("id", "")
    front = node.find("front")
    bib = front.find("biblioData") if front is not None else None
    title = bib.find("title") if bib is not None else None
    article_title = _txt(title.find("mainTitle")) if title is not None else ""
    ymd, dlabel, ganji = _parse_dates(bib)

    # Breadcrumb labels: 조상 level 노드들의 라벨
    breadcrumbs = []
    for a in ancestors:
        lbl = _node_label(a)
        if lbl:
            breadcrumbs.append(lbl)

    text_el = node.find("text")
    if text_el is None:
        return []
    parts: list[str] = []
    ents: dict[str, list[str]] = {}
    for content in text_el.iter("content"):
        for p in content.findall("paragraph"):
            t, e = _para_text_and_entities(p)
            if t:
                parts.append(t)
                _merge_entities(ents, e)
    if not parts:
        return []
    joined = " ".join(parts)

    # 컨텍스트 prefix
    ctx = list(breadcrumbs)
    if article_title:
        ctx.append(article_title)
    header = "[" + " | ".join(ctx) + "]" if ctx else ""
    combined = (header + "\n" + joined) if header else joined
    pieces = _split_long(combined)

    out: list[KjChunk] = []
    for j, piece in enumerate(pieces):
        cid = aid if len(pieces) == 1 else f"{aid}_s{j}"
        out.append(KjChunk(
            chunk_id=cid, article_id=aid, book_id=book_id,
            volume_id=volume_id, volume_label=volume_label,
            breadcrumbs=breadcrumbs,
            article_title=article_title,
            date_western=ymd, date_label=dlabel, ganji=ganji,
            text=piece, text_hangul_aux=to_hangul(piece),
            entities=ents if j == 0 else {},
        ))
    return out


def _walk(node: etree._Element, ancestors: list[etree._Element],
           volume_id: str, volume_label: str, book_id: str) -> list[KjChunk]:
    """level 노드 재귀. <text> 보유 노드 = leaf article로 추출."""
    out: list[KjChunk] = []
    has_text = node.find("text") is not None
    if has_text:
        # 자체가 leaf — 청킹
        out.extend(_emit_article(node, ancestors, book_id, volume_id, volume_label))
    # 자식 level 들도 항상 재귀 (leaf 노드가 자식 level을 갖지 않는 게 일반적이지만 안전하게)
    for child in node:
        if not isinstance(child.tag, str):
            continue
        tag = etree.QName(child).localname
        if tag.startswith("level"):
            out.extend(_walk(child, ancestors + [node], volume_id, volume_label, book_id))
    return out


def chunk_file(xml_path: Path) -> list[KjChunk]:
    parser = etree.XMLParser(load_dtd=False, resolve_entities=False, recover=True)
    tree = etree.parse(str(xml_path), parser)
    root = tree.getroot()
    book_id = "kj"

    # root은 level1 (권 단위) 또는 item (kj_000 메타)
    if root.tag == "item":
        # kj_000: <item> 안에 여러 level1 (서문/범례/수사관/목록)
        chunks = []
        for l1 in root.findall("level1"):
            vid = l1.get("id", "")
            vlabel = _node_label(l1) or l1.get("type", "")
            chunks.extend(_walk(l1, [], vid, vlabel, book_id))
        return chunks
    else:
        # level1 root (권)
        vid = root.get("id", "")
        vlabel = _node_label(root) or root.get("type", "")
        return _walk(root, [], vid, vlabel, book_id)


def main() -> int:
    xml_files = sorted(RAW.glob("kj_*.xml"))
    if not xml_files:
        print(f"[err] no kj_*.xml in {RAW}", file=sys.stderr)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    long_articles = 0
    seen: set[str] = set()
    with OUT.open("w", encoding="utf-8") as f:
        for xf in tqdm(xml_files, desc="chunk"):
            try:
                chunks = chunk_file(xf)
            except Exception as e:
                print(f"[warn] {xf.name}: {e}", file=sys.stderr)
                continue
            for c in chunks:
                if c.chunk_id != c.article_id and c.article_id not in seen:
                    long_articles += 1
                    seen.add(c.article_id)
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            total += len(chunks)
    print(f"[done] wrote {total} chunks → {OUT}")
    print(f"[info] {long_articles} long articles split (>{MAX_CHARS})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
