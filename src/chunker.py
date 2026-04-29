"""한국사료총서 XML → paragraph 단위 청크 (chunks.jsonl).

각 청크 스키마:
{
  "chunk_id": "sa_003_$1int_p07",
  "volume_id": "sa_003",
  "volume_title_hanja": "海鶴遺書",
  "volume_title_hangul": "해학유서",
  "series_volume": 3,
  "author": "이기(李沂)",
  "period_begin": "1800-00-00",
  "period_end": "1899-00-00",
  "subject_class": "집부(集部)_별집류(別集類)",
  "level1_title": "海鶴遺書",
  "level2_title": "序",
  "level3_title": null,
  "text": "<원문(한자)>",
  "text_hangul_aux": "<음차본 — 검색 보조>",
  "entities": {"person": [...], "place": [...], "title": [...], ...}
}
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path

import regex as re
from lxml import etree
from tqdm import tqdm

from hanja_util import to_hangul

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "chunks.jsonl"

# DTD index type → 내부 카테고리 (entity_dict.py와 동일)
TYPE_MAP = {
    "이름": "person", "지명": "place", "관직": "position",
    "서명": "title", "관서": "office", "단체": "group",
    "사건": "event", "국명": "nation", "연호": "era",
    "학교": "school", "회사조합": "company", "기타": "other",
}

# 청킹 임계값 (문자 수 기준; BGE-M3는 한자/한글 1자 ≈ 1~1.5 token)
MIN_CHARS = 40
MAX_CHARS = 800
OVERLAP_CHARS = 100


@dataclass
class VolumeMeta:
    volume_id: str
    volume_title_hanja: str = ""
    volume_title_hangul: str = ""
    series_volume: int | None = None
    author: str = ""
    period_begin: str = ""
    period_end: str = ""
    subject_class: str = ""


@dataclass
class Chunk:
    chunk_id: str
    volume_id: str
    volume_title_hanja: str
    volume_title_hangul: str
    series_volume: int | None
    author: str
    period_begin: str
    period_end: str
    subject_class: str
    level1_title: str
    level2_title: str
    level3_title: str | None
    text: str
    text_hangul_aux: str
    entities: dict[str, list[str]] = field(default_factory=dict)


def _txt(el: etree._Element | None) -> str:
    if el is None:
        return ""
    return re.sub(r"\s+", "", "".join(el.itertext())).strip()


def _para_text_and_entities(p: etree._Element) -> tuple[str, dict[str, list[str]]]:
    """paragraph 안의 평문 + 엔티티 목록 추출.
    <index>·<annotation>·<number>(페이지) 등은 평문화하되 엔티티는 별도 수집.
    """
    ents: dict[str, set[str]] = defaultdict(set)
    parts: list[str] = []

    def walk(node: etree._Element) -> None:
        if node.text:
            parts.append(node.text)
        for child in node:
            if not isinstance(child.tag, str):
                # comment/PI 등은 건너뜀 (꼬리 텍스트만 보존)
                if child.tail:
                    parts.append(child.tail)
                continue
            tag = etree.QName(child).localname
            if tag == "index":
                t = child.get("type")
                inner = "".join(child.itertext())
                if t in TYPE_MAP and inner:
                    ents[TYPE_MAP[t]].add(inner.strip())
                parts.append(inner)
            elif tag == "number":
                # 페이지 번호 등은 본문에서 제외
                pass
            elif tag == "br":
                parts.append(" ")
            elif tag == "annotation":
                # 편자주·원주 등은 본문 흐름에 미세하게 포함
                inner = "".join(child.itertext())
                parts.append(inner)
            else:
                walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(p)
    text = re.sub(r"\s+", " ", "".join(parts)).strip()
    return text, {k: sorted(v) for k, v in ents.items()}


def _merge_entities(dst: dict[str, list[str]], src: dict[str, list[str]]) -> None:
    for k, vs in src.items():
        merged = set(dst.get(k, [])) | set(vs)
        dst[k] = sorted(merged)


def _split_long(text: str) -> list[str]:
    """MAX_CHARS 초과 시 OVERLAP을 두고 슬라이딩."""
    if len(text) <= MAX_CHARS:
        return [text]
    out: list[str] = []
    step = MAX_CHARS - OVERLAP_CHARS
    i = 0
    while i < len(text):
        out.append(text[i : i + MAX_CHARS])
        if i + MAX_CHARS >= len(text):
            break
        i += step
    return out


def _extract_volume_meta(root: etree._Element) -> VolumeMeta:
    """level1 또는 item 하위 첫 level1에서 권 메타 추출."""
    # 진짜 본문 권 노드 (sa_NNN) 탐색
    if root.tag == "item":
        l1 = root.find("level1")
    else:
        l1 = root if root.tag == "level1" else root.find(".//level1")
    if l1 is None:
        return VolumeMeta(volume_id="unknown")

    vol_id = l1.get("id", "unknown")
    front = l1.find("front")
    biblio = front.find("biblioData") if front is not None else None
    desc = front.find("description") if front is not None else None

    meta = VolumeMeta(volume_id=vol_id)
    if biblio is not None:
        title = biblio.find("title")
        if title is not None:
            meta.volume_title_hanja = _txt(title.find("mainTitle"))
            alt = title.find("alternative")
            if alt is not None:
                meta.volume_title_hangul = _txt(alt)
            else:
                meta.volume_title_hangul = to_hangul(meta.volume_title_hanja)
            series = title.find("seriesTitle")
            if series is not None and series.get("volume"):
                try:
                    meta.series_volume = int(series.get("volume"))
                except ValueError:
                    pass
        creator = biblio.find("creator")
        if creator is not None:
            meta.author = _txt(creator.find("author"))
        sc = biblio.find("subjectClass")
        if sc is not None:
            meta.subject_class = _txt(sc)
    if desc is not None:
        cp = desc.find("coveragePeriod")
        if cp is not None:
            meta.period_begin = cp.get("begin", "")
            meta.period_end = cp.get("end", "")
    return meta


def _level_title(level_el: etree._Element) -> str:
    """level의 front/biblioData/title/mainTitle 추출."""
    front = level_el.find("front")
    if front is None:
        return ""
    biblio = front.find("biblioData")
    if biblio is None:
        return ""
    title = biblio.find("title")
    if title is None:
        return ""
    return _txt(title.find("mainTitle"))


def _process_text_node(
    text_el: etree._Element,
    base_id: str,
) -> list[tuple[str, str, dict[str, list[str]]]]:
    """text 요소 → [(suffix_id, raw_text, entities), ...]  paragraph 단위."""
    paras: list[tuple[str, str, dict[str, list[str]]]] = []
    pidx = 0
    for content in text_el.iter("content"):
        for p in content.findall("paragraph"):
            text, ents = _para_text_and_entities(p)
            if not text:
                continue
            paras.append((f"p{pidx:03d}", text, ents))
            pidx += 1
    return paras


def _collect_chunks_for_level(
    level_el: etree._Element,
    vol: VolumeMeta,
    level1_title: str,
    level2_title: str,
    level3_title: str | None,
) -> list[Chunk]:
    """level{1,2,3}의 text를 청크로 변환. level 자체에 text 있으면 처리."""
    out: list[Chunk] = []
    text_el = level_el.find("text")
    if text_el is None:
        return out

    base_id = level_el.get("id", "noid")
    paras = _process_text_node(text_el, base_id)

    # 머지: 짧은 단락 연속 → 합치기
    merged: list[tuple[str, str, dict[str, list[str]]]] = []
    buf_text = ""
    buf_ents: dict[str, list[str]] = {}
    buf_first_id = ""
    for suffix, text, ents in paras:
        if not buf_text:
            buf_text = text
            buf_ents = dict(ents)
            buf_first_id = suffix
            continue
        if len(buf_text) < MIN_CHARS:
            buf_text = (buf_text + " " + text).strip()
            _merge_entities(buf_ents, ents)
        else:
            merged.append((buf_first_id, buf_text, buf_ents))
            buf_text = text
            buf_ents = dict(ents)
            buf_first_id = suffix
    if buf_text:
        merged.append((buf_first_id, buf_text, buf_ents))

    # split long, build chunks
    for suffix, text, ents in merged:
        pieces = _split_long(text)
        for j, piece in enumerate(pieces):
            chunk_id = f"{base_id}_{suffix}"
            if len(pieces) > 1:
                chunk_id += f"_s{j}"
            out.append(
                Chunk(
                    chunk_id=chunk_id,
                    volume_id=vol.volume_id,
                    volume_title_hanja=vol.volume_title_hanja,
                    volume_title_hangul=vol.volume_title_hangul,
                    series_volume=vol.series_volume,
                    author=vol.author,
                    period_begin=vol.period_begin,
                    period_end=vol.period_end,
                    subject_class=vol.subject_class,
                    level1_title=level1_title,
                    level2_title=level2_title,
                    level3_title=level3_title,
                    text=piece,
                    text_hangul_aux=to_hangul(piece),
                    entities=ents,
                )
            )
    return out


def chunk_volume(xml_path: Path) -> list[Chunk]:
    parser = etree.XMLParser(load_dtd=False, resolve_entities=False, recover=True)
    tree = etree.parse(str(xml_path), parser)
    root = tree.getroot()

    vol = _extract_volume_meta(root)
    chunks: list[Chunk] = []

    # 본문 level1 위치
    if root.tag == "item":
        l1_nodes = root.findall("level1")
    elif root.tag == "level1":
        l1_nodes = [root]
    else:
        l1_nodes = root.findall(".//level1")

    for l1 in l1_nodes:
        l1_title = _level_title(l1) or vol.volume_title_hanja
        chunks += _descend(l1, vol, l1_title, "", None, depth=1)

    return chunks


def _descend(
    node: etree._Element,
    vol: VolumeMeta,
    l1_title: str,
    l2_title: str,
    l3_title: str | None,
    depth: int,
) -> list[Chunk]:
    """level{N} 재귀 순회. 각 노드의 text를 수집하고 자식 levelM(M>N)으로 내려감.
    l1/l2/l3_title 메타는 처음 등장하는 3 레벨까지만 잡고 더 깊어지면 l3에 누적.
    """
    chunks = _collect_chunks_for_level(node, vol, l1_title, l2_title, l3_title)
    # 자식 level 노드 (level2/3/4/5 등) 모두 탐색
    for child in node:
        if not isinstance(child.tag, str):
            continue
        tag = etree.QName(child).localname
        if not tag.startswith("level"):
            continue
        try:
            child_depth = int(tag[5:])
        except ValueError:
            continue
        c_title = _level_title(child)
        if depth == 1:
            chunks += _descend(child, vol, l1_title, c_title, None, child_depth)
        elif depth == 2:
            chunks += _descend(child, vol, l1_title, l2_title, c_title, child_depth)
        else:
            # depth ≥ 3: 더 깊은 제목은 l3_title에 " > " 로 누적
            new_l3 = f"{l3_title} > {c_title}" if l3_title and c_title else (l3_title or c_title)
            chunks += _descend(child, vol, l1_title, l2_title, new_l3, child_depth)
    return chunks


def main() -> int:
    xml_files = sorted(RAW.glob("sa_*.xml"))
    if not xml_files:
        print(f"[err] no XML in {RAW}", file=sys.stderr)
        return 1

    total = 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for xf in tqdm(xml_files, desc="chunking"):
            try:
                chunks = chunk_volume(xf)
            except Exception as e:
                print(f"[warn] {xf.name}: {e}", file=sys.stderr)
                continue
            for c in chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            total += len(chunks)

    print(f"[done] wrote {total} chunks → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
