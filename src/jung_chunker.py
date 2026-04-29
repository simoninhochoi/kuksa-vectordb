"""중국정사외국전 XML → 기사 단위 청크 (data/중국정사외국전/chunks.jsonl).

구조:
  <item id="jo.d_NNNN" name="史記">       — 한 책(예: 사기, 한서, …)
    <level1>                                — 책 전체 메타
      <front>...</front>
      <level2 id="jo.d_NNNN_LL">            — 외국전(편/권), 예: 卷110 匈奴列傳
        <front>...</front>
        <level3 id="...">                   — ★ leaf article (sub-section 없으면)
          <front><biblioData><title><mainTitle/>...
                            <subjectClass schema="국가">흉노(匈奴)</subjectClass>
          <text><content><paragraph>...</paragraph></content></text>
        OR
        <level3>                            — directory (sub-section 있으면)
          <front>...</front>
          <level4 id="...">                 — ★ leaf article (sub-section)
            <front>...</front>
            <text><content><paragraph>...</paragraph></content></text>

청크 단위: leaf article (level3 with `<text>` 또는 level4) 1건 = 1 chunk
(텍스트가 매우 길면 sliding window로 분할).

jo.d_NNNN과 jo.k_NNNN 은 동일한 leaf id 체계 → 한 leaf 당 한 청크에
원문(한문)과 국역(한글) 양쪽을 함께 담는다.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
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

from chunker import MAX_CHARS, OVERLAP_CHARS
from hanja_util import to_hangul

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "중국정사외국전" / "raw"
OUT = ROOT / "data" / "중국정사외국전" / "chunks.jsonl"

# index type → 내부 카테고리. 중국정사외국전은 지명/이름/서명만 사용 (history.dtd 부분집합).
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


@dataclass
class JungBookMeta:
    """한 책(jo_NNNN)의 level1 메타."""
    book_id: str = ""              # "jo_0001"
    book_name: str = ""            # "사기(史記)"
    book_title_hanja: str = ""     # "史記"
    book_title_alt: str = ""       # "譯註 中國 正史 外國傳 1 史記外國傳 譯註"
    book_author: str = ""          # "사마천(司馬遷)"
    book_dynasty: str = ""         # "중국 정사 외국전" (item itemName)
    period_label: str = ""         # "정화(征和) 2년 경"


@dataclass
class JungChunk:
    chunk_id: str
    article_id: str                # leaf id (prefix 제거 형, 예: "jo_0001_0110_0010")
    book_id: str
    book_name: str
    book_title_hanja: str
    book_title_alt: str
    book_author: str
    book_dynasty: str
    period_label: str
    chapter_id: str                # level2 id (prefix 제거 형, 예: "jo_0001_0110")
    chapter_title: str             # level2 mainTitle, 예: "卷110 匈奴列傳 第50"
    article_title: str             # leaf mainTitle, 예: "흉노(匈奴)의 선조와 문화 및 풍속에 대한 설명"
    subject_country: str           # leaf subjectClass schema=국가, 예: "흉노(匈奴)"
    text: str                      # 임베딩 + 표시용 청크 슬라이스
    text_hanmun: str               # 해당 leaf의 한문 원문 전체 (분할 안 됨)
    text_korean: str               # 해당 leaf의 국역 전체 (분할 안 됨, 각주 포함)
    text_hangul_aux: str           # text의 한자 → 한글 음차본
    has_footnotes_korean: int      # 각주 개수
    has_footnotes_hanmun: int      # 교감주 개수
    entities: dict[str, list[str]] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────

def _txt(el: etree._Element | None) -> str:
    if el is None:
        return ""
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _strip_prefix(node_id: str) -> str:
    """jo.d_0001_0110_0010 / jo.k_0001_0110_0010 → jo_0001_0110_0010."""
    if not node_id:
        return node_id
    return node_id.replace("jo.d_", "jo_", 1).replace("jo.k_", "jo_", 1)


def _flatten_paragraphs(text_el: etree._Element) -> tuple[str, dict[str, list[str]], int]:
    """text element 안의 모든 paragraph를 평문화. (text, entities, footnote_count) 반환.

    - <index> : 본문에 inner 텍스트 그대로 + entities 기록
    - <annotation> : noteContent 내부 텍스트도 그대로 본문에 포함 (각주 본문도 검색 대상에)
    - <br/> : 공백
    """
    if text_el is None:
        return "", {}, 0
    ents: dict[str, set[str]] = defaultdict(set)
    parts: list[str] = []
    fn_count = 0

    def walk(node: etree._Element) -> None:
        nonlocal fn_count
        if node.text:
            parts.append(node.text)
        for child in node:
            if not isinstance(child.tag, str):
                if child.tail:
                    parts.append(child.tail)
                continue
            tag = etree.QName(child).localname
            if tag == "index":
                t = child.get("type")
                inner = "".join(child.itertext())
                if t in TYPE_MAP and inner:
                    ents[TYPE_MAP[t]].add(re.sub(r"\s+", "", inner).strip())
                parts.append(inner)
            elif tag == "annotation":
                fn_count += 1
                # 각주(번역주·교감주·각주) 본문도 그대로 본문 흐름에 흡수
                inner = "".join(child.itertext())
                parts.append(" " + inner + " ")
            elif tag == "br":
                parts.append(" ")
            elif tag == "number":
                # 페이지 번호 등은 제외
                pass
            else:
                walk(child)
            if child.tail:
                parts.append(child.tail)

    for content in text_el.iter("content"):
        for p in content.findall("paragraph"):
            walk(p)
            parts.append("\n")

    text = re.sub(r"[ \t]+", " ", "".join(parts))
    text = re.sub(r"\n[ \n]*", "\n", text).strip()
    return text, {k: sorted(v) for k, v in ents.items()}, fn_count


def _split_combined(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """긴 본문을 sliding window로 분할. chunker._split_long의 jung 버전(파라미터 직접)."""
    if len(text) <= max_chars:
        return [text]
    out: list[str] = []
    step = max_chars - overlap
    i = 0
    while i < len(text):
        out.append(text[i : i + max_chars])
        if i + max_chars >= len(text):
            break
        i += step
    return out


def _merge_entities(dst: dict[str, list[str]], src: dict[str, list[str]]) -> None:
    for k, vs in src.items():
        merged = set(dst.get(k, [])) | set(vs)
        dst[k] = sorted(merged)


# ─────────────────────────────────────────────────────────────────
# 메타·leaf 수집
# ─────────────────────────────────────────────────────────────────

def _extract_book_meta(root: etree._Element) -> JungBookMeta:
    """item / level1 → book 메타."""
    meta = JungBookMeta()
    meta.book_dynasty = root.get("itemName", "") or "중국정사외국전"
    raw_id = root.get("id", "")  # "jo.d_NNNN"
    meta.book_id = _strip_prefix(raw_id)
    meta.book_name = root.get("name", "")

    l1 = root.find("level1")
    if l1 is None:
        return meta

    front = l1.find("front")
    if front is None:
        return meta
    bib = front.find("biblioData")
    if bib is None:
        return meta

    title = bib.find("title")
    if title is not None:
        meta.book_title_hanja = _txt(title.find("mainTitle"))
        meta.book_title_alt = _txt(title.find("alternative"))

    creator = bib.find("creator")
    if creator is not None:
        author = creator.find("author")
        if author is not None:
            meta.book_author = _txt(author)

    date_el = bib.find("date")
    if date_el is not None:
        dc = date_el.find("dateCreated")
        if dc is not None:
            meta.period_label = _txt(dc)

    return meta


def _chapter_title(level2: etree._Element) -> str:
    """level2 front/biblioData/title/mainTitle."""
    front = level2.find("front")
    if front is None:
        return ""
    bib = front.find("biblioData")
    if bib is None:
        return ""
    title = bib.find("title")
    if title is None:
        return ""
    return _txt(title.find("mainTitle"))


def _leaf_meta(leaf: etree._Element) -> tuple[str, str]:
    """leaf element → (article_title, subject_country)."""
    front = leaf.find("front")
    article_title = ""
    subject_country = ""
    if front is None:
        return article_title, subject_country
    bib = front.find("biblioData")
    if bib is None:
        return article_title, subject_country
    title = bib.find("title")
    if title is not None:
        article_title = _txt(title.find("mainTitle"))
    for sc in bib.findall("subjectClass"):
        schema = sc.get("schema", "")
        v = _txt(sc)
        if v and (schema == "국가" or not subject_country):
            subject_country = v
            if schema == "국가":
                break
    return article_title, subject_country


def _iter_leaves(level2: etree._Element):
    """level2 → leaf 노드 yield (level3 with own <text>, or level4 with <text>).

    leaf id, leaf element, level2 id (chapter_id) 반환.
    """
    for l3 in level2.findall("level3"):
        l4s = l3.findall("level4")
        if l4s:
            for l4 in l4s:
                if l4.find("text") is not None:
                    yield l4.get("id", ""), l4
        else:
            if l3.find("text") is not None:
                yield l3.get("id", ""), l3


def _build_jok_index(jok_path: Path) -> tuple[etree._Element, dict[str, etree._Element]]:
    """jo.k_NNNN.xml → (root, {leaf_id_stripped: leaf_element}).

    leaf id는 prefix 제거형(jo_0001_…)을 키로 함.
    """
    parser = etree.XMLParser(load_dtd=False, resolve_entities=False, recover=True)
    tree = etree.parse(str(jok_path), parser)
    root = tree.getroot()
    idx: dict[str, etree._Element] = {}
    for l1 in [root.find("level1")] if root.find("level1") is not None else []:
        for l2 in l1.findall("level2"):
            for leaf_id, leaf in _iter_leaves(l2):
                idx[_strip_prefix(leaf_id)] = leaf
    return root, idx


# ─────────────────────────────────────────────────────────────────
# 메인 처리
# ─────────────────────────────────────────────────────────────────

def chunk_book(jod_path: Path, jok_path: Path | None) -> list[JungChunk]:
    parser = etree.XMLParser(load_dtd=False, resolve_entities=False, recover=True)
    jod_tree = etree.parse(str(jod_path), parser)
    jod_root = jod_tree.getroot()
    book_meta = _extract_book_meta(jod_root)

    jok_idx: dict[str, etree._Element] = {}
    jok_meta: JungBookMeta | None = None
    if jok_path is not None and jok_path.exists():
        try:
            jok_root, jok_idx = _build_jok_index(jok_path)
            jok_meta = _extract_book_meta(jok_root)
        except etree.XMLSyntaxError as e:
            print(f"[warn] {jok_path.name}: {e}", file=sys.stderr)

    # book_name/title_alt는 jo.d/jo.k가 같지만, 누락된 쪽 보강
    if jok_meta is not None:
        if not book_meta.book_title_alt and jok_meta.book_title_alt:
            book_meta.book_title_alt = jok_meta.book_title_alt
        if not book_meta.book_author and jok_meta.book_author:
            book_meta.book_author = jok_meta.book_author

    out: list[JungChunk] = []

    l1 = jod_root.find("level1")
    if l1 is None:
        return out

    for l2 in l1.findall("level2"):
        chapter_title = _chapter_title(l2)
        chapter_id = _strip_prefix(l2.get("id", ""))

        for jod_leaf_id, jod_leaf in _iter_leaves(l2):
            article_id = _strip_prefix(jod_leaf_id)
            article_title, subject_country = _leaf_meta(jod_leaf)

            # 한문 원문 추출
            hanmun_text, hanmun_ents, hanmun_fn = _flatten_paragraphs(jod_leaf.find("text"))

            # 매칭되는 jo.k leaf
            korean_text = ""
            korean_ents: dict[str, list[str]] = {}
            korean_fn = 0
            jok_leaf = jok_idx.get(article_id)
            if jok_leaf is not None:
                korean_text, korean_ents, korean_fn = _flatten_paragraphs(jok_leaf.find("text"))
                # 국역 쪽 leaf의 article_title이 더 풍부할 수 있음 (둘 다 동일하나 fallback)
                if not article_title:
                    a2, sc2 = _leaf_meta(jok_leaf)
                    article_title = article_title or a2
                    subject_country = subject_country or sc2

            if not hanmun_text and not korean_text:
                continue

            # 검색용 결합 텍스트:
            # - 제목 컨텍스트 + 국역(한국어) + 한문 원문
            #   국역이 풍부한 한국어로 임베딩되어야 한국어 질의에 잘 hit.
            ctx_bits = []
            if book_meta.book_name:
                ctx_bits.append(book_meta.book_name)
            if chapter_title:
                ctx_bits.append(chapter_title)
            if subject_country:
                ctx_bits.append(subject_country)
            if article_title:
                ctx_bits.append(article_title)
            header = "[" + " | ".join(ctx_bits) + "]" if ctx_bits else ""

            combined_parts: list[str] = []
            if header:
                combined_parts.append(header)
            if korean_text:
                combined_parts.append("[국역] " + korean_text)
            if hanmun_text:
                combined_parts.append("[원문] " + hanmun_text)
            combined = "\n".join(combined_parts)

            # 엔티티 합집합
            ents: dict[str, list[str]] = {}
            _merge_entities(ents, hanmun_ents)
            _merge_entities(ents, korean_ents)

            pieces = _split_combined(combined)
            for j, piece in enumerate(pieces):
                cid = article_id
                if len(pieces) > 1:
                    cid = f"{article_id}_s{j}"
                out.append(
                    JungChunk(
                        chunk_id=cid,
                        article_id=article_id,
                        book_id=book_meta.book_id,
                        book_name=book_meta.book_name,
                        book_title_hanja=book_meta.book_title_hanja,
                        book_title_alt=book_meta.book_title_alt,
                        book_author=book_meta.book_author,
                        book_dynasty=book_meta.book_dynasty,
                        period_label=book_meta.period_label,
                        chapter_id=chapter_id,
                        chapter_title=chapter_title,
                        article_title=article_title,
                        subject_country=subject_country,
                        text=piece,
                        text_hanmun=hanmun_text,
                        text_korean=korean_text,
                        text_hangul_aux=to_hangul(piece),
                        has_footnotes_korean=korean_fn,
                        has_footnotes_hanmun=hanmun_fn,
                        entities=ents if j == 0 else {},
                    )
                )

    return out


def main() -> int:
    jod_files = sorted(RAW.glob("jo.d_*.xml"))
    if not jod_files:
        print(f"[err] no jo.d_*.xml in {RAW} (run jung_extract.py first)", file=sys.stderr)
        return 1
    print(f"[info] {len(jod_files)} 책(book) → {OUT}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    long_articles = 0
    with OUT.open("w", encoding="utf-8") as f:
        for jod in tqdm(jod_files, desc="chunking"):
            jok = RAW / jod.name.replace("jo.d_", "jo.k_")
            try:
                chunks = chunk_book(jod, jok if jok.exists() else None)
            except Exception as e:
                print(f"[warn] {jod.name}: {e}", file=sys.stderr)
                continue
            article_ids_seen: set[str] = set()
            for c in chunks:
                if c.chunk_id != c.article_id:
                    if c.article_id not in article_ids_seen:
                        long_articles += 1
                        article_ids_seen.add(c.article_id)
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            total += len(chunks)

    print(f"[done] wrote {total} chunks → {OUT}")
    print(f"[info] {long_articles} articles split into multiple chunks (>{MAX_CHARS} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
