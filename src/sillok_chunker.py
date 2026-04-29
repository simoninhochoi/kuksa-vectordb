"""조선왕조실록 XML → 기사 단위 청크 (data/조선왕조실록/chunks.jsonl).

구조:
  <level1 id="waa">                            — 太祖實錄 전체 (파일 여러 개)
    <level2 id="waa_101">                      — 元年 (연 단위 권)
      <level3 id="waa_101070">                 — 秋七月 (월)
        <level4 id="waa_10107017">             — 7월 17일 (일)
          <level5 id="waa_10107017_001">       — ★ 개별 기사 (biblioData type="T")
            <front><biblioData type="T">...</biblioData></front>
            <text><content><paragraph>...</paragraph></content></text>

總序(_000) / 附錄(_200)는 level3에 직접 기사가 들어감 (얕은 트리).

청크 단위: biblioData type="T" 노드 1개 = 기사 1건.
조상 "L"(디렉터리) 노드에서 sillok/volume/month/day/date 컨텍스트 누적.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

import regex as re
from lxml import etree
from tqdm import tqdm

from chunker import (
    MIN_CHARS, MAX_CHARS,
    _para_text_and_entities, _merge_entities, _split_long,
)
from hanja_util import to_hangul

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "조선왕조실록" / "raw"
OUT = ROOT / "data" / "조선왕조실록" / "chunks.jsonl"


@dataclass
class Ctx:
    """level을 내려가면서 누적되는 디렉터리 컨텍스트."""
    sillok_id: str = ""               # "waa" 등
    sillok_title_hanja: str = ""      # "太祖實錄"
    sillok_title_hangul: str = ""
    king: str = ""                    # "太祖" (sillok_title_hanja에서 實錄 제거)
    volume_id: str = ""               # level2 id, 예: "waa_101"
    volume_title: str = ""            # level2 mainTitle, 예: "太祖實錄 元年"
    year_ce: str = ""                 # level2의 date[서기]
    month_title: str = ""             # level3 mainTitle, 예: "太祖實錄 元年 秋七月"
    day_title: str = ""               # level4 mainTitle, 예: "太祖 元年 7月 17日"
    date_western: str = ""            # "YYYY-MM-DD" (leap 플래그 분리)
    date_is_leap: bool = False
    ganji: str = ""
    reign_year: str = ""              # "태조 1년 7월 17일"
    china_era: str = ""               # "명 홍무(洪武) 25년 7월 17일"


@dataclass
class SillokChunk:
    chunk_id: str
    article_id: str                   # 기사 노드 id (level5 또는 level3)
    sillok_id: str
    sillok_title_hanja: str
    sillok_title_hangul: str
    king: str
    volume_id: str
    volume_title: str
    year_ce: str
    month_title: str
    day_title: str
    date_western: str
    date_is_leap: bool
    ganji: str
    reign_year: str
    china_era: str
    article_title: str                # 기사 mainTitle (한글 제목)
    subject_classes: list[str]
    source_refs: list[str]            # ["太祖實錄 1책 1권 37장 A면", ...]
    text: str
    text_hangul_aux: str
    entities: dict[str, list[str]] = field(default_factory=dict)


def _txt(el: etree._Element | None) -> str:
    """공백은 단일 스페이스로 정규화. 한글 제목 가독성 유지를 위해 공백 제거 X."""
    if el is None:
        return ""
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _find_biblio(node: etree._Element) -> etree._Element | None:
    """node의 front/biblioData 반환. 없으면 None."""
    front = node.find("front")
    if front is None:
        return None
    return front.find("biblioData")


def _parse_date(bib: etree._Element) -> tuple[str, bool, str, str, str, str]:
    """biblioData/date → (서기_YYYY-MM-DD, is_leap, 간지, 재위연도, 중국연호, year_ce_only).

    서기 attribute 형식: "1392-07-17L0" (L0 = 평달, L1 = 윤달) 또는 "1392" (연 단위).
    """
    date_el = bib.find("date")
    if date_el is None:
        return "", False, "", "", "", ""
    ymd = ""
    is_leap = False
    year_ce = ""
    ganji = ""
    reign = ""
    china = ""
    for d in date_el.findall("dateOccured"):
        t = d.get("type")
        if t == "서기":
            raw = d.get("date", "") or _txt(d)
            # "1392-07-17L0" → date="1392-07-17", leap=False
            m = re.match(r"^(\d{4}(?:-\d{2}(?:-\d{2})?)?)(?:L([01]))?$", raw)
            if m:
                ymd_candidate = m.group(1)
                is_leap = m.group(2) == "1"
                if len(ymd_candidate) == 4:
                    year_ce = ymd_candidate
                else:
                    ymd = ymd_candidate
                    year_ce = ymd_candidate[:4]
        elif t == "간지":
            ganji = _txt(d)
        elif t == "재위연도":
            reign = _txt(d)
        elif t == "중국연호":
            china = _txt(d)
    return ymd, is_leap, ganji, reign, china, year_ce


def _update_ctx_from_level(node: etree._Element, level_num: int, ctx: Ctx) -> Ctx:
    """L(디렉터리) 노드의 bibliographic 정보를 ctx에 누적해 새 ctx 반환."""
    bib = _find_biblio(node)
    new = Ctx(**asdict(ctx))

    if level_num == 1:
        new.sillok_id = node.get("id", new.sillok_id)
        if bib is not None:
            title = bib.find("title")
            if title is not None:
                main = _txt(title.find("mainTitle"))
                if main:
                    new.sillok_title_hanja = main
                    new.sillok_title_hangul = to_hangul(main)
                    # "太祖實錄" → "太祖" (實錄 접미사 제거)
                    new.king = re.sub(r"實錄$", "", main)
    elif level_num == 2:
        new.volume_id = node.get("id", new.volume_id)
        if bib is not None:
            title = bib.find("title")
            if title is not None:
                new.volume_title = _txt(title.find("mainTitle")) or new.volume_title
            ymd, leap, ganji, reign, china, year_ce = _parse_date(bib)
            if year_ce:
                new.year_ce = year_ce
            # 일별 컨텍스트는 level4에서 채워짐
    elif level_num == 3:
        if bib is not None:
            title = bib.find("title")
            if title is not None:
                new.month_title = _txt(title.find("mainTitle")) or ""
    elif level_num == 4:
        if bib is not None:
            title = bib.find("title")
            if title is not None:
                new.day_title = _txt(title.find("mainTitle")) or ""
            ymd, leap, ganji, reign, china, year_ce = _parse_date(bib)
            if ymd:
                new.date_western = ymd
                new.date_is_leap = leap
            if ganji:
                new.ganji = ganji
            if reign:
                new.reign_year = reign
            if china:
                new.china_era = china
            if year_ce and not new.year_ce:
                new.year_ce = year_ce
    return new


def _article_to_chunks(node: etree._Element, ctx: Ctx) -> list[SillokChunk]:
    """biblioData type="T" 노드 → 1개 이상의 SillokChunk."""
    bib = _find_biblio(node)
    article_id = node.get("id", "noid")
    article_title = ""
    subject_classes: list[str] = []
    source_refs: list[str] = []
    # 기사 자체 날짜(있으면 override)
    local_date = ""
    local_leap = False
    local_year = ""

    if bib is not None:
        title = bib.find("title")
        if title is not None:
            article_title = _txt(title.find("mainTitle"))
        for sc in bib.findall("subjectClass"):
            t = _txt(sc)
            if t:
                subject_classes.append(t)
        for src in bib.findall("source"):
            src_title = _txt(src.find("mainTitle"))
            page = src.find("page")
            page_str = page.get("begin", "") if page is not None else ""
            if src_title:
                source_refs.append(f"{src_title} {page_str}".strip())
        ymd, leap, _, _, _, year_ce = _parse_date(bib)
        if ymd:
            local_date = ymd
            local_leap = leap
            local_year = year_ce

    # 기사 본문: paragraph들 수집
    text_el = node.find("text")
    paras: list[tuple[str, dict[str, list[str]]]] = []
    if text_el is not None:
        for content in text_el.iter("content"):
            for p in content.findall("paragraph"):
                t, ents = _para_text_and_entities(p)
                if t:
                    paras.append((t, ents))

    if not paras:
        return []

    # 짧은 단락 연쇄 병합
    merged: list[tuple[str, dict[str, list[str]]]] = []
    buf_text = ""
    buf_ents: dict[str, list[str]] = {}
    for text, ents in paras:
        if not buf_text:
            buf_text = text
            buf_ents = dict(ents)
        elif len(buf_text) < MIN_CHARS:
            buf_text = (buf_text + " " + text).strip()
            _merge_entities(buf_ents, ents)
        else:
            merged.append((buf_text, buf_ents))
            buf_text = text
            buf_ents = dict(ents)
    if buf_text:
        merged.append((buf_text, buf_ents))

    # 최종 청크 생성 (기사 전체 엔티티는 모든 단락 병합)
    out: list[SillokChunk] = []
    global_ents: dict[str, list[str]] = {}
    for _, ents in paras:
        _merge_entities(global_ents, ents)

    # 기사 날짜 override
    eff_date = local_date or ctx.date_western
    eff_leap = local_leap or ctx.date_is_leap
    eff_year = local_year or ctx.year_ce

    flat_parts: list[str] = []
    for text, _ in merged:
        flat_parts.append(text)
    # 너무 길면 슬라이딩; 기사가 짧으면 단락 보존
    joined = " ".join(flat_parts)
    pieces = _split_long(joined)

    for j, piece in enumerate(pieces):
        cid = article_id
        if len(pieces) > 1:
            cid = f"{article_id}_s{j}"
        out.append(
            SillokChunk(
                chunk_id=cid,
                article_id=article_id,
                sillok_id=ctx.sillok_id,
                sillok_title_hanja=ctx.sillok_title_hanja,
                sillok_title_hangul=ctx.sillok_title_hangul,
                king=ctx.king,
                volume_id=ctx.volume_id,
                volume_title=ctx.volume_title,
                year_ce=eff_year,
                month_title=ctx.month_title,
                day_title=ctx.day_title,
                date_western=eff_date,
                date_is_leap=eff_leap,
                ganji=ctx.ganji,
                reign_year=ctx.reign_year,
                china_era=ctx.china_era,
                article_title=article_title,
                subject_classes=subject_classes,
                source_refs=source_refs,
                text=piece,
                text_hangul_aux=to_hangul(piece),
                entities=global_ents if j == 0 else {},
            )
        )
    return out


def _descend(node: etree._Element, level_num: int, ctx: Ctx) -> list[SillokChunk]:
    """level N 노드 방문. "T" 면 기사 추출, "L" 면 ctx 갱신 후 자식 재귀."""
    chunks: list[SillokChunk] = []
    bib = _find_biblio(node)
    is_article = bib is not None and bib.get("type") == "T"

    if is_article:
        chunks.extend(_article_to_chunks(node, ctx))

    # "L" 노드는 ctx 갱신 (level1~4만)
    if level_num <= 4:
        ctx = _update_ctx_from_level(node, level_num, ctx)

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
        chunks.extend(_descend(child, child_depth, ctx))
    return chunks


def _build_sillok_meta_map() -> dict[str, tuple[str, str, str]]:
    """RAW의 _000.xml(level1 root)들에서 sillok_id → (king, title_hanja, title_hangul) 추출.

    실록 본문 파일은 level2-root이라 그 안에 level1 메타가 없음.
    _000.xml 만이 level1 root + 전체 실록 제목을 가짐 → 매핑을 미리 빌드해 둠.
    """
    meta: dict[str, tuple[str, str, str]] = {}
    parser = etree.XMLParser(load_dtd=False, resolve_entities=False, recover=True)
    for xf in sorted(RAW.glob("2nd_*_000.xml")):
        try:
            tree = etree.parse(str(xf), parser)
        except etree.XMLSyntaxError:
            continue
        root = tree.getroot()
        tag = etree.QName(root).localname if isinstance(root.tag, str) else ""
        if tag != "level1":
            continue
        sid = root.get("id", "")
        if not sid:
            continue
        bib = _find_biblio(root)
        title_hanja = ""
        if bib is not None:
            t = bib.find("title")
            if t is not None:
                title_hanja = _txt(t.find("mainTitle"))
        title_hangul = to_hangul(title_hanja) if title_hanja else ""
        king = re.sub(r"實錄$", "", title_hanja)
        meta[sid] = (king, title_hanja, title_hangul)
    return meta


_SILLOK_META: dict[str, tuple[str, str, str]] = {}


def chunk_file(xml_path: Path) -> list[SillokChunk]:
    parser = etree.XMLParser(load_dtd=False, resolve_entities=False, recover=True)
    tree = etree.parse(str(xml_path), parser)
    root = tree.getroot()
    tag = etree.QName(root).localname if isinstance(root.tag, str) else ""
    try:
        depth = int(tag[5:]) if tag.startswith("level") else 1
    except ValueError:
        depth = 1

    ctx = Ctx()
    # level2+ root (예: <level2 id="waa_101">) 인 경우 sillok 메타를 외부 매핑에서 시드
    if depth >= 2:
        root_id = root.get("id", "")
        if root_id:
            sid = root_id.split("_", 1)[0]
            ctx.sillok_id = sid
            if sid in _SILLOK_META:
                king, th, thg = _SILLOK_META[sid]
                ctx.king = king
                ctx.sillok_title_hanja = th
                ctx.sillok_title_hangul = thg
    return _descend(root, depth, ctx)


def main() -> int:
    xml_files = sorted(RAW.glob("2nd_*.xml"))
    if not xml_files:
        print(f"[err] no XML in {RAW}", file=sys.stderr)
        return 1
    print(f"[info] {len(xml_files)} xml files → {OUT}")

    # _000.xml에서 sillok 메타 매핑 미리 빌드
    global _SILLOK_META
    _SILLOK_META = _build_sillok_meta_map()
    print(f"[info] sillok meta map: {len(_SILLOK_META)} entries")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with OUT.open("w", encoding="utf-8") as f:
        for xf in tqdm(xml_files, desc="chunking"):
            try:
                chunks = chunk_file(xf)
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
