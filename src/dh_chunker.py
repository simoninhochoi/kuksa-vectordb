"""동문휘고 XML → level3 단위 청크 (data/동문휘고/chunks.jsonl).

청킹 전략 (test_chunking 비교에서 확정된 Strategy A):
- 한 권에 d/m/k 세 버전이 평행 존재할 수 있음. m(표점본) > d(원본) 우선 사용.
- k(한국어 번역본)이 있으면 같은 level3_id에 매핑해 text_translation 필드에 부가.
- level3 = 외교문서 1건 = 1 청크 (800자 초과 시 sliding split).
- 첫 paragraph가 짧고(<30자) 본문이 따로 있으면 헤더로 간주, 본문(p1+)만 text로.
- 메타: title 한글/한자, sender, receiver, date(YYYY-MM-DD), 권명.
- 엔티티: <index type="이름|지명"> 추출.
"""
from __future__ import annotations

import json
import re as stdre
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path

import regex as re
from lxml import etree
from tqdm import tqdm

from chunker import _merge_entities  # 기존 모듈에서 재사용
from hanja_util import to_hangul

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "동문휘고" / "raw"
OUT = ROOT / "data" / "동문휘고" / "chunks.jsonl"

MAX_CHARS = 800
OVERLAP_CHARS = 100

# nahf.dtd index type → 카테고리 (이름·지명만 있음)
TYPE_MAP = {
    "이름": "person",
    "지명": "place",
}


@dataclass
class DhChunk:
    chunk_id: str
    level3_id: str                    # dh.X_NNNN_PPPP_QQQQ — 최종 글자 수 결정 ID
    volume_num: str                   # "0001"
    volume_name: str                  # "同文彙考 原編 卷之一"
    hanja_kind: str                   # 'm' (표점본) | 'd' (원본)
    title_hanja: str                  # alternative
    title_hangul: str                 # mainTitle (국역위 제목)
    sender: str
    receiver: str
    date_western: str                 # "YYYY-MM-DD" (99 = 일자 미상)
    date_is_leap: bool
    date_text: str                    # "順治二年 八月 二十日" 같은 원문 표기
    text: str                         # 한자 본문 (m 우선)
    text_hangul_aux: str              # hanja_util.to_hangul 음차본
    text_translation_korean: str      # k 한국어 번역 본문 (있을 때만, 첫 슬라이스만)
    entities: dict[str, list[str]] = field(default_factory=dict)


def _parse(p: Path) -> etree._Element:
    parser = etree.XMLParser(load_dtd=False, resolve_entities=False, recover=True)
    return etree.parse(str(p), parser).getroot()


def _txt_norm(el: etree._Element | None) -> str:
    if el is None:
        return ""
    return " ".join("".join(el.itertext()).split()).strip()


def _para_text_and_entities(p: etree._Element) -> tuple[str, dict[str, list[str]]]:
    """paragraph 텍스트 + index 엔티티 추출. <br/>·<annotation>·<pTitle> 처리."""
    ents: dict[str, set[str]] = defaultdict(set)
    parts: list[str] = []

    def walk(node: etree._Element) -> None:
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
                    ents[TYPE_MAP[t]].add(inner.strip())
                parts.append(inner)
            elif tag == "br":
                parts.append(" ")
            elif tag == "annotation":
                parts.append(" " + "".join(child.itertext()) + " ")
            elif tag == "pTitle":
                # 부제목은 본문에서 제외
                pass
            elif tag in ("number", "page", "illustration", "illustGroup"):
                pass
            else:
                walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(p)
    text = " ".join("".join(parts).split()).strip()
    return text, {k: sorted(v) for k, v in ents.items()}


def _split_long(text: str) -> list[str]:
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


def _parse_date(date_el: etree._Element | None) -> tuple[str, bool, str]:
    """date 태그 → (서기 YYYY-MM-DD, is_leap, 원문 표기)."""
    if date_el is None:
        return "", False, ""
    # dateSend / dateOccured / dateIssued 등 어느 것이든 첫 자식
    d = None
    for child in date_el:
        if isinstance(child.tag, str):
            d = child
            break
    if d is None:
        return "", False, ""
    raw = d.get("date", "")
    text = "".join(d.itertext()).strip()
    m = stdre.match(r"^(\d{4})-(\d{2})-(\d{2})(?:L([01]))?$", raw)
    if m:
        y, mo, da, leap = m.group(1), m.group(2), m.group(3), m.group(4)
        # 99-99 같은 경우는 일자 미상 → 그대로 보존하되 검색용 필드로
        ymd = f"{y}-{mo}-{da}"
        return ymd, leap == "1", text
    return raw, False, text


def _l3_meta(l3: etree._Element) -> dict:
    meta = {
        "level3_id": l3.get("id", ""),
        "title_hanja": "",
        "title_hangul": "",
        "sender": "",
        "receiver": "",
        "date_western": "",
        "date_is_leap": False,
        "date_text": "",
        "is_toc": False,
    }
    front = l3.find("front")
    if front is not None:
        bib = front.find("biblioData")
        if bib is not None:
            title = bib.find("title")
            if title is not None:
                meta["title_hangul"] = _txt_norm(title.find("mainTitle"))
                meta["title_hanja"] = _txt_norm(title.find("alternative"))
            if "수록문서 목록" in meta["title_hangul"]:
                meta["is_toc"] = True
            creator = bib.find("creator")
            if creator is not None:
                meta["sender"] = _txt_norm(creator.find("sender"))
                meta["receiver"] = _txt_norm(creator.find("receiver"))
            ymd, leap, dtext = _parse_date(bib.find("date"))
            meta["date_western"] = ymd
            meta["date_is_leap"] = leap
            meta["date_text"] = dtext
    return meta


def _l3_body_and_entities(l3: etree._Element) -> tuple[str, dict[str, list[str]]]:
    """level3 안의 paragraph들 → 본문 텍스트 + 엔티티 set."""
    text_el = l3.find("text")
    if text_el is None:
        return "", {}
    paras: list[tuple[str, dict[str, list[str]]]] = []
    for p in text_el.iter("paragraph"):
        t, e = _para_text_and_entities(p)
        if t:
            paras.append((t, e))
    if not paras:
        return "", {}
    # 헤더(짧은 첫 paragraph) 제거 패턴
    if len(paras) >= 2 and len(paras[0][0]) < 30 and \
       sum(len(p[0]) for p in paras[1:]) > len(paras[0][0]):
        body_paras = paras[1:]
    else:
        body_paras = paras
    body = " ".join(p[0] for p in body_paras)
    ents: dict[str, list[str]] = {}
    for _, e in body_paras:
        _merge_entities(ents, e)
    return body, ents


def _group_files() -> dict[str, dict[str, Path]]:
    """{volume_num: {kind: path}}"""
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for f in sorted(RAW.glob("dh.*.xml")):
        m = stdre.match(r"dh\.([dmk])_(\d{4})\.xml", f.name)
        if m:
            kind, num = m.group(1), m.group(2)
            groups[num][kind] = f
    return groups


def chunk_volume(vol_num: str, kinds: dict[str, Path]) -> list[DhChunk]:
    # 한자 본문: m > d 우선
    if "m" in kinds:
        h_root = _parse(kinds["m"])
        h_kind = "m"
    elif "d" in kinds:
        h_root = _parse(kinds["d"])
        h_kind = "d"
    else:
        return []

    k_root = _parse(kinds["k"]) if "k" in kinds else None

    volume_name = h_root.get("name", "")

    # k의 level3_id suffix 매핑
    k_body_by_suffix: dict[str, str] = {}
    if k_root is not None:
        for kl3 in k_root.iter("level3"):
            kid = kl3.get("id", "")
            # dh.k_0041_0010_0010 → suffix = 0010_0010
            parts = kid.split("_", 2)
            if len(parts) >= 3:
                suffix = parts[2]
                k_body, _ = _l3_body_and_entities(kl3)
                if k_body:
                    k_body_by_suffix[suffix] = k_body

    chunks: list[DhChunk] = []
    for h_l3 in h_root.iter("level3"):
        meta = _l3_meta(h_l3)
        if meta["is_toc"]:
            continue
        body, ents = _l3_body_and_entities(h_l3)
        if not body:
            continue
        # k 번역 매핑
        l3_id = meta["level3_id"]
        parts = l3_id.split("_", 2)
        suffix = parts[2] if len(parts) >= 3 else ""
        k_body = k_body_by_suffix.get(suffix, "")

        pieces = _split_long(body)
        for j, piece in enumerate(pieces):
            cid = l3_id
            if len(pieces) > 1:
                cid = f"{l3_id}_s{j}"
            chunks.append(
                DhChunk(
                    chunk_id=cid,
                    level3_id=l3_id,
                    volume_num=vol_num,
                    volume_name=volume_name,
                    hanja_kind=h_kind,
                    title_hanja=meta["title_hanja"],
                    title_hangul=meta["title_hangul"],
                    sender=meta["sender"],
                    receiver=meta["receiver"],
                    date_western=meta["date_western"],
                    date_is_leap=meta["date_is_leap"],
                    date_text=meta["date_text"],
                    text=piece,
                    text_hangul_aux=to_hangul(piece),
                    text_translation_korean=k_body if j == 0 else "",
                    entities=ents if j == 0 else {},
                )
            )
    return chunks


def main() -> int:
    groups = _group_files()
    if not groups:
        print(f"[err] no XML in {RAW}", file=sys.stderr)
        return 1
    print(f"[info] {len(groups)} unique volumes")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with OUT.open("w", encoding="utf-8") as f:
        for vol_num in tqdm(sorted(groups.keys()), desc="chunking"):
            try:
                chunks = chunk_volume(vol_num, groups[vol_num])
            except Exception as e:
                print(f"[warn] {vol_num}: {e}", file=sys.stderr)
                continue
            for c in chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            total += len(chunks)
    print(f"[done] wrote {total} chunks → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
