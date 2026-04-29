"""86건 article 전체 한문 원문 + 메타를 JSON으로 추출.

병합된 마크다운에서 article_id 86개 추출 → chunks.jsonl 에서 같은 article_id 모든
슬라이스 병합 + 헤더 prefix 제거 → 완전한 한문 원문 복원.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parents[1]
MERGED_MD = ROOT / "data" / "승정원일기" / "정조_대청사대_종합.md"
CHUNKS = ROOT / "data" / "승정원일기" / "chunks.jsonl"
OUT = ROOT / "data" / "승정원일기" / "정조_대청사대_원문.json"

# 마크다운에서 article_id 추출
ID_PATTERN = re.compile(r"\[(SJW-G\d+-\d+)\]")
# 카테고리 (이모지 + sem_score / kw_score 추출)
LINE_PATTERN = re.compile(
    r"^- (🟢|🔵|🟣) \*\*([^*]+)\*\* (.+?)\s+(`sem=[\d.]+`(?:  `kw=[\d.]+`)?(?:\s+`hit:[^`]+`)?|`kw=[\d.]+`(?:\s+`hit:[^`]+`)?)\s+`\[(SJW-G\d+-\d+)\]`",
    re.MULTILINE,
)
HEADER_PATTERN = re.compile(r"^\s*\[정조 \| [^\]]+\]\s*", re.MULTILINE)


def parse_merged_md() -> list[dict]:
    """병합 MD → article_id 순서 + 카테고리 추출."""
    md = MERGED_MD.read_text(encoding="utf-8")
    # 본문 (## 통합 기사 목록 이후만)
    main_idx = md.find("## 통합 기사 목록")
    if main_idx < 0:
        return []
    body = md[main_idx:]

    records = []
    seen = set()
    # 모든 list item line 파싱
    for line in body.split("\n"):
        m = re.match(
            r"^- (🟢|🔵|🟣) \*\*([^*]+)\*\* (.+?)\s+`(sem|kw)=[\d.]+`",
            line,
        )
        # ID는 별도 추출 (line 끝에 위치)
        id_match = re.search(r"`\[(SJW-G\d+-\d+)\]`", line)
        if not id_match:
            continue
        aid = id_match.group(1)
        if aid in seen:
            continue
        seen.add(aid)

        emoji_match = re.match(r"^- (🟢|🔵|🟣)", line)
        emoji = emoji_match.group(1) if emoji_match else "?"
        cat = {"🟢": "BOTH", "🔵": "SEM", "🟣": "KW"}.get(emoji, "?")

        # day_label 과 title
        m2 = re.match(r"^- [🟢🔵🟣] \*\*([^*]+)\*\* (.+?)\s+`", line)
        day_label = m2.group(1).strip() if m2 else ""
        title_with_type = m2.group(2).strip() if m2 else ""
        type_match = re.search(r"\s*\(([^)]+)\)\s*$", title_with_type)
        if type_match:
            atype = type_match.group(1)
            title = title_with_type[: type_match.start()].strip()
        else:
            atype = ""
            title = title_with_type

        # 점수 추출
        sem_m = re.search(r"`sem=([\d.]+)`", line)
        kw_m = re.search(r"`kw=([\d.]+)`", line)
        hit_m = re.search(r"`hit:([^`]+)`", line)

        records.append({
            "article_id": aid,
            "category": cat,
            "day_label": day_label,
            "article_title": title,
            "article_type": atype,
            "sem_score": float(sem_m.group(1)) if sem_m else None,
            "kw_score": float(kw_m.group(1)) if kw_m else None,
            "kw_keywords": hit_m.group(1).strip() if hit_m else None,
        })
    return records


def load_chunks_by_aid() -> dict[str, list[dict]]:
    """chunks.jsonl 의 정조 청크를 article_id 별로 모음."""
    by_aid: dict[str, list[dict]] = defaultdict(list)
    with CHUNKS.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c.get("king_prefix") != "G":
                continue
            by_aid[c["article_id"]].append(c)
    # chunk_id 순으로 정렬 (s0, s1, s2...)
    for aid in by_aid:
        by_aid[aid].sort(key=lambda x: x["chunk_id"])
    return by_aid


def reconstruct_hanmun(slices: list[dict]) -> tuple[str, str]:
    """
    같은 article_id 의 청크들 병합 → 완전한 한문 원문.
    슬라이스가 있는 경우 (chunk_id != article_id) 800자 sliding window with 100 overlap 으로 분할됐으므로
    겹치는 부분 제거하면서 합침.
    헤더 prefix `[정조 | 승정원일기 ... | 제목]` 도 제거.
    Returns: (full_text_no_header, header)
    """
    if not slices:
        return "", ""
    if len(slices) == 1:
        text = slices[0]["text"]
        m = HEADER_PATTERN.match(text)
        if m:
            header = m.group(0).strip()
            body = text[m.end():].strip()
        else:
            header = ""
            body = text.strip()
        return body, header

    # 여러 슬라이스 → overlap 제거하며 합침
    # 첫 슬라이스에서 헤더 추출
    first_text = slices[0]["text"]
    m = HEADER_PATTERN.match(first_text)
    if m:
        header = m.group(0).strip()
        first_body = first_text[m.end():].strip()
    else:
        header = ""
        first_body = first_text.strip()

    OVERLAP = 100   # chunker.py 의 OVERLAP_CHARS
    parts = [first_body]
    for s in slices[1:]:
        t = s["text"]
        # 슬라이스 2번째 이후도 같은 헤더가 prefix 로 또 들어가 있을 수 있음
        m2 = HEADER_PATTERN.match(t)
        body = t[m2.end():].strip() if m2 else t.strip()
        # body 의 앞 OVERLAP 자는 이전 슬라이스의 끝과 중복일 가능성
        # body 와 직전 part 의 끝부분이 일치하는 만큼 잘라냄
        prev = parts[-1]
        # 최대 OVERLAP 길이까지 시도
        max_check = min(OVERLAP + 20, len(body), len(prev))
        cut = 0
        for L in range(max_check, 30, -1):
            if body[:L] == prev[-L:]:
                cut = L
                break
        parts.append(body[cut:] if cut else body)
    return "".join(parts).strip(), header


def main() -> int:
    records = parse_merged_md()
    print(f"[parse] 병합 MD에서 {len(records)} articles", file=sys.stderr)

    by_aid = load_chunks_by_aid()
    print(f"[load] chunks.jsonl 정조 article: {len(by_aid):,}", file=sys.stderr)

    out_records = []
    for r in records:
        aid = r["article_id"]
        slices = by_aid.get(aid, [])
        if not slices:
            print(f"[warn] {aid} 청크 없음", file=sys.stderr)
            continue
        body, header = reconstruct_hanmun(slices)
        rep = slices[0]   # 메타용 첫 슬라이스
        out_records.append({
            "article_id": aid,
            "category": r["category"],
            "day_label": r["day_label"] or rep.get("day_title", ""),
            "year_ce": rep.get("year_ce", ""),
            "ganji": rep.get("ganji", ""),
            "article_title": r["article_title"] or rep.get("article_title", ""),
            "article_type": r["article_type"] or rep.get("article_type", ""),
            "reign_year": rep.get("reign_year", ""),
            "date_western": rep.get("date_western", ""),
            "header": header,
            "hanmun": body,
            "n_slices": len(slices),
            "char_len": len(body),
            "sem_score": r["sem_score"],
            "kw_score": r["kw_score"],
            "kw_keywords": r["kw_keywords"],
        })

    OUT.write_text(json.dumps(out_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {OUT}", file=sys.stderr)

    # 통계
    total_chars = sum(r["char_len"] for r in out_records)
    print(f"\n[stats] {len(out_records)} articles, "
          f"총 한문 원문 {total_chars:,} 자, 평균 {total_chars // len(out_records):,} 자")
    by_cat = defaultdict(int)
    for r in out_records:
        by_cat[r["category"]] += 1
    print(f"  카테고리: {dict(by_cat)}")
    by_year = defaultdict(int)
    for r in out_records:
        by_year[r["year_ce"]] += 1
    print(f"  연도 수: {len(by_year)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
