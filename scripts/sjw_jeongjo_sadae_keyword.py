"""정조 시대 chunks에서 '事大' / '사대' 키워드 직접 검색.

chunks.jsonl 의 정조(king_prefix='G') 청크만 추출해 텍스트에 키워드 포함 기사 수집.
의미검색 결과(33건)와 비교용. 출처는 sjw_full server 가 아닌 chunks.jsonl 직접.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "승정원일기" / "chunks.jsonl"
OUT = ROOT / "data" / "승정원일기" / "정조_사대_키워드_articles.md"

KEYWORDS = ["事大", "사대"]


def main() -> int:
    if not CHUNKS.exists():
        print(f"[err] {CHUNKS}", file=sys.stderr)
        return 1

    # article_id 단위 dedup, 어떤 키워드가 hit 됐는지 기록
    articles: dict[str, dict] = {}
    n_jeongjo = 0
    print("[info] 정조(G) chunks 스캔 + 사대 키워드 매칭", file=sys.stderr)
    with CHUNKS.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c.get("king_prefix") != "G":
                continue
            n_jeongjo += 1
            text = c.get("text", "")
            matched = [k for k in KEYWORDS if k in text]
            if not matched:
                continue
            aid = c["article_id"]
            if aid not in articles:
                articles[aid] = {
                    "article_id": aid,
                    "chunk_id": c["chunk_id"],
                    "year_ce": c.get("year_ce", ""),
                    "day_title": c.get("day_title", ""),
                    "date_western": c.get("date_western", ""),
                    "ganji": c.get("ganji", ""),
                    "article_title": c.get("article_title", ""),
                    "article_type": c.get("article_type", ""),
                    "text": text,
                    "matched": set(matched),
                }
            else:
                articles[aid]["matched"].update(matched)

    print(f"[info] 정조 chunks 총 {n_jeongjo:,}개, 키워드 hit {len(articles)}개 article", file=sys.stderr)

    by_year: dict[str, list[dict]] = defaultdict(list)
    for rec in articles.values():
        by_year[rec["year_ce"] or "?"].append(rec)

    lines = []
    lines.append("# 정조 시대 — '事大'/'사대' 키워드 검색 (승정원일기)")
    lines.append("")
    lines.append(f"chunks.jsonl 직접 grep, king_prefix='G'(정조) 한정.  ")
    lines.append(f"키워드: {', '.join(repr(k) for k in KEYWORDS)}.  ")
    lines.append(f"**총 {len(articles)}개 기사** ({len(by_year)}개 연도).")
    lines.append("")
    lines.append("정렬: 연도 → 날짜.")
    lines.append("")
    lines.append("---")
    lines.append("")

    for year in sorted(by_year.keys()):
        arts = sorted(by_year[year], key=lambda x: (x["date_western"] or "", x["chunk_id"]))
        lines.append(f"## {year}년 — {len(arts)}건")
        lines.append("")
        for a in arts:
            title = a["article_title"] or "(제목 없음)"
            atype = f"({a['article_type']})" if a["article_type"] else ""
            day = a["day_title"] or a["date_western"] or "?"
            ganji = f" {a['ganji']}" if a["ganji"] else ""
            sigs = "/".join(sorted(a["matched"]))
            txt = (a["text"] or "").replace("\n", " ").strip()

            # 키워드 주변 ±60자 발췌
            lo_idx = -1
            for k in a["matched"]:
                idx = txt.find(k)
                if idx >= 0 and (lo_idx < 0 or idx < lo_idx):
                    lo_idx = idx
            if lo_idx >= 0:
                start = max(0, lo_idx - 60)
                end = min(len(txt), lo_idx + 200)
                excerpt = ("…" if start > 0 else "") + txt[start:end] + ("…" if end < len(txt) else "")
            else:
                excerpt = txt[:200]

            lines.append(
                f"- **{day}{ganji}** {title} {atype}  `keyword: {sigs}` `[{a['article_id']}]`"
            )
            if excerpt:
                lines.append(f"  > {excerpt}")
        lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
