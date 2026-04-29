"""정조 시대 대청 사대 관련 기사 의미 검색 (Docker Qdrant 서버).

여러 시드 질의로 sjw_full collection 검색 (king_prefix='G' 강제).
article_id 단위 dedup, reranker score 보존, 마크다운 산출.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sjw_retrieval import search  # noqa: E402

OUT = ROOT / "data" / "승정원일기" / "정조_대청사대_articles.md"

# 정조 대청 사대의 다양한 측면을 시드:
QUERIES = [
    "정조가 청나라에 보낸 사은사 동지사 진하사",
    "황제의 칙사가 한양에 와서 영접하다",
    "표문 자문 동자 전문을 청나라에 보내다",
    "동지사 정사 부사 서장관 사신단 파견",
    "청나라 황제의 조서 칙서 받음 영조의례",
    "북경 사행 별단 보고 회환",
    "중국 사신과 의례 접견 연향 다례",
    "건륭제 가경제 황제 만수성절 진하",
    "청나라에 보내는 방물 세폐 공물",
    "심양관 승덕 열하 문안 사신",
    "迎勅都監 영칙도감 모화관 영은문",
    "禮部 咨文 회답 외교 문서",
    "원접사 관반사 칙사 안내 의례",
    "황실 경조사 진위사 진향사 청에 파견",
    "관문 의주 사행로 사신 왕래",
]

SCORE_THRESHOLD = 0.35
TOP_K_PER_QUERY = 30


def main() -> int:
    aggregated: dict[str, dict] = {}
    print(f"[info] {len(QUERIES)} 시드 질의 (king_prefix=G 정조)", file=sys.stderr)
    for i, q in enumerate(QUERIES, 1):
        print(f"  [{i}/{len(QUERIES)}] {q}", file=sys.stderr)
        hits = search(q, top_k=TOP_K_PER_QUERY, filters={"king_prefix": "G"})
        for h in hits:
            if h.score < SCORE_THRESHOLD:
                continue
            aid = h.article_id
            existing = aggregated.get(aid)
            if existing is None or h.score > existing["score"]:
                aggregated[aid] = {
                    "score": h.score,
                    "chunk_id": h.chunk_id,
                    "article_id": h.article_id,
                    "year_ce": h.year_ce,
                    "day_title": h.day_title,
                    "date_western": h.date_western,
                    "ganji": h.ganji,
                    "article_title": h.article_title,
                    "article_type": h.article_type,
                    "text": h.text,
                    "matched_query": q,
                }
    print(f"[info] {len(aggregated)} 고유 기사 (score ≥ {SCORE_THRESHOLD})", file=sys.stderr)

    # 연도별 정렬
    by_year: dict[str, list[dict]] = defaultdict(list)
    for rec in aggregated.values():
        by_year[rec["year_ce"] or "?"].append(rec)

    lines: list[str] = []
    lines.append("# 정조 시대 대청(對淸) 사대(事大) 관련 기사 — 승정원일기")
    lines.append("")
    lines.append(f"BGE-M3 dense+sparse + BGE-reranker-v2-m3, Docker Qdrant 서버 `sjw_full`.  ")
    lines.append(f"king_prefix='G'(정조 1776~1800) 필터, 시드 {len(QUERIES)}개 × top_k {TOP_K_PER_QUERY}, ")
    lines.append(f"reranker score ≥ {SCORE_THRESHOLD}.  **총 {len(aggregated)}개 기사** ({len(by_year)}개 연도).")
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
            txt = (a["text"] or "").replace("\n", " ").strip()
            excerpt = txt[:200].rstrip()
            if len(txt) > 200:
                excerpt += "…"
            lines.append(
                f"- **{day}{ganji}** {title} {atype}  `score={a['score']:.3f}` `[{a['article_id']}]`"
            )
            if excerpt:
                lines.append(f"  > {excerpt}")
        lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
