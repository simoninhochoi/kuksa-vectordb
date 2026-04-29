"""정조 시대 대청 사대 — 기존 두 마크다운 파일 직접 병합 (재검색 안 함).

입력:
  - data/승정원일기/정조_대청사대_articles.md       (의미검색 33건)
  - data/승정원일기/정조_사대_filtered_articles.md   (키워드+reranker 53건, '진성' 섹션만)

출력:
  - data/승정원일기/정조_대청사대_종합.md
"""
from __future__ import annotations

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
SEM_MD = ROOT / "data" / "승정원일기" / "정조_대청사대_articles.md"
KW_MD = ROOT / "data" / "승정원일기" / "정조_사대_filtered_articles.md"
OUT = ROOT / "data" / "승정원일기" / "정조_대청사대_종합.md"


# 의미검색 마크다운 파싱 — `## YYYY년` 섹션 안의 list item 형식:
#   - **승정원일기 정조00년 ... 갑인** title  `score=0.931` `[ARTICLE_ID]`
#     > excerpt
SEM_PATTERN = re.compile(
    r"^- \*\*([^*]+)\*\* (.+?)  `score=([\d.]+)` `\[([^\]]+)\]`\n  > (.+)$",
    re.MULTILINE,
)

# 키워드+reranker 마크다운 파싱 — '진성' 섹션 내부에서:
#   - **승정원일기 정조00년 ...** title  `score=0.309` `kw: 事大` `[ARTICLE_ID]`
#     > excerpt
KW_PATTERN = re.compile(
    r"^- \*\*([^*]+)\*\* (.+?)  `score=([\d.]+)` `kw: ([^`]+)` `\[([^\]]+)\]`\n  > (.+)$",
    re.MULTILINE,
)


def parse_semantic(md: str) -> dict[str, dict]:
    """의미검색 마크다운 → article_id → record."""
    records: dict[str, dict] = {}
    for m in SEM_PATTERN.finditer(md):
        day_label = m.group(1).strip()
        title_with_type = m.group(2).strip()
        score = float(m.group(3))
        aid = m.group(4).strip()
        excerpt = m.group(5).strip()

        # title_with_type 예: "勅使가 館所로 들어갔다는 迎接都監의 草記 (기사)"
        type_match = re.search(r"\s*\((기사|좌목|[^)]+)\)\s*$", title_with_type)
        if type_match:
            atype = type_match.group(1)
            title = title_with_type[: type_match.start()].strip()
        else:
            atype = ""
            title = title_with_type

        # day_label 예: "승정원일기 정조00년 10월 27일 을축"
        # 연도는 day_label 에서 정조NN년을 검색하지 말고 ID로 추정 가능. 여기선 day_label 자체 보존.
        records[aid] = {
            "article_id": aid,
            "day_label": day_label,
            "article_title": title,
            "article_type": atype,
            "sem_score": score,
            "sem_excerpt": excerpt,
        }
    return records


def parse_keyword(md: str) -> dict[str, dict]:
    """키워드+reranker 마크다운 → article_id → record. '진성' 섹션만."""
    # 진성 섹션 추출: "## 🎯 진성 사대 mention" ~ "## 🟡" 또는 "## ❌"
    m_start = re.search(r"## 🎯[^\n]*\n", md)
    if not m_start:
        return {}
    rest = md[m_start.end():]
    m_end = re.search(r"\n## (?:🟡|❌)", rest)
    section = rest[: m_end.start()] if m_end else rest

    records: dict[str, dict] = {}
    for m in KW_PATTERN.finditer(section):
        day_label = m.group(1).strip()
        title_with_type = m.group(2).strip()
        score = float(m.group(3))
        kws = m.group(4).strip()
        aid = m.group(5).strip()
        excerpt = m.group(6).strip()

        type_match = re.search(r"\s*\((기사|좌목|[^)]+)\)\s*$", title_with_type)
        if type_match:
            atype = type_match.group(1)
            title = title_with_type[: type_match.start()].strip()
        else:
            atype = ""
            title = title_with_type

        records[aid] = {
            "article_id": aid,
            "day_label": day_label,
            "article_title": title,
            "article_type": atype,
            "kw_score": score,
            "kw_keywords": kws,
            "kw_excerpt": excerpt,
        }
    return records


def article_id_to_year(aid: str) -> str:
    """SJW-G07090300-... → 1776 + 7 = 1783 (정조 즉위 1776년 = G00)."""
    m = re.match(r"^SJW-G(\d{2})\d+-\d+$", aid)
    if not m:
        return "?"
    reign = int(m.group(1))
    return str(1776 + reign)


def article_id_to_date(aid: str) -> str:
    """SJW-G07090300-... → 7년 09월 30일 (윤월 무시)."""
    m = re.match(r"^SJW-G(\d{2})(\d{2})(\d{2})\d+-\d+$", aid)
    if not m:
        # 윤월 패턴 SJW-G04061060-... 형식 시도
        m = re.match(r"^SJW-G(\d{2})0(\d)(\d{2})(\d{2})-\d+$", aid)
        if m:
            return f"정조{int(m.group(1))}년 윤{int(m.group(2)):02d}월 {int(m.group(3)):02d}일"
        return ""
    reign = int(m.group(1))
    month = int(m.group(2))
    day = int(m.group(3))
    return f"정조{reign}년 {month:02d}월 {day:02d}일"


def main() -> int:
    sem_md = SEM_MD.read_text(encoding="utf-8")
    kw_md = KW_MD.read_text(encoding="utf-8")

    sem = parse_semantic(sem_md)
    kw = parse_keyword(kw_md)
    print(f"[parse] 의미검색 {len(sem)}건, 키워드+reranker {len(kw)}건", file=sys.stderr)

    aids_sem = set(sem.keys())
    aids_kw = set(kw.keys())
    both = aids_sem & aids_kw
    sem_only = aids_sem - aids_kw
    kw_only = aids_kw - aids_sem
    total = len(aids_sem | aids_kw)
    print(f"[merge] BOTH={len(both)}, SEM-only={len(sem_only)}, KW-only={len(kw_only)}, total={total}",
          file=sys.stderr)

    # 통합 record
    all_records: dict[str, dict] = {}
    for aid in aids_sem | aids_kw:
        s = sem.get(aid, {})
        k = kw.get(aid, {})
        rec = {
            "article_id": aid,
            "year_ce": article_id_to_year(aid),
            "day_label": s.get("day_label") or k.get("day_label", ""),
            "article_title": s.get("article_title") or k.get("article_title", ""),
            "article_type": s.get("article_type") or k.get("article_type", ""),
            "sem_score": s.get("sem_score"),
            "sem_excerpt": s.get("sem_excerpt"),
            "kw_score": k.get("kw_score"),
            "kw_keywords": k.get("kw_keywords"),
            "kw_excerpt": k.get("kw_excerpt"),
            "category": "BOTH" if aid in both else ("SEM" if aid in sem_only else "KW"),
        }
        all_records[aid] = rec

    # 연도별 카테고리 분포
    by_year_cat: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in all_records.values():
        by_year_cat[r["year_ce"]][r["category"]] += 1

    # ─── 마크다운 출력 ──────────────────────────────────
    lines = []
    lines.append("# 정조 시대 대청 사대 — 통합 종합 (의미검색 ∪ 키워드+reranker)")
    lines.append("")
    lines.append("기존 두 검색 결과를 article_id 기준 병합:")
    lines.append(f"- ① **의미검색** (BGE-M3 + reranker, 15시드): {len(sem)}건")
    lines.append(f"- ② **키워드+reranker** (`'事大'`/`'사대'` grep + reranker score ≥0.10): {len(kw)}건")
    lines.append("")
    lines.append(f"중복 제거 후 **총 {total}건** ({len(by_year_cat)}개 연도)")
    lines.append("")
    lines.append("**카테고리**:")
    lines.append(f"- 🟢 **BOTH** ({len(both)}건): 양쪽 모두 hit — 최고 신뢰도")
    lines.append(f"- 🔵 **SEMANTIC-only** ({len(sem_only)}건): 사대 의례·실무 (사대 단어 직접 언급 없음)")
    lines.append(f"- 🟣 **KEYWORD-only** ({len(kw_only)}건): 사대 담론·관용구 (이벤트형 아님)")
    lines.append("")

    # 연도별 분포 표
    lines.append("## 연도별 분포")
    lines.append("")
    lines.append("| 연도 | BOTH 🟢 | SEM 🔵 | KW 🟣 | 계 |")
    lines.append("|---|---:|---:|---:|---:|")
    grand = {"BOTH": 0, "SEM": 0, "KW": 0}
    for year in sorted(by_year_cat.keys()):
        d = by_year_cat[year]
        b, s, k = d.get("BOTH", 0), d.get("SEM", 0), d.get("KW", 0)
        grand["BOTH"] += b; grand["SEM"] += s; grand["KW"] += k
        lines.append(f"| {year} | {b} | {s} | {k} | {b+s+k} |")
    lines.append(f"| **계** | **{grand['BOTH']}** | **{grand['SEM']}** | **{grand['KW']}** "
                  f"| **{sum(grand.values())}** |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 본문 — 연도별 → 카테고리별
    lines.append("## 통합 기사 목록 (연도순, 카테고리 표시)")
    lines.append("")
    by_year: dict[str, list[dict]] = defaultdict(list)
    for r in all_records.values():
        by_year[r["year_ce"]].append(r)

    cat_emoji = {"BOTH": "🟢", "SEM": "🔵", "KW": "🟣"}
    cat_order = {"BOTH": 0, "SEM": 1, "KW": 2}

    for year in sorted(by_year.keys()):
        arts = sorted(by_year[year], key=lambda x: (cat_order[x["category"]], x["article_id"]))
        lines.append(f"### {year}년 — {len(arts)}건")
        lines.append("")
        for a in arts:
            emoji = cat_emoji[a["category"]]
            title = a["article_title"] or "(제목 없음)"
            atype = f"({a['article_type']})" if a["article_type"] else ""
            day = a["day_label"] or article_id_to_date(a["article_id"]) or "?"

            # 점수 표기
            score_parts = []
            if a["sem_score"] is not None:
                score_parts.append(f"sem={a['sem_score']:.3f}")
            if a["kw_score"] is not None:
                score_parts.append(f"kw={a['kw_score']:.3f}")
            if a["kw_keywords"]:
                score_parts.append(f"hit:{a['kw_keywords']}")
            score_tag = "  ".join(f"`{p}`" for p in score_parts)

            lines.append(f"- {emoji} **{day}** {title} {atype}  {score_tag} `[{a['article_id']}]`")

            # 발췌 — 양쪽 다 있으면 키워드 발췌가 더 specific (사대 컨텍스트), 의미 발췌는 events
            excerpt = a.get("kw_excerpt") or a.get("sem_excerpt") or ""
            if excerpt:
                lines.append(f"  > {excerpt}")
        lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
