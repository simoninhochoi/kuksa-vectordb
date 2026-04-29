"""정조 시대 대청 사대 — 의미검색 + 키워드+reranker 통합 종합.

두 파이프라인 공동 실행:
  - 의미검색 (15 시드 × top_k 30, threshold 0.35) → 영접 의례·실무 hit
  - 키워드 grep + reranker (score ≥ 0.10) → 사대 담론·문서 hit

article_id 단위 dedup, 카테고리:
  - BOTH: 양쪽 모두 hit (최고 신뢰도)
  - SEMANTIC-ONLY: 의례/사건 (사대 단어 직접 언급 없음)
  - KEYWORD-ONLY: 사대 담론·관용구 (이벤트형 아님)
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
OUT = ROOT / "data" / "승정원일기" / "정조_대청사대_종합.md"

sys.path.insert(0, str(ROOT / "src"))

# ─── 의미검색 시드 ──────────────────────────────────
SEMANTIC_QUERIES = [
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
SEMANTIC_THRESHOLD = 0.35
SEMANTIC_TOP_K = 30

# ─── 키워드 + reranker ──────────────────────────────────
KEYWORDS = ["事大", "사대"]
KEYWORD_CONTEXT_RADIUS = 80
RERANK_QUERY = (
    "事大之禮 事大之誠 事大交隣 承文院 事大衙門 "
    "對淸 朝貢 表箋 咨文 勅使 迎勅 祭天 尊周大義 "
    "조선이 中國(淸)에 사대 외교를 행하는 의례와 정신"
)
RERANK_THRESHOLD = 0.10


def run_semantic() -> dict[str, dict]:
    from sjw_retrieval import search

    print("[1] 의미검색 — 15 시드 × top_k 30", file=sys.stderr)
    aggregated: dict[str, dict] = {}
    for i, q in enumerate(SEMANTIC_QUERIES, 1):
        print(f"   [{i}/{len(SEMANTIC_QUERIES)}] {q[:40]}…", file=sys.stderr)
        hits = search(q, top_k=SEMANTIC_TOP_K, filters={"king_prefix": "G"})
        for h in hits:
            if h.score < SEMANTIC_THRESHOLD:
                continue
            aid = h.article_id
            if aid not in aggregated or h.score > aggregated[aid]["score"]:
                aggregated[aid] = {
                    "article_id": aid,
                    "year_ce": h.year_ce,
                    "day_title": h.day_title,
                    "date_western": h.date_western,
                    "ganji": h.ganji,
                    "article_title": h.article_title,
                    "article_type": h.article_type,
                    "text": h.text,
                    "score": h.score,
                    "matched_query": q,
                }
    print(f"   → {len(aggregated)} articles", file=sys.stderr)
    return aggregated


def run_keyword_rerank() -> dict[str, dict]:
    from FlagEmbedding import FlagReranker

    print("[2] 키워드 grep — '事大' / '사대' 정조 chunks", file=sys.stderr)
    candidates: list[dict] = []
    seen: set[str] = set()
    with CHUNKS.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c.get("king_prefix") != "G":
                continue
            text = c.get("text", "")
            matched = [k for k in KEYWORDS if k in text]
            if not matched:
                continue
            aid = c["article_id"]
            if aid in seen:
                continue
            seen.add(aid)

            idx = text.find(matched[0])
            start = max(0, idx - KEYWORD_CONTEXT_RADIUS)
            end = min(len(text), idx + len(matched[0]) + KEYWORD_CONTEXT_RADIUS)
            ctx = ("…" if start > 0 else "") + text[start:end] + ("…" if end < len(text) else "")
            candidates.append({
                "article_id": aid,
                "year_ce": c.get("year_ce", ""),
                "day_title": c.get("day_title", ""),
                "date_western": c.get("date_western", ""),
                "ganji": c.get("ganji", ""),
                "article_title": c.get("article_title", ""),
                "article_type": c.get("article_type", ""),
                "text": text,
                "matched_keywords": matched,
                "context": ctx,
            })
    print(f"   → {len(candidates)} 후보", file=sys.stderr)

    print("[2] reranker 적합도 score", file=sys.stderr)
    reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, device="cuda")
    pairs = [[RERANK_QUERY, c["context"]] for c in candidates]
    scores = reranker.compute_score(pairs, normalize=True, batch_size=32)
    if not isinstance(scores, list):
        scores = [scores]
    for c, sc in zip(candidates, scores):
        c["score"] = float(sc)

    high = [c for c in candidates if c["score"] >= RERANK_THRESHOLD]
    print(f"   → {len(high)} 진성 (score ≥ {RERANK_THRESHOLD})", file=sys.stderr)
    return {c["article_id"]: c for c in high}


def main() -> int:
    semantic = run_semantic()
    keyword = run_keyword_rerank()

    aids_sem = set(semantic.keys())
    aids_kw = set(keyword.keys())
    both = aids_sem & aids_kw
    sem_only = aids_sem - aids_kw
    kw_only = aids_kw - aids_sem
    print(f"\n[merge] BOTH={len(both)}, SEMANTIC-only={len(sem_only)}, KEYWORD-only={len(kw_only)}",
          file=sys.stderr)

    # 통합 record (양쪽 정보 결합)
    all_records: dict[str, dict] = {}
    for aid in aids_sem | aids_kw:
        s = semantic.get(aid)
        k = keyword.get(aid)
        rec = (s or k).copy()
        rec["sem_score"] = s["score"] if s else None
        rec["kw_score"] = k["score"] if k else None
        rec["category"] = "BOTH" if aid in both else ("SEM" if aid in sem_only else "KW")
        if k and "context" in k:
            rec["kw_context"] = k["context"]
        if k and "matched_keywords" in k:
            rec["matched_keywords"] = k["matched_keywords"]
        all_records[aid] = rec

    # 연도별 분포
    by_cat_year: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rec in all_records.values():
        by_cat_year[rec["year_ce"] or "?"][rec["category"]] += 1

    # ─── 마크다운 출력 ──────────────────────────────────
    lines = []
    lines.append("# 정조 시대 대청 사대 — 통합 종합 (의미검색 ∪ 키워드+reranker)")
    lines.append("")
    lines.append("**파이프라인**:")
    lines.append("- ① **의미검색**: BGE-M3 dense+sparse + reranker, "
                  f"king_prefix=G, {len(SEMANTIC_QUERIES)} 시드 × top_k {SEMANTIC_TOP_K}, "
                  f"score ≥ {SEMANTIC_THRESHOLD}")
    lines.append("- ② **키워드+reranker**: chunks.jsonl 정조 청크 grep "
                  f"(`'事大'`/`'사대'`) → reranker 적합도 score ≥ {RERANK_THRESHOLD}")
    lines.append("")
    lines.append("**카테고리 분류**:")
    lines.append(f"- 🟢 **BOTH** ({len(both)}건): 양쪽 모두 hit — 최고 신뢰도")
    lines.append(f"- 🔵 **SEMANTIC-only** ({len(sem_only)}건): 사대 의례·사건 (사대 단어 직접 언급 없음 — 영접도감 草記 등)")
    lines.append(f"- 🟣 **KEYWORD-only** ({len(kw_only)}건): 사대 담론·관용구 (이벤트형 아닌 추상 언설)")
    lines.append("")
    lines.append(f"**총 {len(all_records)}건 / {len(by_cat_year)}개 연도**")
    lines.append("")

    # 연도별 분포 표
    lines.append("## 연도별 분포")
    lines.append("")
    lines.append("| 연도 | BOTH | SEM | KW | 계 |")
    lines.append("|---|---:|---:|---:|---:|")
    grand = {"BOTH": 0, "SEM": 0, "KW": 0}
    for year in sorted(by_cat_year.keys()):
        d = by_cat_year[year]
        b, s, k = d.get("BOTH", 0), d.get("SEM", 0), d.get("KW", 0)
        grand["BOTH"] += b; grand["SEM"] += s; grand["KW"] += k
        lines.append(f"| {year} | {b} | {s} | {k} | {b+s+k} |")
    lines.append(f"| **계** | **{grand['BOTH']}** | **{grand['SEM']}** | **{grand['KW']}** "
                  f"| **{sum(grand.values())}** |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 카테고리별 섹션
    def render_section(title: str, records: list[dict], note: str = "") -> list[str]:
        out = [f"## {title} — {len(records)}건"]
        if note:
            out.append("")
            out.append(note)
        out.append("")
        by_year: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            by_year[r["year_ce"] or "?"].append(r)
        for year in sorted(by_year.keys()):
            arts = sorted(by_year[year],
                           key=lambda x: (x["date_western"] or "", x["article_id"]))
            out.append(f"### {year}년 ({len(arts)}건)")
            out.append("")
            for a in arts:
                title_a = a["article_title"] or "(제목 없음)"
                day = a["day_title"] or a["date_western"] or "?"
                ganji = f" {a['ganji']}" if a["ganji"] else ""
                meta_parts = []
                if a["sem_score"] is not None:
                    meta_parts.append(f"sem={a['sem_score']:.3f}")
                if a["kw_score"] is not None:
                    meta_parts.append(f"kw={a['kw_score']:.3f}")
                if a.get("matched_keywords"):
                    meta_parts.append("kw:" + "/".join(a["matched_keywords"]))
                meta = "  ".join(f"`{m}`" for m in meta_parts)
                out.append(f"- **{day}{ganji}** {title_a}  {meta} `[{a['article_id']}]`")
                # 컨텍스트: 키워드 hit 있으면 키워드 주변, 아니면 text 처음 200자
                if a.get("kw_context"):
                    excerpt = a["kw_context"]
                else:
                    txt = (a["text"] or "").replace("\n", " ").strip()
                    excerpt = txt[:200] + ("…" if len(txt) > 200 else "")
                out.append(f"  > {excerpt}")
            out.append("")
        return out

    lines.extend(render_section(
        "🟢 BOTH — 의미검색·키워드 양쪽 모두 hit",
        [all_records[a] for a in both],
        "→ 사대 단어를 직접 언급하면서 동시에 사대 의례·실무 사건을 다룬 기사. "
        "정조의 對淸 외교 핵심 사료."
    ))
    lines.append("---")
    lines.append("")
    lines.extend(render_section(
        "🔵 SEMANTIC-only — 사대 의례·실무 (사대 단어 미언급)",
        [all_records[a] for a in sem_only],
        "→ 사대 단어는 안 나오지만 칙사 영접·迎接都監 草記 등 의례 운영 기록. "
        "사대의 일상적 실행."
    ))
    lines.append("---")
    lines.append("")
    lines.extend(render_section(
        "🟣 KEYWORD-only — 사대 담론·관용구",
        [all_records[a] for a in kw_only],
        "→ 칙사 영접 같은 이벤트는 아니지만 `事大之誠`·`事大交隣`·`事大之禮` 같은 "
        "추상 담론·관용구가 나오는 기사. 정조의 對淸 외교철학·문서·인사 맥락."
    ))

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[done] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
