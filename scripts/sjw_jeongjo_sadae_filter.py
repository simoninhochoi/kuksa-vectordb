"""정조 시대 사대 키워드 178건 → reranker로 진성 사대 mention 필터링.

각 hit 의 키워드 ±60자 컨텍스트를 reranker query 와 pair 매칭 → score 분류.
'사대(事大)' 가 동사적/명사적 외교 의례로 쓰인 경우는 high score, 단어 경계
어긋남(處事大為, 役事大體 등 false positive) 은 low score.
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
OUT = ROOT / "data" / "승정원일기" / "정조_사대_filtered_articles.md"

KEYWORDS = ["事大", "사대"]
CONTEXT_RADIUS = 80  # 키워드 좌우 80자

# 사대 의미 평가 query — 한문 코퍼스에 맞춰 한문 위주로 작성 (cross-lingual 약점 회피)
RERANK_QUERY = (
    "事大之禮 事大之誠 事大交隣 承文院 事大衙門 "
    "對淸 朝貢 表箋 咨文 勅使 迎勅 祭天 尊周大義 "
    "조선이 中國(淸)에 사대 외교를 행하는 의례와 정신"
)
# 점수 분포 분석 결과 한문 cross-lingual 페어는 score 낮음 → 임계값 동적 조정
SCORE_THRESHOLD_HIGH = 0.10   # 진성 사대 mention (전체 중 상위)
SCORE_THRESHOLD_MID = 0.03    # 모호


def extract_context(text: str, keyword: str, radius: int) -> str:
    idx = text.find(keyword)
    if idx < 0:
        return text[:200]
    start = max(0, idx - radius)
    end = min(len(text), idx + len(keyword) + radius)
    return ("…" if start > 0 else "") + text[start:end] + ("…" if end < len(text) else "")


def main() -> int:
    # 1) 정조 chunks 에서 키워드 hit 모음 (article 단위 dedup)
    print("[step 1] 정조 chunks 스캔 + 키워드 hit 수집", file=sys.stderr)
    candidates: list[dict] = []
    seen_aids: set[str] = set()
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
            if aid in seen_aids:
                continue
            seen_aids.add(aid)
            ctx = extract_context(text, matched[0], CONTEXT_RADIUS)
            candidates.append({
                "article_id": aid,
                "chunk_id": c["chunk_id"],
                "year_ce": c.get("year_ce", ""),
                "day_title": c.get("day_title", ""),
                "date_western": c.get("date_western", ""),
                "ganji": c.get("ganji", ""),
                "article_title": c.get("article_title", ""),
                "article_type": c.get("article_type", ""),
                "text": text,
                "matched": matched,
                "context": ctx,
            })
    print(f"[step 1] {len(candidates)} 후보", file=sys.stderr)

    # 2) reranker 로 사대 적합도 score
    print("[step 2] BGE-reranker-v2-m3 로 컨텍스트 → query 적합도 score", file=sys.stderr)
    sys.path.insert(0, str(ROOT / "src"))
    from FlagEmbedding import FlagReranker  # noqa: E402

    reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, device="cuda")
    pairs = [[RERANK_QUERY, c["context"]] for c in candidates]
    scores = reranker.compute_score(pairs, normalize=True, batch_size=32)
    if not isinstance(scores, list):
        scores = [scores]
    for c, sc in zip(candidates, scores):
        c["score"] = float(sc)

    # 3) 분류
    high = [c for c in candidates if c["score"] >= SCORE_THRESHOLD_HIGH]
    mid = [c for c in candidates if SCORE_THRESHOLD_MID <= c["score"] < SCORE_THRESHOLD_HIGH]
    low = [c for c in candidates if c["score"] < SCORE_THRESHOLD_MID]
    print(f"[step 3] high(≥{SCORE_THRESHOLD_HIGH})={len(high)}, "
          f"mid={len(mid)}, low(<{SCORE_THRESHOLD_MID})={len(low)}", file=sys.stderr)

    # 4) 마크다운 — high + mid (low 는 부록으로 sample 만)
    by_year_high: dict[str, list[dict]] = defaultdict(list)
    for c in high:
        by_year_high[c["year_ce"] or "?"].append(c)
    by_year_mid: dict[str, list[dict]] = defaultdict(list)
    for c in mid:
        by_year_mid[c["year_ce"] or "?"].append(c)

    lines = []
    lines.append("# 정조 시대 — '事大/사대' 키워드 + reranker 필터링 결과")
    lines.append("")
    lines.append("**Pipeline**: ① chunks.jsonl 에서 정조(king_prefix=G) 청크 중 "
                  "`'事大'` 또는 `'사대'` 포함 article 178건 surface  ")
    lines.append(f"② BGE-reranker-v2-m3 로 키워드 ±{CONTEXT_RADIUS}자 컨텍스트를 query "
                  f"〈{RERANK_QUERY[:50]}…〉 와 매칭해 적합도 score 산출.")
    lines.append("")
    lines.append(f"**분류**: 진성(score ≥ {SCORE_THRESHOLD_HIGH}) **{len(high)}건**, "
                  f"모호({SCORE_THRESHOLD_MID}≤score<{SCORE_THRESHOLD_HIGH}) **{len(mid)}건**, "
                  f"false positive(<{SCORE_THRESHOLD_MID}) **{len(low)}건**.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 진성 사대 hits
    lines.append("## 🎯 진성 사대 mention (score ≥ {:.2f}, {}건)".format(SCORE_THRESHOLD_HIGH, len(high)))
    lines.append("")
    for year in sorted(by_year_high.keys()):
        arts = sorted(by_year_high[year], key=lambda x: (-x["score"], x["chunk_id"]))
        lines.append(f"### {year}년 ({len(arts)}건)")
        lines.append("")
        for a in arts:
            title = a["article_title"] or "(제목 없음)"
            day = a["day_title"] or a["date_western"] or "?"
            ganji = f" {a['ganji']}" if a["ganji"] else ""
            sigs = "/".join(a["matched"])
            lines.append(
                f"- **{day}{ganji}** {title}  `score={a['score']:.3f}` `kw: {sigs}` "
                f"`[{a['article_id']}]`"
            )
            lines.append(f"  > {a['context']}")
        lines.append("")

    # 모호 — 사람이 검토할 영역
    if mid:
        lines.append("---")
        lines.append("")
        lines.append("## 🟡 모호 (수동 검토 권장, {}건)".format(len(mid)))
        lines.append("")
        for year in sorted(by_year_mid.keys()):
            arts = sorted(by_year_mid[year], key=lambda x: (-x["score"], x["chunk_id"]))
            lines.append(f"### {year}년 ({len(arts)}건)")
            lines.append("")
            for a in arts:
                title = a["article_title"] or "(제목 없음)"
                day = a["day_title"] or a["date_western"] or "?"
                ganji = f" {a['ganji']}" if a["ganji"] else ""
                sigs = "/".join(a["matched"])
                lines.append(
                    f"- **{day}{ganji}** {title}  `score={a['score']:.3f}` `kw: {sigs}` "
                    f"`[{a['article_id']}]`"
                )
                lines.append(f"  > {a['context']}")
            lines.append("")

    # false positive 일부 샘플 (감을 위해)
    lines.append("---")
    lines.append("")
    lines.append("## ❌ False positive 샘플 (전체 {}건 중 5건만)".format(len(low)))
    lines.append("")
    low_samples = sorted(low, key=lambda x: x["score"])[:5]
    for a in low_samples:
        lines.append(
            f"- `score={a['score']:.3f}` {a['day_title'] or a['date_western']}  "
            f"{a['article_title'] or '(제목없음)'} `[{a['article_id']}]`"
        )
        lines.append(f"  > {a['context']}")
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
