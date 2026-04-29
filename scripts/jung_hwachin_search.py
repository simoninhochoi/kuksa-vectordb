"""중국정사외국전 - '화친(和親)' 관련 기사 의미 검색 + 마크다운 저장.

3-pass:
  1. 글로벌 시드 의미 검색 (한·당·송·요·금·원·명대 화친 어휘 종합)
  2. 책별 필터드 의미 검색 (송사·요사·금사·원사·명사 누락 보강)
  3. 강한 시그널 키워드 보강 — 歲幣·澶淵·誓書·慶曆和議·紹興和議·和親·화친·세폐·전연
     (의미 검색이 놓친 기사를 corpus에서 직접 keyword로 surface, 출처 표기)

article_id 단위로 dedup. 같은 기사는 더 높은 reranker score / 더 구체적인
시그널을 우선 채택. 산출물: data/중국정사외국전/화친_articles.md
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

# src 모듈 경로 등록
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jung_retrieval import search  # noqa: E402

OUT = ROOT / "data" / "중국정사외국전" / "화친_articles.md"

# 시드 질의 — 시기·어휘 다양화. 한대 흉노 패러다임에 편향되지 않도록
# 당·송·요·금·원·명대 화친 어휘를 명시적으로 보강.
QUERIES_GLOBAL = [
    # 일반 / 한대
    "중국과 이민족이 화친을 맺다",
    "흉노와 한나라의 화친 조약",
    "공주를 시집보내어 화친을 맺다 화번공주",
    "이민족 군장에게 종실 여인을 처로 주는 통혼 외교",
    "和親 結盟 嫁公主",
    "약조를 어겨 화친이 깨지다 흉노 침입",
    "세폐 비단과 양식을 보내 화친을 유지하다",
    "강화하여 형제의 의를 맺다 군신 관계",
    # 당대
    "토번과 당나라의 혼인 동맹 문성공주",
    "회홀 위구르와의 통혼 영국공주",
    "남흉노 호한야 선우 한나라의 신하가 되다",
    "당과 돌궐 가한이 화친을 맺다",
    # 송대 — 전연의 맹·경력화의·세폐·맹약 어휘
    "송과 거란 요나라의 전연지맹 세폐 은과 비단",
    "송이 거란과 화친 형제지국 백질지국 군신지국 명분",
    "송과 서하 이원호 경력화의 책봉 사성 조씨",
    "송과 서하 봉책 봉작 세사 견백",
    "송과 토번 청당 화의 사신 왕래",
    "誓書 盟約 歲幣 銀絹 通好",
    "송과 교지 안남 봉책 책봉 화호",
    "당항 이계천 송과의 강화 화친",
    # 금·요·원·명대
    "금과 송의 강화 紹興和議 칭신 군신 관계",
    "원과 외국의 통혼 화친 부마 사위국",
    "고려와 원의 부마국 공주 통혼",
    "명과 와랄 타타르 몽골의 통공 화친",
    "명과 토번 안남 책봉 조공 통호",
]

# 책별 보강 시드 — 해당 책에만 한정해 broader 질의로 누락 캐치
# (한·당대는 위 GLOBAL 시드만으로 충분히 잡혔으므로 송·요·금·원·명만 보강)
QUERIES_BY_BOOK: dict[str, list[str]] = {
    "jo_0020": [  # 송사
        "송이 외국과 강화하고 사신을 보내며 세폐를 보내 화친 관계를 유지하다",
        "송이 외국 군장을 책봉하여 봉작을 내리고 사성하다",
        "송과 서하 사이의 맹약 칭신 통화 사절 왕래",
        "송과 토번 사이의 사신 왕래 약조 화호",
        "송과 거란 사이의 강화 형제지국 명분 세폐",
    ],
    "jo_0017": [  # 요사
        "요와 송이 형제지국이 되어 화친 강화 명분 세폐",
        "요와 외국의 통혼 부마 공주 약조",
    ],
    "jo_0018": [  # 금사
        "금과 송이 강화 칭신 세폐 紹興和議",
        "금과 서하 고려 화친 사신 왕래",
    ],
    "jo_0019": [  # 원사
        "원이 외국과 통혼 부마 사위국 공주 화친",
        "원과 고려 일본 안남 사신 왕래 화호",
    ],
    "jo_0021": [  # 명사
        "명이 와랄 타타르 몽골과 통공 화의",
        "명과 토번 안남 외국에 책봉 봉작 사여",
        "명과 서역 회회 외국의 통공 사신",
    ],
}

# reranker 점수 임계값 — 이 이하는 "화친"과 직접 관련성 낮다고 보고 제외
SCORE_THRESHOLD = 0.40
TOP_K_PER_QUERY = 80
TOP_K_PER_BOOK_QUERY = 60

# Pass 3 — 의미 검색이 놓친 기사를 끌어오는 강한 시그널 키워드.
# 추상적인 "통혼", "강화" 같은 단어는 거짓양성이 많아 제외하고
# 화친·세폐·맹약 사건의 결정적 시그널만.
KEYWORD_SIGNALS = [
    # 한자 (jo.d)
    "和親",
    "歲幣",
    "澶淵",        # 澶淵之盟 (1005)
    "誓書",
    "慶曆和議",
    "紹興和議",
    "尙公主",      # 공주를 시집보내다
    # 한글 (jo.k 국역)
    "화친",
    "세폐",
    "전연지맹",
    "전연의 맹",
    "경력화의",
    "소흥화의",
]
CHUNKS = ROOT / "data" / "중국정사외국전" / "chunks.jsonl"


def _absorb(aggregated: dict[str, dict], hits, query: str) -> None:
    for h in hits:
        if h.score < SCORE_THRESHOLD:
            continue
        aid = h.article_id
        existing = aggregated.get(aid)
        if existing is None or h.score > existing.get("score", 0.0):
            aggregated[aid] = {
                "source": "semantic",
                "score": h.score,
                "chunk_id": h.chunk_id,
                "article_id": h.article_id,
                "book_id": h.book_id,
                "book_name": h.book_name,
                "chapter_id": h.chapter_id,
                "chapter_title": h.chapter_title,
                "article_title": h.article_title,
                "subject_country": h.subject_country,
                "text_korean": h.text_korean,
                "matched_query": query,
            }


def _absorb_keyword(aggregated: dict[str, dict], chunk: dict, signals: list[str]) -> None:
    """semantic pass에서 놓친 기사만 keyword 시그널로 보강."""
    aid = chunk["article_id"]
    if aid in aggregated:
        # 이미 semantic 으로 잡혔으면 keyword signal만 부가 정보로 기록
        existing = aggregated[aid]
        existing.setdefault("keyword_signals", set()).update(signals)
        return
    aggregated[aid] = {
        "source": "keyword",
        "score": None,
        "keyword_signals": set(signals),
        "chunk_id": chunk["chunk_id"],
        "article_id": aid,
        "book_id": chunk["book_id"],
        "book_name": chunk["book_name"],
        "chapter_id": chunk["chapter_id"],
        "chapter_title": chunk["chapter_title"],
        "article_title": chunk["article_title"],
        "subject_country": chunk.get("subject_country", ""),
        "text_korean": chunk.get("text_korean", ""),
        "matched_query": "keyword:" + ",".join(sorted(signals)),
    }


def main() -> int:
    aggregated: dict[str, dict] = {}

    # Pass 1: 책 무관 글로벌 시드
    print(f"[pass1] {len(QUERIES_GLOBAL)} global queries (top_k={TOP_K_PER_QUERY})", file=sys.stderr)
    for i, q in enumerate(QUERIES_GLOBAL, 1):
        print(f"  [{i}/{len(QUERIES_GLOBAL)}] {q}", file=sys.stderr)
        hits = search(q, top_k=TOP_K_PER_QUERY)
        _absorb(aggregated, hits, q)
    print(f"[pass1] {len(aggregated)} unique articles", file=sys.stderr)

    # Pass 2: 책별 필터드 시드 (송사·요사·금사·원사·명사 누락 보강)
    n_book_queries = sum(len(qs) for qs in QUERIES_BY_BOOK.values())
    print(f"[pass2] {n_book_queries} book-filtered queries (top_k={TOP_K_PER_BOOK_QUERY})", file=sys.stderr)
    pre = len(aggregated)
    for book_id, qs in QUERIES_BY_BOOK.items():
        for q in qs:
            print(f"  [{book_id}] {q}", file=sys.stderr)
            hits = search(q, top_k=TOP_K_PER_BOOK_QUERY, filters={"book_id": book_id})
            _absorb(aggregated, hits, f"[{book_id}] {q}")
    print(f"[pass2] {len(aggregated)} unique articles (+{len(aggregated)-pre} from book passes)", file=sys.stderr)

    # Pass 3: 강한 시그널 키워드 — 의미 검색이 놓친 기사 보강
    print(f"[pass3] keyword recall scan ({len(KEYWORD_SIGNALS)} signals)", file=sys.stderr)
    pre = len(aggregated)
    article_signals: dict[str, set[str]] = defaultdict(set)
    article_chunks: dict[str, dict] = {}
    with CHUNKS.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            haystack = c.get("text_korean", "") + "\n" + c.get("text_hanmun", "")
            matched = [s for s in KEYWORD_SIGNALS if s in haystack]
            if not matched:
                continue
            aid = c["article_id"]
            article_signals[aid].update(matched)
            # article_id 당 첫 청크만 저장 (chunk_id 사전순)
            if aid not in article_chunks or c["chunk_id"] < article_chunks[aid]["chunk_id"]:
                article_chunks[aid] = c
    for aid, signals in article_signals.items():
        _absorb_keyword(aggregated, article_chunks[aid], sorted(signals))
    print(f"[pass3] {len(aggregated)} unique articles (+{len(aggregated)-pre} from keyword scan)",
          file=sys.stderr)

    # 책 → 장 → article_id 정렬
    by_book: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for rec in aggregated.values():
        by_book[rec["book_name"]][rec["chapter_title"]].append(rec)

    # 책 정렬 키 = book_id (jo_0001 = 사기 → ... 순)
    book_order = sorted(by_book.keys(),
                         key=lambda b: by_book[b][next(iter(by_book[b]))][0]["book_id"])

    lines: list[str] = []
    lines.append("# 중국정사외국전 — 화친(和親) 관련 기사 목록")
    lines.append("")
    lines.append(f"BGE-M3 dense+sparse 하이브리드 + BGE-reranker-v2-m3 재순위 검색.  ")
    lines.append(
        f"Pass1: 글로벌 시드 {len(QUERIES_GLOBAL)}개 × top_k {TOP_K_PER_QUERY}, "
        f"Pass2: 책별 필터드 시드 {sum(len(v) for v in QUERIES_BY_BOOK.values())}개 × top_k {TOP_K_PER_BOOK_QUERY} "
        f"(송사·요사·금사·원사·명사 보강).  "
    )
    lines.append(f"reranker score ≥ {SCORE_THRESHOLD} 필터.  ")
    lines.append(
        f"Pass3: 강한 시그널 키워드 {len(KEYWORD_SIGNALS)}개 (和親·歲幣·澶淵·誓書·慶曆和議·紹興和議·尙公主·"
        f"화친·세폐·전연지맹·경력화의·소흥화의)로 의미 검색이 놓친 기사 보강.  "
    )
    n_sem = sum(1 for r in aggregated.values() if r["source"] == "semantic")
    n_kw = len(aggregated) - n_sem
    lines.append(
        f"**총 {len(aggregated)}개 기사** ({len(by_book)}개 책) — "
        f"의미 검색 {n_sem}건, 키워드 보강 {n_kw}건."
    )
    lines.append("")
    lines.append("정렬: 책(book_id 오름차순) → 장(chapter_id) → 기사(article_id). ")
    lines.append(
        "각 항목 표기:  "
        "**`semantic`** = 의미 검색 (reranker score 0~1, 1에 가까울수록 관련 높음).  "
        "**`keyword`** = 의미 검색이 누락한 기사를 강한 시그널 키워드로 보강 (감지된 시그널 명시)."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    for book in book_order:
        chapters = by_book[book]
        n_articles = sum(len(v) for v in chapters.values())
        n_book_sem = sum(1 for ch in chapters.values() for a in ch if a["source"] == "semantic")
        n_book_kw = n_articles - n_book_sem
        lines.append(f"## {book} — {n_articles}건 (의미 {n_book_sem} · 키워드 {n_book_kw})")
        lines.append("")
        chapter_order = sorted(chapters.keys(), key=lambda ch: chapters[ch][0]["chapter_id"])
        for ch in chapter_order:
            arts = sorted(chapters[ch], key=lambda x: x["article_id"])
            lines.append(f"### {ch}")
            lines.append("")
            for a in arts:
                sc = a["subject_country"] or "—"
                title = a["article_title"] or "(제목 없음)"
                # 짧은 발췌 (text_korean 첫 줄/160자)
                kr = (a["text_korean"] or "").replace("\n", " ").strip()
                excerpt = kr[:160].rstrip()
                if len(kr) > 160:
                    excerpt += "…"
                if a["source"] == "semantic":
                    extra_sigs = a.get("keyword_signals")
                    sig_note = ""
                    if extra_sigs:
                        sig_note = f" · sigs: {', '.join(sorted(extra_sigs))}"
                    tag = f"`semantic score={a['score']:.3f}`{sig_note}"
                else:
                    sigs = ", ".join(sorted(a.get("keyword_signals", [])))
                    tag = f"`keyword: {sigs}`"
                lines.append(f"- **[{a['article_id']}]** *({sc})* {title}  {tag}")
                if excerpt:
                    lines.append(f"  > {excerpt}")
            lines.append("")
        lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
