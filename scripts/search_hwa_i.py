"""매천야록(sa_001)에서 화이론 관련 기사 회수.

여러 키워드로 sa_001 필터 검색 → dedup → 출력.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrieval import search

QUERIES = [
    "華夷之辨 중화와 오랑캐 구분",
    "夷狄 오랑캐",
    "中華 中國 동방예의",
    "攘夷 척화 서양 오랑캐 배척",
    "倭夷 왜놈 일본 오랑캐",
    "洋夷 서양 양놈 양이",
    "衣冠文物 의관 문물 중화",
    "斥邪 위정척사 정학",
    "小中華 조선 中華 정통",
    "胡虜 만주 청나라 오랑캐",
]

TOP_K = 30
FILTER = {"volume_id": "sa_001"}

seen: dict[str, dict] = {}
for q in QUERIES:
    print(f"[query] {q}", file=sys.stderr)
    hits = search(q, top_k=TOP_K, filters=FILTER)
    for h in hits:
        if h.chunk_id in seen:
            # 더 높은 점수면 갱신
            if h.score > seen[h.chunk_id]["score"]:
                seen[h.chunk_id] = {
                    "score": h.score, "level2": h.level2_title,
                    "level3": h.level3_title, "text": h.text,
                    "queries": seen[h.chunk_id]["queries"] + [q],
                }
        else:
            seen[h.chunk_id] = {
                "score": h.score, "level2": h.level2_title,
                "level3": h.level3_title, "text": h.text,
                "queries": [q],
            }

# 점수 정렬
ranked = sorted(seen.items(), key=lambda x: x[1]["score"], reverse=True)

# 화이 키워드가 본문에 직접 등장하는 청크 추출 (좀 더 엄밀히)
KEYWORDS = ["華夷", "夷狄", "中華", "夷虜", "胡夷", "倭夷", "洋夷", "蠻夷", "攘夷", "斥邪", "小中華", "夷狄之", "華夏"]

print(f"\n총 {len(ranked)} 청크 회수 (sa_001 매천야록)\n")
print("="*100)

direct_hits = []
indirect_hits = []
for cid, d in ranked:
    if any(k in d["text"] for k in KEYWORDS):
        direct_hits.append((cid, d))
    else:
        indirect_hits.append((cid, d))

print(f"\n[A] 본문에 화이 키워드 직접 등장: {len(direct_hits)}건\n")
for cid, d in direct_hits:
    print(f"--- [{cid}] score={d['score']:.4f} ---")
    print(f"  {d['level2']}" + (f" > {d['level3']}" if d['level3'] else ""))
    print(f"  매칭 키워드: {[k for k in KEYWORDS if k in d['text']]}")
    print(f"  쿼리: {d['queries']}")
    print(f"  본문: {d['text'][:400]}")
    print()

print(f"\n[B] 의미적 회수만 (직접 키워드 없음): {len(indirect_hits)}건 — 상위 5개만")
for cid, d in indirect_hits[:5]:
    print(f"--- [{cid}] score={d['score']:.4f} ---")
    print(f"  {d['level2']}" + (f" > {d['level3']}" if d['level3'] else ""))
    print(f"  쿼리: {d['queries']}")
    print(f"  본문: {d['text'][:300]}")
    print()
