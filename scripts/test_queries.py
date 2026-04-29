"""회귀 테스트 쿼리 일괄 실행 (모델 한 번 로드)."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrieval import search, expand_query

QUERIES = [
    ("이기", "sa_003 (해학유서) 또는 한자 동명이인"),
    ("海鶴", "sa_003 (해학유서) - 책 제목/저자 호"),
    ("李沂의 田制 사상", "sa_003 - 토지개혁 단락"),
    ("전봉준 동학", "sa_001 (매천야록) - 동학란"),
    ("황현 매천야록", "sa_001 (매천야록) - 직접 매치"),
]

for q, expect in QUERIES:
    print(f"\n{'='*80}\n[query] {q}\n[expect] {expect}\n[expanded] {expand_query(q)}\n")
    hits = search(q, top_k=5)
    sa_ids = sorted({h.volume_id for h in hits})
    print(f"[volumes hit] {sa_ids}")
    for i, h in enumerate(hits, 1):
        print(f"  {i}. score={h.score:.4f} [{h.volume_id}] {h.volume_title_hanja}({h.volume_title_hangul}) > {h.level2_title}")
        print(f"     {h.text[:150]}")
