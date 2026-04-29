"""매천야록(sa_001) 청크에서 화이론 관련 키워드 직접 매칭.

벡터 검색은 약한 신호도 잡지만 정확도가 떨어지므로,
이미 확보한 chunks.jsonl에서 화이 관련 한자/한글 토큰을 직접 grep.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 강한 화이론 키워드 (등장 자체가 화이론적 인식 시사)
STRONG = [
    "華夷", "夷狄", "華夏", "中華", "小中華", "華人",
    "攘夷", "斥邪", "斥洋", "衛正", "正學",
    "倭夷", "洋夷", "胡夷", "蠻夷", "夷虜", "夷狄之",
    "夷狄無", "夷之", "胡虜",
]
# 단독으론 약하지만 화이 맥락에서 자주 쓰이는 단어
WEAK = ["夷", "狄", "蠻", "戎", "胡", "虜"]

# 한글 키워드
HANGUL = ["오랑캐", "중화", "화이", "양이", "왜이", "척사"]

ALL = STRONG + HANGUL


def find_hits():
    hits = []
    with (ROOT / "data" / "chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["volume_id"] != "sa_001":
                continue
            text = c["text"]
            text_hg = c.get("text_hangul_aux", "")
            matched_strong = sorted({k for k in STRONG if k in text})
            matched_hangul = sorted({k for k in HANGUL if k in text or k in text_hg})
            if matched_strong or matched_hangul:
                hits.append({
                    "chunk_id": c["chunk_id"],
                    "level1": c["level1_title"],
                    "level2": c["level2_title"],
                    "level3": c.get("level3_title"),
                    "matched": matched_strong + matched_hangul,
                    "text": text,
                })
    return hits


def main():
    hits = find_hits()
    print(f"총 {len(hits)} 청크 (sa_001 매천야록, 강한 화이 키워드 + 한글)\n")
    print("=" * 100)

    # 매칭 키워드 빈도
    from collections import Counter
    cnt = Counter()
    for h in hits:
        for k in h["matched"]:
            cnt[k] += 1
    print("\n[키워드 빈도]")
    for k, n in cnt.most_common():
        print(f"  {k}: {n}")

    print("\n" + "=" * 100)
    print("\n[전체 청크 — 권/장/매칭/본문]\n")
    for i, h in enumerate(hits, 1):
        path = f"{h['level1']} > {h['level2']}"
        if h["level3"]:
            path += f" > {h['level3']}"
        print(f"--- {i}. [{h['chunk_id']}] {path}")
        print(f"    매칭: {h['matched']}")
        print(f"    본문: {h['text']}")
        print()


if __name__ == "__main__":
    main()
