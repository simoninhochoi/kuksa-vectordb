"""매천야록 화이론 기사 v2: 한자 정확 매칭 + 사상사적 키워드.

text_hangul_aux는 hanja 자동 음차라 false positive 많음 → 한자 본문에서만 직접 매칭.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 화이론 사상/담론의 강한 표지 (등장 자체가 화이 인식 시사)
TIER1 = [
    "華夷", "夷狄", "中華", "華夏", "小中華",
    "攘夷", "斥邪", "斥洋", "衛正", "正學",
    "倭夷", "洋夷", "胡夷", "蠻夷", "夷虜",
    "邪學", "邪敎",
]
# 사건/제도로 화이론이 표면화된 표지
TIER2 = [
    "斥洋碑", "綸音", "丙寅洋擾", "辛未洋擾", "西敎",
    "華制", "華服", "胡服", "胡虜",
]

ALL = TIER1 + TIER2


def main():
    hits = []
    with (ROOT / "data" / "chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["volume_id"] != "sa_001":
                continue
            text = c["text"]
            t1 = sorted({k for k in TIER1 if k in text})
            t2 = sorted({k for k in TIER2 if k in text})
            if t1 or t2:
                hits.append({
                    "chunk_id": c["chunk_id"],
                    "level1": c["level1_title"],
                    "level2": c["level2_title"],
                    "level3": c.get("level3_title"),
                    "tier1": t1,
                    "tier2": t2,
                    "text": text,
                })

    # tier1 우선 정렬, 그 안에서 chunk_id순
    hits.sort(key=lambda h: (-len(h["tier1"]), h["chunk_id"]))

    print(f"sa_001 매천야록 — 한자 직접 매칭 {len(hits)} 청크\n")

    from collections import Counter
    cnt = Counter()
    for h in hits:
        for k in h["tier1"] + h["tier2"]:
            cnt[k] += 1
    print("[키워드 빈도]")
    for k, n in cnt.most_common():
        print(f"  {k}: {n}")

    print("\n" + "=" * 100)
    tier1_only = [h for h in hits if h["tier1"]]
    tier2_only = [h for h in hits if not h["tier1"] and h["tier2"]]

    print(f"\n[A] Tier1 (사상 직접): {len(tier1_only)}건\n")
    for i, h in enumerate(tier1_only, 1):
        path = f"{h['level2']}" + (f" > {h['level3']}" if h['level3'] else "")
        print(f"--- {i}. [{h['chunk_id']}] {path}")
        print(f"    매칭: {h['tier1'] + h['tier2']}")
        print(f"    본문: {h['text']}")
        print()

    print(f"\n[B] Tier2만 (사건/제도): {len(tier2_only)}건\n")
    for i, h in enumerate(tier2_only, 1):
        path = f"{h['level2']}" + (f" > {h['level3']}" if h['level3'] else "")
        print(f"--- {i}. [{h['chunk_id']}] {path}")
        print(f"    매칭: {h['tier2']}")
        print(f"    본문: {h['text']}")
        print()


if __name__ == "__main__":
    main()
