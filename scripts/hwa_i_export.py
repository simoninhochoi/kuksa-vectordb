"""화이론 21개 청크의 chunk_id별 원문을 dict로 추출."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# v2 결과의 chunk_id 순서 그대로 (Tier1 17 + Tier2 4)
ORDER = [
    # Tier1
    ("A", 1, "sa_001_0010_0020_0020_p000", "병인양요", ["斥邪", "邪學", "丙寅洋擾", "綸音"]),
    ("A", 2, "sa_001_0010_0020_0310_p000", "斥洋碑", ["斥洋", "洋夷", "斥洋碑"]),
    ("A", 3, "sa_001_0010_0010_0230_p000", "都某知의어원", ["邪學"]),
    ("A", 4, "sa_001_0010_0010_0240_p000", "이경하의살인만행", ["邪學"]),
    ("A", 5, "sa_001_0010_0020_0230_p000", "이항노와기정진", ["斥邪"]),
    ("A", 6, "sa_001_0010_0030_0050_p000", "朴珪壽의시세영합", ["斥洋"]),
    ("A", 7, "sa_001_0010_0080_0070_p000", "영남유생들의상소", ["邪敎"]),
    ("A", 8, "sa_001_0010_0120_0220_p000", "만국공법과일본의배상요구", ["夷狄"]),
    ("A", 9, "sa_001_0010_0200_0010_p000_s0", "권봉희의상소(1/3)", ["衛正"]),
    ("A", 10, "sa_001_0010_0200_0010_p000_s1", "권봉희의상소(2/3)", ["衛正"]),
    ("A", 11, "sa_001_0010_0200_0010_p000_s2", "권봉희의상소(3/3)", ["衛正"]),
    ("A", 12, "sa_001_0010_0200_0030_p000", "어윤중의동학선유", ["正學", "綸音"]),
    ("A", 13, "sa_001_0030_0010_0090_p000", "成均館에博士制를설치", ["邪敎"]),
    ("A", 14, "sa_001_0030_0040_0370_p000", "의화단사건(1/2)", ["斥洋"]),
    ("A", 15, "sa_001_0030_0040_0370_p002", "의화단사건(2/2)", ["斥洋"]),
    ("A", 16, "sa_001_0050_0020_0140_p000", "諸道의의병봉기", ["邪敎"]),
    ("A", 17, "sa_001_0050_0070_0310_p000", "崔益鉉의遺疏", ["華夏"]),
    # Tier2
    ("B", 1, "sa_001_0010_0020_0160_p000", "신미양요와魚在淵의순절", ["辛未洋擾"]),
    ("B", 2, "sa_001_0020_0020_0040_p000", "홍계훈의전주동학군구축", ["綸音"]),
    ("B", 3, "sa_001_0020_0120_0030_p000", "동남선유사선유문", ["綸音"]),
    ("B", 4, "sa_001_0030_0090_0620_p000", "프랑스공사와제주도천주교", ["西敎"]),
]


def main():
    # chunk_id → text 맵
    cmap: dict[str, dict] = {}
    target_ids = {x[2] for x in ORDER}
    with (ROOT / "data" / "chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["chunk_id"] in target_ids:
                cmap[c["chunk_id"]] = c

    out = {"order": ORDER, "chunks": cmap}
    (ROOT / "data" / "hwa_i_articles.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"extracted {len(cmap)} chunks → data/hwa_i_articles.json")
    missing = target_ids - set(cmap.keys())
    if missing:
        print(f"WARN missing: {missing}")


if __name__ == "__main__":
    main()
