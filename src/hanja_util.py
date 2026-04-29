"""한자↔한글 음차 유틸. hanja 라이브러리 + 두음법칙 변형."""
from __future__ import annotations

import hanja
import regex as re

_HANJA_RE = re.compile(r"\p{Han}")
_HANGUL_RE = re.compile(r"\p{Hangul}")

# 두음법칙: 첫 음절이 ㄹ/ㄴ로 시작하는 한자음의 두 가지 표기.
# 한자 1자 → (본음, 두음변형) — 인명·지명에서 흔히 등장.
DUUM_MAP: dict[str, str] = {
    # ㄹ → ㅇ
    "라": "나", "락": "낙", "란": "난", "랄": "날", "람": "남", "랍": "납",
    "랑": "낭", "래": "내", "랭": "냉", "량": "양", "려": "여", "력": "역",
    "련": "연", "렬": "열", "렴": "염", "렵": "엽", "령": "영", "례": "예",
    "로": "노", "록": "녹", "론": "논", "롱": "농", "뢰": "뇌", "료": "요",
    "룡": "용", "루": "누", "류": "유", "륙": "육", "륜": "윤", "률": "율",
    "륭": "융", "륵": "늑", "름": "늠", "릉": "능", "리": "이", "린": "인",
    "림": "임", "립": "입",
    # ㄴ → ㅇ
    "냐": "야", "녀": "여", "뇨": "요", "뉴": "유", "니": "이",
}
# 역방향(이→리, 유→류 등) — Sino-Korean 인명의 ㄹ-원음 복원만 허용.
# 'ㄴ→ㅇ' 항목(니→이 등)은 역방향에서 제외(한자음 두음법칙 적용 X).
DUUM_INV: dict[str, str] = {
    v: k for k, v in DUUM_MAP.items() if k[0] in "라락란랄람랍랑래랭량려력련렬렴렵령례로록론롱뢰료룡루류륙륜률륭륵름릉리린림립"
}


def is_hanja_char(c: str) -> bool:
    return bool(_HANJA_RE.match(c))


def is_hangul_char(c: str) -> bool:
    return bool(_HANGUL_RE.match(c))


def has_hanja(s: str) -> bool:
    return bool(_HANJA_RE.search(s))


def has_hangul(s: str) -> bool:
    return bool(_HANGUL_RE.search(s))


def script_of(s: str) -> str:
    """'hanja' | 'hangul' | 'mixed' | 'other'."""
    h = has_hanja(s)
    k = has_hangul(s)
    if h and k:
        return "mixed"
    if h:
        return "hanja"
    if k:
        return "hangul"
    return "other"


def to_hangul(s: str) -> str:
    """한자 부분만 한글로 음차. 한글·기타 문자는 그대로."""
    return hanja.translate(s, "substitution")


def duum_variants(hangul: str) -> set[str]:
    """첫 글자에 두음법칙 변형 추가. 빈 문자열·다른 자모면 원본만."""
    if not hangul:
        return {hangul}
    head = hangul[0]
    out = {hangul}
    if head in DUUM_MAP:
        out.add(DUUM_MAP[head] + hangul[1:])
    if head in DUUM_INV:
        out.add(DUUM_INV[head] + hangul[1:])
    return out


def to_hangul_variants(s: str) -> set[str]:
    """한자 → 한글 + 두음법칙 변형 모두."""
    base = to_hangul(s)
    return duum_variants(base)


if __name__ == "__main__":
    # smoke test
    samples = ["李沂", "柳馨遠", "李", "度支部", "海鶴遺書", "이기", "李沂(이기)"]
    for s in samples:
        print(f"{s!r}: script={script_of(s)} hangul={to_hangul(s)!r} variants={to_hangul_variants(s)!r}")
