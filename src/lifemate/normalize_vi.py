"""Rule layer for Vietnamese speech transcripts.

Everything here is deterministic code: money / numbers / relative time are
resolved WITHOUT the LLM, then handed to the LLM as hints and used again at
grounding time. Numbers and timestamps are non-fuzzy arguments, so one wrong
digit costs the whole sample (state hash) -- never let a 4B model do this math.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def strip_diacritics(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


def norm_key(s: str) -> str:
    """Comparison key: NFC -> strip diacritics -> lowercase -> collapse spaces."""
    return re.sub(r"\s+", " ", strip_diacritics(nfc(s)).lower()).strip()


def norm_tokens(s: str) -> list[str]:
    return [t for t in re.split(r"[^\w]+", norm_key(s)) if t]


# --------------------------------------------------------------------------- #
# Vietnamese number words
# --------------------------------------------------------------------------- #

# Number words are matched WITH diacritics (NFC, lowercase): "sáu"(6) vs "sau"(after),
# "tư"(4) vs "từ"(from), "bốn"(4) vs "bón" would collide if diacritics were stripped.
DIGIT_WORDS = {
    "không": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5,
    "lăm": 5, "sáu": 6, "bảy": 7, "bẩy": 7, "tám": 8, "chín": 9,
}
TEN_WORDS = {"mười", "mươi", "chục"}       # "mười" (10) ; "mươi"/"chục" (x10): "năm chục nghìn" = 50000
HUNDRED_WORDS = {"trăm"}
THOUSAND_WORDS = {"nghìn", "ngàn", "k"}
MILLION_WORDS = {"triệu", "tr"}
BILLION_WORDS = {"tỷ", "tỉ"}
HALF_WORDS = {"rưỡi", "rưởi"}
LINK_WORDS = {"linh", "lẻ"}
CURRENCY_WORDS = {"đồng", "vnd", "vnđ", "đ", "d"}

# "năm" followed by these is the word "year", not the digit 5
YEAR_CONTEXT = {"nay", "sau", "tới", "ngoái", "trước", "qua", "rồi", "kia", "nhuận", "học", "nào"}

_DIGIT_RE = re.compile(r"^\d+(?:[.,]\d+)*$")


def parse_digit_token(tok: str) -> Optional[float]:
    """'1.500.000' -> 1500000 ; '1,5' -> 1.5 ; '2.25' -> 2.25 ; '12' -> 12."""
    if not _DIGIT_RE.match(tok):
        return None
    parts = re.split(r"[.,]", tok)
    if len(parts) == 1:
        return float(parts[0])
    # all groups after the first have 3 digits -> thousands separators
    if all(len(p) == 3 for p in parts[1:]):
        return float("".join(parts))
    if len(parts) == 2:
        return float(parts[0] + "." + parts[1])
    return None


@dataclass
class _Tok:
    kind: str          # NUM DIGIT TEN HUNDRED THOUSAND MILLION BILLION HALF LINK CUR
    value: float = 0.0
    raw: str = ""


def _classify(raw: str, prev_raw: Optional[str], next_raw: Optional[str]) -> Optional[_Tok]:
    k = nfc(raw).lower()
    v = parse_digit_token(raw)
    if v is not None:
        return _Tok("NUM", v, raw)
    if k == "năm" and next_raw is not None and nfc(next_raw).lower() in YEAR_CONTEXT:
        return None
    if k in DIGIT_WORDS:
        return _Tok("DIGIT", DIGIT_WORDS[k], raw)
    if k in TEN_WORDS:
        return _Tok("TEN", 10, raw)
    if k in HUNDRED_WORDS:
        return _Tok("HUNDRED", 100, raw)
    if k in THOUSAND_WORDS:
        return _Tok("THOUSAND", 1e3, raw)
    if k in MILLION_WORDS:
        return _Tok("MILLION", 1e6, raw)
    if k in BILLION_WORDS:
        return _Tok("BILLION", 1e9, raw)
    if k in HALF_WORDS:
        return _Tok("HALF", 0.5, raw)
    if k in LINK_WORDS:
        return _Tok("LINK", 0, raw)
    if k in CURRENCY_WORDS:
        return _Tok("CUR", 0, raw)
    return None


BIG_KINDS = {"THOUSAND", "MILLION", "BILLION"}


def eval_number_run(toks: list[_Tok]) -> tuple[Optional[float], bool]:
    """Evaluate a run of number tokens. Returns (value, has_unit).

    Handles spoken conventions:
      "hai mươi lăm"        -> 25
      "hai trăm ba"         -> 230     (bare digit after 'trăm' = tens)
      "hai trăm linh ba"    -> 203
      "ba triệu hai"        -> 3_200_000
      "ba triệu hai trăm"   -> 3_200_000
      "hai triệu rưỡi"      -> 2_500_000
      "1,5 triệu" / "1tr5"  -> 1_500_000
      "năm trăm k"          -> 500_000
    """
    total = 0.0
    cur = 0.0
    last_big: Optional[float] = None
    last_small: Optional[float] = None   # 100 / 10 / 1 / None
    cur_has_hundred = False
    has_unit = False
    prev: Optional[_Tok] = None

    def flush_tail():
        nonlocal total, cur, cur_has_hundred
        if cur <= 0:
            return
        if last_big is None:
            total += cur
        else:
            if cur_has_hundred or cur >= 100:
                total += cur * last_big / 1000
            elif cur < 10:
                total += cur * last_big / 10
            else:
                total += cur * last_big / 100
        cur = 0.0
        cur_has_hundred = False

    for t in toks:
        if t.kind == "NUM":
            if cur > 0 and prev is not None and prev.kind in ("NUM", "DIGIT"):
                # two numbers back to back without a unit: treat as separate -> stop
                break
            cur += t.value
            last_small = None
        elif t.kind == "DIGIT":
            if last_small == 100:
                cur += t.value * 10
            else:
                cur += t.value   # "bốn lăm": the first token was already rewritten as 40
            if last_small not in (100,):
                last_small = 1 if last_small is None else last_small
        elif t.kind == "TEN":
            if prev is not None and prev.kind == "DIGIT":
                # "hai mươi": previous digit d was added as d (or d*10 after 'trăm'); make it d*10
                d = prev.value
                if last_small == 100:
                    pass  # already counted as tens after 'trăm'
                else:
                    cur = cur - d + d * 10
            else:
                cur += 10
            last_small = 10
        elif t.kind == "HUNDRED":
            if prev is not None and prev.kind in ("DIGIT", "NUM") and cur > 0:
                cur = cur * 100 if cur < 10 else cur  # "hai trăm" ; "mười hai trăm" (rare)
            else:
                cur += 100
            last_small = 100
            cur_has_hundred = True
        elif t.kind in BIG_KINDS:
            has_unit = True
            group = cur if cur > 0 else 1.0
            if last_big is not None and t.value >= last_big:
                # "một nghìn triệu" - weird; just multiply
                total = (total + group) * t.value if total else group * t.value
            else:
                total += group * t.value
            cur = 0.0
            cur_has_hundred = False
            last_big = t.value
            last_small = None
        elif t.kind == "HALF":
            if prev is not None and prev.kind in BIG_KINDS:
                total += prev.value / 2
            elif last_small == 100:
                cur += 50
            elif last_small == 10:
                cur += 5
            else:
                cur += 0.5
        elif t.kind == "LINK":
            last_small = 1
        elif t.kind == "CUR":
            has_unit = True
        prev = t

    flush_tail()
    if total == 0 and cur == 0:
        return None, has_unit
    return total, has_unit


_WORD_RE = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+", re.UNICODE)


def _pre_substitute(text: str) -> str:
    t = nfc(text)
    # 1tr5 -> 1.5 tr ; 500k -> 500 k ; 1k5 -> 1.5 k ; 2tr -> 2 tr ; 5000đ -> 5000 đ
    t = re.sub(r"(?i)(\d+)(tr|k)(\d)\b", r"\1.\3 \2", t)
    t = re.sub(r"(?i)(\d)(tr|k|đ|d)\b", r"\1 \2", t)
    t = re.sub(r"(\d)\s*%", r"\1 phantram", t)
    return t


@dataclass
class NumberSpan:
    text: str
    value: float
    has_unit: bool
    start: int
    end: int

    @property
    def int_value(self) -> int:
        return int(round(self.value))


def extract_numbers(text: str) -> list[NumberSpan]:
    """All numeric runs in the text (spoken or digit form)."""
    t = _pre_substitute(text)
    matches = list(_WORD_RE.finditer(t))
    raws = [m.group(0) for m in matches]
    toks: list[Optional[_Tok]] = []
    for i, raw in enumerate(raws):
        prev_raw = raws[i - 1] if i > 0 else None
        next_raw = raws[i + 1] if i + 1 < len(raws) else None
        toks.append(_classify(raw, prev_raw, next_raw))

    spans: list[NumberSpan] = []
    BARE = ("NUM", "DIGIT")
    START = ("NUM", "DIGIT", "TEN", "HUNDRED")   # a run must start with a quantity
    i = 0
    while i < len(toks):
        tk = toks[i]
        if tk is None:
            i += 1
            continue
        if tk.kind in BIG_KINDS:
            # "ngàn rưỡi" / "triệu rưỡi" (1500 / 1.5M): a bare unit followed by "rưỡi" starts a run
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            if nxt is None or nxt.kind != "HALF":
                i += 1
                continue
        elif tk.kind not in START or (tk.kind == "TEN" and nfc(tk.raw).lower() not in ("mười", "chục")):
            i += 1
            continue
        if tk.kind == "HUNDRED":
            # bare "trăm" only starts a run as "trăm nghìn" / "trăm triệu" (not "phần trăm")
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            if nxt is None or nxt.kind not in BIG_KINDS:
                i += 1
                continue
        j = i + 1
        run: list[_Tok] = [tk]
        while j < len(toks) and toks[j] is not None:
            nxt = toks[j]
            # "ba năm trăm" = [ba] [năm trăm]: two bare quantities in a row start a new run,
            # except "bốn lăm" / "hai mốt" (lăm, mốt only follow a tens digit: 45, 21)
            if run[-1].kind in BARE and nxt.kind in BARE:
                if run[-1].kind == "DIGIT" and nfc(nxt.raw).lower() in ("lăm", "mốt") and len(run) == 1:
                    run[-1] = _Tok("DIGIT", run[-1].value * 10, run[-1].raw)   # tens
                    run.append(nxt)
                    j += 1
                    continue
                # compact spoken tens: "hai sáu độ" = 26, "năm chín nghìn" = 59 000 (two digits before a unit),
                # "một trăm năm tám" = 158 (two digits after "trăm")
                if run[-1].kind == "DIGIT" and nxt.kind == "DIGIT" and 2 <= run[-1].value <= 9 and 1 <= nxt.value <= 9:
                    after = raws[j + 1] if j + 1 < len(raws) else ""
                    after_hundred = len(run) >= 2 and run[-2].kind == "HUNDRED"
                    if after_hundred:   # one two-digit quantity (the evaluator would scale a bare digit after "trăm")
                        run[-1] = _Tok("NUM", run[-1].value * 10 + nxt.value, run[-1].raw)
                        j += 1
                        continue
                    if len(run) == 1 and _COMPACT_UNIT.match(after):
                        run[-1] = _Tok("DIGIT", run[-1].value * 10, run[-1].raw)   # tens
                        run.append(nxt)
                        j += 1
                        continue
                break
            # "mười" (10) can only start a run; "mươi"/"chục" (x10) must follow a digit
            if nxt.kind == "TEN" and nfc(nxt.raw).lower() == "mười":
                break
            # a big unit can only continue a run that has a quantity ("ba triệu"), except "<unit> rưỡi"
            if nxt.kind in BIG_KINDS and run[-1].kind in BIG_KINDS:
                break
            run.append(nxt)
            j += 1
        while run and run[-1].kind in ("LINK",):
            run.pop()
            j -= 1
        value, has_unit = eval_number_run(run)
        single_word = len(run) == 1 and run[0].kind == "DIGIT"
        if single_word and j < len(raws) and _MEASURE_WORDS.match(raws[j]):
            single_word = False   # "ba mét", "một cuộn", "bảy phút": a quantity, not an article
        if value is not None and not single_word:
            start = matches[i].start()
            end = matches[j - 1].end()
            spans.append(NumberSpan(t[start:end], value, has_unit, start, end))
        i = j
    return spans


_COMPACT_UNIT = re.compile(r"^(?:[dđ][oộ]|ngh[iì]n|ng[aà]n|k|tri[eệ]u|ph[aầ]n|tu[oổ]i|k[ýy]|kg|c[aâ]n)$", re.IGNORECASE)
MONEY_VERBS = {"chuyen", "nap", "thanh", "tra", "gui", "rut", "dong", "mua", "gia", "tien"}
_MEASURE_WORDS = re.compile(
    r"^(?:m[eé]t|c[aâ]y|km|cm|mm|kg|g|l[ií]t|lit|ph[uú]t|ti[eế]ng|gi[oờ]|ng[aà]y|tu[aầ]n|th[aá]ng|[dđ][oộ]|b[uư][oớ]c|cu[oộ]n|t[aấ]m|c[aá]i|chi[eế]c|"
    r"l[aầ]n|chuy[eế]n|v[eé]|ly|c[oố]c|su[aấ]t|ph[aầ]n|b[aà]i|t[oờ]|quy[eể]n|cu[oố]n|ch[oỗ]|ng[uư][oờ]i|h[oộ]p|chai|lon|t[uú]i|g[oó]i|b[oộ]|"
    r"o[aá]t|w|kw|kwh|inch|m|s[oố]|ch[iỉ]|l[uư][oợ]ng|c[aâ]n|y[eê]n|[dđ][oô]|usd|eur|[oơ]-?r[oô]|b[aả]ng)$",
    re.IGNORECASE,
)


def extract_money(text: str) -> list[NumberSpan]:
    """Numbers that look like VND amounts: have a unit (nghìn/triệu/k/đ...) or >= 1000."""
    out = []
    for s in extract_numbers(text):
        if s.has_unit or s.value >= 1000:
            out.append(s)
    return out


# --------------------------------------------------------------------------- #
# Time expressions
# --------------------------------------------------------------------------- #

WEEKDAYS = {
    "thu hai": 0, "thu 2": 0, "thu ba": 1, "thu 3": 1, "thu tu": 2, "thu 4": 2,
    "thu nam": 3, "thu 5": 3, "thu sau": 4, "thu 6": 4, "thu bay": 5, "thu 7": 5,
    "chu nhat": 6, "cn": 6,
}


@dataclass
class TimeHint:
    text: str
    iso: str
    kind: str           # "datetime" | "date" | "duration"
    note: str = ""


def _parse_iso(now: str) -> datetime:
    return datetime.fromisoformat(now)


def _v(word: str) -> str:
    """Regex alternative accepting the word with or without diacritics (ASR usually keeps them)."""
    plain = strip_diacritics(word)
    return re.escape(word) if plain == word else f"(?:{re.escape(word)}|{re.escape(plain)})"


def _alt(words) -> str:
    return "(?:" + "|".join(_v(w) for w in sorted(words, key=len, reverse=True)) + ")"


_D = _alt([w for w in DIGIT_WORDS if w != "không"] + ["lăm"])
_TENW = _alt(["mươi", "chục"])
_MUOI = _v("mười")
# spoken number 0-99: "5", "mười", "mười lăm", "hai mươi", "hai mươi lăm", "năm chục", "tám"
_LAM = _alt(["lăm", "mốt"])
NUMW = rf"(?:\d{{1,2}}|{_D}\s+{_TENW}(?:\s+{_D})?|{_MUOI}(?:\s+{_D})?|{_D}\s+{_LAM}|{_D})"
# in a date, two digit words in a row are tens+units ("hôm hai tám" = the 28th)
DATE_NUMW = rf"(?:\d{{1,2}}|{_D}\s+{_TENW}(?:\s+{_D})?|{_MUOI}(?:\s+{_D})?|{_D}(?:\s+{_D})?)"


def _num_words_value(s: str) -> Optional[int]:
    """'5' -> 5, 'mười lăm' -> 15, 'hai mươi' -> 20, 'năm chục' -> 50, 'hai tám' -> 28 (date style)."""
    s = nfc(s).lower().strip()
    if s.isdigit():
        return int(s)
    toks = [_classify(w, None, None) for w in s.split()]
    if not toks or any(t is None for t in toks):
        return None
    kinds = [t.kind for t in toks]
    if kinds == ["DIGIT"]:
        return int(toks[0].value)
    if kinds == ["TEN"] and nfc(toks[0].raw).lower() == "mười":
        return 10
    if kinds == ["TEN", "DIGIT"] and nfc(toks[0].raw).lower() == "mười":
        return 10 + int(toks[1].value)
    if kinds == ["DIGIT", "TEN"]:
        return int(toks[0].value) * 10
    if kinds == ["DIGIT", "TEN", "DIGIT"]:
        return int(toks[0].value) * 10 + int(toks[2].value)
    if kinds == ["DIGIT", "DIGIT"]:
        return int(toks[0].value) * 10 + int(toks[1].value)
    return None


_word_hour = _num_words_value   # backwards-compatible name


_PERIODS = ["sáng", "trưa", "chiều", "tối", "đêm", "khuya"]
PERIOD_RE = re.compile(rf"(?<![^\W\d_]){_alt(_PERIODS)}(?![^\W\d_])", re.IGNORECASE)

TIME_RE = re.compile(
    rf"(?<![^\W\d_])(?P<h>{NUMW})\s*(?:{_v('giờ')}|g|h)(?![^\W\d_])\s*"
    rf"(?:(?P<half>{_v('rưỡi')})|{_v('kém')}\s+(?P<kem>{NUMW})|(?P<m>\d{{1,2}}|{NUMW})(?:\s*(?:{_v('phút')}|p)(?![^\W\d_]))?)?"
    rf"|(?<!\d)(?P<h2>\d{{1,2}}):(?P<m2>\d{{2}})(?!\d)"
    rf"|(?<![^\W\d_])(?P<h3>{NUMW})\s+{_v('rưỡi')}(?=\s+{_alt(_PERIODS)})"
    rf"|(?<![^\W\d_])(?:{_v('từ')}|{_v('lúc')}|{_v('tới')}|{_v('đến')})\s+(?P<h3b>{NUMW})\s+{_v('rưỡi')}(?![^\W\d_])"
    rf"|(?<![^\W\d_])(?P<h3c>{NUMW})\s+{_v('rưỡi')}(?![^\W\d_])",
    re.IGNORECASE,
)

_TIME_CONTEXT_RE = re.compile(rf"(?<![^\W\d_])(?:{_alt(['giờ', 'báo thức', 'nhắc', 'lúc', 'hẹn', 'sáng', 'trưa', 'chiều', 'tối', 'đêm'])})(?![^\W\d_])", re.IGNORECASE)

_DAY_WORDS = {
    # text -> day offset
    "ngày mai": 1, "sáng mai": 1, "trưa mai": 1, "chiều mai": 1, "tối mai": 1, "đêm mai": 1, "mai": 1,
    "ngày mốt": 2, "ngày kia": 2,
    "hôm nay": 0, "bữa nay": 0, "sáng nay": 0, "trưa nay": 0, "chiều nay": 0, "tối nay": 0, "đêm nay": 0, "nay": 0,
    "hôm qua": -1, "hôm kia": -2,
    "tuần sau nữa": 14, "tuần sau": 7, "tuần tới": 7,
}
DAY_RE = re.compile(rf"(?<![^\W\d_]){_alt(_DAY_WORDS)}(?![^\W\d_])", re.IGNORECASE)
MONTH_RE = re.compile(rf"(?<![^\W\d_]){_alt(['tháng sau', 'tháng tới'])}(?![^\W\d_])", re.IGNORECASE)
WEEKDAY_RE = re.compile(
    rf"(?<![^\W\d_])(?P<wd>{_v('thứ')}\s+(?:hai|ba|{_v('tư')}|{_v('năm')}|{_v('sáu')}|{_v('bảy')}|[2-7])|{_v('chủ')}\s+{_v('nhật')}|cn)"
    rf"(?:\s+(?P<mod>{_v('tuần')}\s+(?:sau|{_v('tới')})|{_v('này')}|{_v('tới')}))?(?![^\W\d_])",
    re.IGNORECASE,
)
DATE_RE = re.compile(
    rf"(?<![\d/])(?:(?:{_v('ngày')}|{_v('mùng')}|{_v('hôm')})\s+)?(?P<d>\d{{1,2}})\s*(?:/|-|{_v('tháng')})\s*(?P<mo>\d{{1,2}})"
    rf"(?:\s*(?:/|-|{_v('năm')})\s*(?P<y>\d{{4}}))?(?![\d/])",
    re.IGNORECASE,
)
_MONTH_NAMES = {"giêng": 1, "chạp": 12}
WORD_DATE_RE = re.compile(
    rf"(?<![^\W\d_])(?:{_v('ngày')}|{_v('mùng')}|{_v('hôm')})\s+(?P<dw>{DATE_NUMW})"
    rf"(?:\s+{_v('tháng')}\s+(?P<mw>\d{{1,2}}|{NUMW}|{_v('giêng')}|{_v('chạp')}))?(?![^\W\d_])",
    re.IGNORECASE,
)
# "hai mốt tháng mười" (no ngày/mùng prefix): the month is then mandatory; "sáu tháng mười triệu" is money, not a date
WORD_DATE2_RE = re.compile(
    rf"(?<![^\W\d_])(?P<dw>{DATE_NUMW})\s+{_v('tháng')}\s+(?P<mw>\d{{1,2}}|{NUMW}|{_v('giêng')}|{_v('chạp')})(?![^\W\d_])"
    rf"(?!\s+(?:{_alt(['triệu', 'nghìn', 'ngàn', 'trăm', 'tỷ', 'tỉ', 'k', 'đồng', 'lăm', 'năm', 'tư'])})(?![^\W\d_]))",
    re.IGNORECASE,
)
DURATION_RE = re.compile(
    rf"(?<![^\W\d_])(?:sau|trong|{_v('khoảng')})?\s*(?P<n>\d{{1,3}}|{NUMW}|{_v('nửa')})\s*"
    rf"(?P<u>{_v('phút')}|{_v('tiếng')}|{_v('giờ')}|{_v('ngày')}|{_v('tuần')})\s*(?:{_v('nữa')}|sau|{_v('tới')})(?![^\W\d_])",
    re.IGNORECASE,
)
# plain spans ("nửa tiếng", "45 phút", "một tiếng rưỡi") -> minutes, for shifting event times
SPAN_RE = re.compile(
    rf"(?<![^\W\d_])(?P<n>{_v('nửa')}|\d{{1,3}}|{NUMW})\s*(?P<u>{_v('phút')}|{_v('tiếng')})(?:\s+(?P<half>{_v('rưỡi')}))?(?![^\W\d_])",
    re.IGNORECASE,
)
# "mai" preceded by a title / kinship word is a person's name (cô Mai, chị Mai), not "tomorrow"
_NAME_TITLES = {"anh", "chị", "em", "cô", "chú", "bác", "ông", "bà", "thầy", "cháu", "dì", "cậu", "mợ", "bạn", "sếp",
                "cho", "với", "gửi", "của", "nhắn", "gọi", "tên", "bé", "con", "thằng", "cái", "nhà", "hỏi", "báo", "kêu", "mời"}
_MEASURE_AFTER = re.compile(
    rf"^\s*(?:%|{_alt(['độ', 'phần trăm', 'phần', 'bước', 'mét', 'm', 'cm', 'km', 'kg', 'g', 'lít', 'lit', 'phút', 'giờ', 'tiếng', 'ngày', 'tuần', 'tháng', 'năm', 'tuổi', 'oát', 'w', 'số', 'chỗ', 'người', 'cái', 'chiếc', 'lần', 'chuyến', 'vé', 'ly', 'cốc', 'suất', 'bài', 'tấm', 'tờ', 'cây', 'quyển', 'cuốn', 'kw', 'kwh', 'inch', 'tỉ', 'tỷ', 'triệu', 'nghìn', 'ngàn', 'k', 'đồng'])})(?![^\W\d_])",
    re.IGNORECASE,
)


def _apply_period(hour: int, period: Optional[str]) -> int:
    if period is None:
        return hour
    p = norm_key(period)
    if p == "sang":
        return hour if hour != 12 else 0
    if p == "trua":
        return hour + 12 if hour < 11 else hour  # "1 giờ trưa" -> 13, "11 giờ trưa" -> 11
    if p in ("chieu", "toi"):
        return hour + 12 if hour < 12 else hour
    if p in ("dem", "khuya"):
        return hour + 12 if 6 <= hour < 12 else hour  # "11 giờ đêm" -> 23 ; "1 giờ đêm" -> 1
    return hour


@dataclass
class _DayCand:
    start: int
    end: int
    text: str
    date: Any
    note: str


def _is_name_mai(t: str, m: re.Match) -> bool:
    if nfc(m.group(0)).lower() != "mai":
        return False
    before = t[: m.start()].rstrip()
    prev = re.findall(r"[^\W\d_]+$", before)
    if prev and nfc(prev[0]).lower() in _NAME_TITLES:
        return True
    return m.group(0)[0].isupper() and bool(before) and before[-1] not in ".!?"


def _day_candidates(t: str, now: datetime) -> list[_DayCand]:
    cands: list[_DayCand] = []
    today = now.date()
    used: list[tuple[int, int]] = []

    def free(a, b):
        return not any(a < e and b > s0 for s0, e in used)

    for m in DATE_RE.finditer(t):
        try:
            y = int(m.group("y")) if m.group("y") else now.year
            d = today.replace(year=y, month=int(m.group("mo")), day=int(m.group("d")))
            if not m.group("y") and d < today:
                d = d.replace(year=y + 1)
        except ValueError:
            continue
        cands.append(_DayCand(m.start(), m.end(), m.group(0), d, "ngày/tháng"))
        used.append((m.start(), m.end()))
    for m in list(WORD_DATE_RE.finditer(t)) + list(WORD_DATE2_RE.finditer(t)):
        if not free(m.start(), m.end()):
            continue
        day = _num_words_value(m.group("dw"))
        if day is None or not 1 <= day <= 31:
            continue
        mw = m.group("mw")
        if mw:
            mo = _MONTH_NAMES.get(nfc(mw).lower()) or _num_words_value(mw)
            if mo is None or not 1 <= mo <= 12:
                continue
            y = now.year
            try:
                d = today.replace(year=y, month=mo, day=day)
            except ValueError:
                continue
            if d < today - timedelta(days=60):
                d = d.replace(year=y + 1)
        else:
            mo, y = now.month, now.year
            if day < today.day:            # the day already passed this month -> next month
                mo, y = (mo % 12) + 1, y + (1 if mo == 12 else 0)
            try:
                d = today.replace(year=y, month=mo, day=day)
            except ValueError:
                continue
        cands.append(_DayCand(m.start(), m.end(), m.group(0), d, "ngày trong tháng"))
        used.append((m.start(), m.end()))
    for m in WEEKDAY_RE.finditer(t):
        if not free(m.start(), m.end()):
            continue
        wd = WEEKDAYS.get(norm_key(m.group("wd")))
        if wd is None:
            continue
        mod = norm_key(m.group("mod") or "")
        if mod.startswith("tuan"):
            start_next_week = today + timedelta(days=7 - now.weekday())
            d = start_next_week + timedelta(days=wd)
        else:
            delta = (wd - now.weekday()) % 7
            d = today + timedelta(days=delta or 7)   # said on that weekday -> next occurrence
        cands.append(_DayCand(m.start(), m.end(), m.group(0), d, "thứ trong tuần"))
        used.append((m.start(), m.end()))
    for m in DAY_RE.finditer(t):
        if not free(m.start(), m.end()) or _is_name_mai(t, m):
            continue
        key = next((k for k in _DAY_WORDS if norm_key(k) == norm_key(m.group(0))), None)
        if key is None:
            continue
        cands.append(_DayCand(m.start(), m.end(), m.group(0), today + timedelta(days=_DAY_WORDS[key]), key))
        used.append((m.start(), m.end()))
    for m in MONTH_RE.finditer(t):
        if not free(m.start(), m.end()):
            continue
        nm = (today.replace(day=1) + timedelta(days=32)).replace(day=min(today.day, 28))
        cands.append(_DayCand(m.start(), m.end(), m.group(0), nm, "tháng sau"))
        used.append((m.start(), m.end()))
    cands.sort(key=lambda c: c.start)
    return cands


def _nearest(items, start: int, end: int, after: int, before: int):
    """Closest item (with .start/.end) to the span [start, end): prefer one starting within `after`
    chars after it, else one ending within `before` chars before it, else the closest anywhere."""
    best, best_d = None, None
    for it in items:
        if it.start >= end and it.start - end <= after:
            d = it.start - end
        elif it.end <= start and start - it.end <= before:
            d = start - it.end + 1000
        else:
            d = 10_000 + min(abs(it.start - end), abs(start - it.end))
        if best_d is None or d < best_d:
            best, best_d = it, d
    return best


@dataclass
class _Span:
    start: int
    end: int
    text: str


def extract_time_hints(text: str, current_time: str, roll_past_to_tomorrow: bool = True) -> list[TimeHint]:
    """Resolve Vietnamese clock times / days / durations against current_time (ISO with offset).

    Every clock time in the text gets one datetime hint; its day and period (sáng/chiều/tối) are the
    nearest ones in the sentence. A day expression that no clock time uses gets a date hint.
    """
    now = _parse_iso(current_time)
    tz = now.tzinfo
    t = nfc(text)
    # Central "ni" = "nay": "tối ni", "bữa ni", "hôm ni"
    t = re.sub(r"(?<![^\W\d_])(sáng|trưa|chiều|tối|đêm|hôm|bữa)\s+ni(?![^\W\d_])", lambda mm: mm.group(1) + " nay", t, flags=re.IGNORECASE)
    hints: list[TimeHint] = []
    days = _day_candidates(t, now)
    periods = [_Span(m.start(), m.end(), m.group(0)) for m in PERIOD_RE.finditer(t)]
    used_days: set[int] = set()

    for m in TIME_RE.finditer(t):
        h_raw = m.group("h") or m.group("h2") or m.group("h3") or m.group("h3b") or m.group("h3c")
        if m.group("h3c") and not _TIME_CONTEXT_RE.search(t):
            continue          # "năm rưỡi" alone is a time only in a clock context
        if h_raw is None:
            continue
        hour = _num_words_value(h_raw)
        if hour is None or hour > 24:
            continue
        if m.group("h3") or m.group("h3b") or m.group("h3c") or m.group("half"):
            minute = 30
        elif m.group("kem"):
            k = _num_words_value(m.group("kem"))
            if k is None or not 0 < k < 60:
                continue
            hour, minute = hour - 1, 60 - k
        elif m.group("m2") is not None:
            minute = int(m.group("m2"))
        elif m.group("m"):
            minute = _num_words_value(m.group("m"))
            if minute is None or not 0 <= minute < 60:
                minute = 0
        else:
            minute = 0
        period = None if m.group("h2") else _nearest(periods, m.start(), m.end(), after=14, before=16)
        if period is not None and not (period.start >= m.end() - 1 and period.start - m.end() <= 14
                                       or period.end <= m.start() and m.start() - period.end <= 16):
            period = None
        if period is not None and period.start >= m.end():   # "chín giờ, chiều thì ...": a later clause's period
            if re.search(r"[,.;!?]", t[m.end():period.start]):
                period = None
        ptext = period.text if period else None
        if ptext is None and not m.group("h2"):
            # range "từ hai rưỡi đến ba giờ mười lăm chiều": the end's period applies to the start too
            rng = re.match(rf"[^,.;!?]{{0,12}}?(?:{_v('đến')}|{_v('tới')})\s[^,.;!?]{{0,30}}?", t[m.end():])
            if rng:
                pm = next((p for p in periods if m.end() <= p.start <= m.end() + 45 and not re.search(r"[,.;!?]", t[m.end():p.start])), None)
                if pm is not None and norm_key(pm.text) in ("chieu", "toi", "dem"):
                    ptext = pm.text
        if ptext is None and not m.group("h2"):
            # no period next to the time: borrow it from the day anchor ("Tối nay ... từ bảy rưỡi tới tám giờ mười")
            d0 = _nearest(days, m.start(), m.end(), after=24, before=60)
            if d0 is not None:
                pm = PERIOD_RE.match(d0.text.strip())
                if pm:
                    ptext = pm.group(0)
        hour = _apply_period(hour, ptext)
        if hour == 24:
            hour = 0
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            continue
        day = _nearest(days, m.start(), m.end(), after=24, before=40)
        base_date = day.date if day else now.date()
        dt = datetime(base_date.year, base_date.month, base_date.day, hour, minute, tzinfo=tz)
        if day is not None:
            used_days.add(id(day))
            near = day.start >= m.end() and day.start - m.end() <= 24 or day.end <= m.start() and m.start() - day.end <= 40
            note = f'ngày theo "{day.text}"' + ("" if near else " (xa trong câu, không chắc)")
        elif roll_past_to_tomorrow and dt <= now:
            dt = dt + timedelta(days=1)
            note = "không nói ngày; giờ đó hôm nay đã qua → lấy ngày mai"
        else:
            note = "không nói ngày → hôm nay"
        hints.append(TimeHint(m.group(0).strip(), dt.isoformat(), "datetime", note))
        if len(hints) >= 6:
            break

    for d in days:
        if id(d) not in used_days:
            hints.append(TimeHint(d.text.strip(), d.date.isoformat(), "date", "chỉ có ngày"))

    for m in DURATION_RE.finditer(t):
        n_raw, unit = m.group("n"), norm_key(m.group("u"))
        tail = norm_key(m.group(0).split()[-1])
        if unit == "gio" and tail == "toi":          # "chín giờ tới" = the coming 9 o'clock, not 9 hours from now
            continue
        if re.search(r"(?<![^\W\d_])th[uứ]\s*$", t[:m.start()].rstrip() + " ", re.IGNORECASE) or norm_key(t[max(0, m.start() - 4):m.start()]).strip().endswith("thu"):
            continue                                  # "thứ bảy tuần sau" is a weekday, handled by WEEKDAY_RE
        n = 0.5 if norm_key(n_raw) == "nua" else _num_words_value(n_raw)
        if n is None:
            continue
        delta = {"phut": timedelta(minutes=n), "tieng": timedelta(hours=n), "gio": timedelta(hours=n),
                 "ngay": timedelta(days=n), "tuan": timedelta(weeks=n)}.get(unit)
        if delta is None:
            continue
        dt = now + delta
        hints.append(TimeHint(m.group(0).strip(), dt.replace(microsecond=0).isoformat(), "datetime", "tính từ current_time"))
    return hints


def extract_spans_minutes(text: str) -> list[tuple[str, int]]:
    """Plain durations as minutes: "nửa tiếng" -> 30, "45 phút" -> 45, "một tiếng rưỡi" -> 90 (for shifting event times)."""
    t = nfc(text)
    out: list[tuple[str, int]] = []
    for m in SPAN_RE.finditer(t):
        n_raw, unit = m.group("n"), norm_key(m.group("u"))
        n = 0.5 if norm_key(n_raw) == "nua" else _num_words_value(n_raw)
        if n is None:
            continue
        minutes = n * (60 if unit == "tieng" else 1)
        if m.group("half"):
            minutes += 30 if unit == "tieng" else 0.5
        if minutes > 0 and float(minutes).is_integer():
            out.append((m.group(0).strip(), int(minutes)))
    return out


# --------------------------------------------------------------------------- #
# Hint rendering for the LLM prompt
# --------------------------------------------------------------------------- #

_DIGIT_WORD = {"khong": 0, "linh": 0, "mot": 1, "hai": 2, "ba": 3, "bon": 4, "tu": 4, "nam": 5, "lam": 5, "sau": 6, "bay": 7,
               "tam": 8, "chin": 9, "muoi": 10}
_DW = r"(?:một|hai|ba|bốn|tư|năm|sáu|bảy|tám|chín|mười)"   # not "không": "quạt không cần bật" is a negation
_LONE_DIGIT_AFTER_RE = re.compile(
    rf"(?<![^\W\d_])(?:{_alt(['tầng', 'lầu', 'phòng', 'số', 'kênh', 'nhóm', 'quạt', 'bàn', 'cổng', 'tập', 'chỗ', 'hàng', 'ghế'])})\s+(?P<d>{_DW})(?![^\W\d_])"
    rf"(?!\s+(?:{_alt(['mươi', 'trăm', 'nghìn', 'ngàn', 'triệu', 'tỷ', 'mốt', 'lăm', 'tư', 'rưỡi', 'giờ', 'phút', 'người', 'cái', 'chiếc'])})(?![^\W\d_]))",
    re.IGNORECASE)
# phone numbers / codes read digit by digit, often in comma-separated groups: "không chín không một, hai ba bốn, ..."
_DIGIT_RUN_RE = re.compile(rf"(?<![^\W\d_])(?:(?:không|một|hai|ba|bốn|năm|sáu|bảy|tám|chín)[\s,]+){{3,}}(?:không|một|hai|ba|bốn|năm|sáu|bảy|tám|chín)(?![^\W\d_])", re.IGNORECASE)


def build_hints(transcript: str, current_time: str) -> list[str]:
    lines: list[str] = []
    time_hints: list[TimeHint] = []
    try:
        time_hints = extract_time_hints(transcript, current_time)
    except Exception:  # never let the hint layer crash inference
        time_hints = []
    time_texts = [nfc(h.text).lower() for h in time_hints]
    t = nfc(transcript)
    for s in extract_money(transcript):
        lines.append(f'Số tiền/số lớn: "{s.text}" = {s.int_value}')
    for s in extract_numbers(transcript):
        if s.has_unit or s.value >= 1000:
            continue
        if any(nfc(s.text).lower() in tt for tt in time_texts):
            continue  # digit that belongs to a time expression
        v = s.int_value if float(s.value).is_integer() else s.value
        line = f'Số: "{s.text}" = {v}'
        # "chuyển ba trăm" = 300 000 VND: an amount said without a unit means thousands
        if 100 <= s.value < 1000 and float(s.value).is_integer() and not _MEASURE_AFTER.match(t[s.end:s.end + 14]):
            line += f" (nếu là số tiền nói tắt: {int(s.value) * 1000})"
        lines.append(line)
    # format-invariant: "tầng hai" must hint like "tầng 2"; digits read one by one ("không chín không bốn ...") -> one string
    have = " ".join(lines)
    for m in _LONE_DIGIT_AFTER_RE.finditer(t):
        v = _DIGIT_WORD.get(norm_key(m.group("d")))
        if v is not None and f'"{m.group(0)}"' not in have:
            lines.append(f'Số: "{m.group(0)}" = {v}')
    for m in _DIGIT_RUN_RE.finditer(t):
        ds = [_DIGIT_WORD.get(norm_key(w)) for w in m.group(0).replace(",", " ").split()]
        if len(ds) >= 4 and all(d is not None for d in ds):
            lines.append(f'Dãy số đọc từng chữ: "{m.group(0)}" = "{"".join(str(d) for d in ds)}"')
    for h in time_hints:
        if h.kind == "datetime":
            lines.append(f'Thời gian: "{h.text}" → {h.iso} (= {h.iso[11:16]}; {h.note})')
        else:
            lines.append(f'Ngày: "{h.text}" → {h.iso} ({h.note})')
    for text, minutes in extract_spans_minutes(transcript):
        if not any(nfc(text).lower() in tt for tt in time_texts):
            lines.append(f'Khoảng thời gian: "{text}" = {minutes} phút')
    return lines


# --------------------------------------------------------------------------- #
# Hints derived from a tool result (event times shifted by spans said in the request)
# --------------------------------------------------------------------------- #

_ISO_IN_TEXT = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:\d{2}|Z)")


def result_time_hints(result: Any, transcript: str, max_stamps: int = 6, max_spans: int = 3) -> list[str]:
    """For every ISO timestamp in a tool result and every plain span in the request ("nửa tiếng", "45 phút"),
    the shifted timestamps (+ = muộn/sau/lùi, - = sớm/trước). Deterministic arithmetic for "dời lịch muộn
    nửa tiếng", "đón trước giờ hẹn 45 phút", "báo thức một tiếng trước lịch họp"."""
    try:
        text = json.dumps(result, ensure_ascii=False)
    except Exception:
        return []
    stamps: list[str] = []
    for st in _ISO_IN_TEXT.findall(text):
        if st not in stamps:
            stamps.append(st)
    spans = extract_spans_minutes(transcript)[:max_spans]
    if not stamps or not spans:
        return []
    lines: list[str] = []
    for st in stamps[:max_stamps]:
        try:
            dt = datetime.fromisoformat(st.replace("Z", "+00:00"))
        except ValueError:
            continue
        parts = []
        for txt, mins in spans:
            later = (dt + timedelta(minutes=mins)).isoformat()
            earlier = (dt - timedelta(minutes=mins)).isoformat()
            parts.append(f'"{txt}" muộn/sau → {later}; sớm/trước → {earlier}')
        lines.append(f"{st}: " + " | ".join(parts))
    return lines
