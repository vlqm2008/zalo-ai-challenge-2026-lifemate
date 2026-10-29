"""Argument grounding: make non-fuzzy arguments byte-exact.

The LLM proposes a call; before it is executed we
  * coerce types to the schema (26 -> 26 int, "true" -> True),
  * snap x-ref IDs to IDs actually returned by the read tools of that app,
  * snap other non-fuzzy strings (addresses, names) to exact strings seen in
    the context or in tool results when they are near-identical,
  * fill timezone offsets on ISO timestamps.
Fuzzy (x-match: fuzzy) arguments are left untouched.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Optional

from .data import fuzzy_params, ref_params
from .normalize_vi import norm_key, norm_tokens


# --------------------------------------------------------------------------- #
# Entities seen so far
# --------------------------------------------------------------------------- #

class EntityStore:
    """IDs (and the records holding them) collected from tool results, per app."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict]] = []   # (app, id, record)
        self.strings: set[str] = set()                   # every string value seen (context + results)
        self.groups: list[list[dict]] = []               # records per tool result (to detect same-name duplicates)
        self.group_of: dict[str, int] = {}               # id -> index in groups (latest result wins)
        self.numbers: set[int] = set()                   # every integer seen in results / history (amount grounding)
        self.resolved_ids: set[str] = set()              # ids already used as arguments (history or this session)

    def add_context(self, context: dict) -> None:
        for s in _walk_strings(context):
            if len(s) >= 3 and not _NO_SNAP.search(s):
                self.strings.add(s)

    def add_result(self, tool_name: str, result: Any) -> None:
        app = tool_name.split(".")[0]
        group: list[dict] = []
        for rec in _walk_records(result):
            rid = rec.get("id")
            if isinstance(rid, (str, int)):
                self.records.append((app, str(rid), rec))
                group.append(rec)
                self.group_of[str(rid)] = len(self.groups)
        self.groups.append(group)
        for s in _walk_strings(result):
            if len(s) >= 3 and not _NO_SNAP.search(s):
                self.strings.add(s)
        for n in _walk_numbers(result):
            self.numbers.add(n)

    def add_used_arguments(self, args: dict) -> None:
        for v in (args or {}).values():
            if isinstance(v, str):
                self.resolved_ids.add(v)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                self.numbers.add(int(v))

    def is_ambiguous(self, rid: str) -> bool:
        """The record `rid` came from a result holding >= 2 records with the same (diacritics-insensitive) name."""
        gi = self.group_of.get(rid)
        if gi is None:
            return False
        rec = next((r for r in self.groups[gi] if str(r.get("id")) == rid), None)
        if rec is None or not rec.get("name"):
            return False
        key = norm_key(str(rec["name"]))
        # distinct ids: the same record nested several times in one result (a playlist under each of its tracks) is one item
        return len({str(r.get("id")) for r in self.groups[gi] if norm_key(str(r.get("name") or "")) == key}) >= 2

    def disambiguated(self, rid: str, said: str) -> bool:
        """Same-name candidates, but the user's words (this turn or earlier turns) match details that only `rid` has
        ("chị Mai bên hậu cần" -> the Mai whose relation/notes say hậu cần)."""
        gi = self.group_of.get(rid)
        if gi is None:
            return False
        rec = next((r for r in self.groups[gi] if str(r.get("id")) == rid), None)
        if rec is None:
            return False
        key = norm_key(str(rec.get("name") or ""))
        same = list({str(r.get("id")): r for r in self.groups[gi] if norm_key(str(r.get("name") or "")) == key}.values())
        words = set(norm_tokens(said or ""))

        def extra(r: dict) -> set:
            toks: set = set()
            for k, v in r.items():
                if k in ("id", "name"):
                    continue
                for st in _walk_strings(v):
                    toks |= set(norm_tokens(st))
            return (toks - set(norm_tokens(str(r.get("name") or "")))) & words

        mine = extra(rec)
        return bool(mine) and all(len(mine - extra(r)) > 0 and len(extra(r) - mine) == 0 for r in same if str(r.get("id")) != rid)

    def ids_for(self, ref: str) -> list[tuple[str, dict]]:
        app = ref.split(".")[0]
        return [(rid, rec) for a, rid, rec in self.records if a == app]

    def has_app(self, app: str) -> bool:
        return any(a == app for a, _, _ in self.records)


def _walk_records(obj: Any):
    if isinstance(obj, dict):
        if "id" in obj:
            yield obj
        for v in obj.values():
            yield from _walk_records(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_records(v)


def _walk_numbers(obj: Any):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_numbers(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield int(obj)
    elif isinstance(obj, str) and obj.isdigit():
        yield int(obj)


def _walk_strings(obj: Any):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v)
    elif isinstance(obj, str):
        yield obj


# --------------------------------------------------------------------------- #
# Similarity
# --------------------------------------------------------------------------- #

def similarity(a: str, b: str) -> float:
    ka, kb = norm_key(a), norm_key(b)
    if not ka or not kb:
        return 0.0
    if ka == kb:
        return 1.0
    seq = SequenceMatcher(None, ka, kb).ratio()
    ta, tb = set(norm_tokens(a)), set(norm_tokens(b))
    jac = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    return max(seq, 0.5 * seq + 0.5 * jac)


def _record_texts(rec: dict) -> list[str]:
    """Several ways a user may refer to a record: its name, 'type + room', 'name + room', ..."""
    get = lambda k: str(rec[k]) if rec.get(k) is not None else ""  # noqa: E731
    name, room, typ, rel, floor = get("name"), get("room"), get("type"), get("relation"), get("floor")
    outs = [name, get("title"), get("label"), get("address"), get("phone"), get("email"), rel]
    if room:
        outs += [f"{typ} {room}", f"{name} {room}", room]
    if floor:
        outs += [f"{name} tầng {floor}", f"{typ} tầng {floor}"]
    if rel and name:
        outs += [f"{rel} {name}"]
    return [o for o in outs if o.strip()]


def best_id(value: str, candidates: list[tuple[str, dict]]) -> tuple[Optional[str], float]:
    """Best candidate ID for an LLM-provided value (an id, a name, or a description)."""
    best, best_s = None, 0.0
    for rid, rec in candidates:
        if value == rid:
            return rid, 1.0
        s = similarity(value, rid)
        for txt in _record_texts(rec):
            s = max(s, similarity(value, txt))
        if s > best_s:
            best, best_s = rid, s
    return best, best_s


# --------------------------------------------------------------------------- #
# Type coercion
# --------------------------------------------------------------------------- #

_ISO_NO_TZ = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$")
# values that must never be "snapped" to a near-identical known string: timestamps, dates, times, numbers,
# phone numbers, codes (a 1-digit difference is a different value, not a spelling variant)
_NO_SNAP = re.compile(r"^[\d\s:+\-./TZ,()]+$|^\d{4}-\d{2}-\d{2}|^\d{1,2}:\d{2}|^[+\d][\d\s.-]{6,}$|@|^https?://", re.IGNORECASE)
_EMPTY = (None, "", [], {})


def snappable(v: str) -> bool:
    """Only free text with letters (names, addresses) may be snapped; never timestamps/numbers/phones/codes."""
    return isinstance(v, str) and len(v) >= 6 and not _NO_SNAP.search(v) and any(c.isalpha() for c in v)


def coerce_value(schema: dict, value: Any, tz_offset: Optional[str]) -> Any:
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    if value is None:
        return value
    try:
        if t == "integer":
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return int(round(value))
            if isinstance(value, str):
                s = value.strip().replace(".", "").replace(",", "")
                m = re.match(r"^-?\d+$", s)
                if m:
                    return int(s)
        elif t == "number":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                s = value.strip().replace(",", ".")
                if re.match(r"^-?\d+(\.\d+)?$", s):
                    f = float(s)
                    return int(f) if f.is_integer() else f
            if isinstance(value, float) and value.is_integer():
                return int(value)
        elif t == "boolean":
            if isinstance(value, str):
                v = value.strip().lower()
                if v in ("true", "yes", "1", "on"):
                    return True
                if v in ("false", "no", "0", "off"):
                    return False
        elif t == "string":
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                value = str(value)
            if isinstance(value, str):
                value = value.strip()
                if tz_offset and _ISO_NO_TZ.match(value):
                    if len(value) == 16:
                        value += ":00"
                    value += tz_offset
                enum = schema.get("enum")
                if enum and value not in enum:
                    # case-insensitive / diacritics-insensitive enum snap
                    for e in enum:
                        if isinstance(e, str) and norm_key(e) == norm_key(value):
                            return e
        elif t == "array" and isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
    except Exception:
        return value
    return value


_HOME_PHRASES = {"nha", "nha toi", "nha minh", "nha em", "nha anh", "nha chi", "o nha", "tu nha", "ve nha", "nha rieng", "home", "home address"}
_WORK_PHRASES = {"cong ty", "co quan", "cho lam", "van phong", "cty", "noi lam viec", "cho lam viec", "o cong ty", "tu cong ty", "work", "work address"}


_ADDRESS_PARAMS = {"pickup", "dropoff", "origin", "destination", "address", "location", "from", "to", "start_location",
                   "end_location", "delivery_address", "place"}


def place_from_profile(value: str, context: dict) -> Optional[str]:
    """'nhà' -> user_profile.home_address, 'công ty' -> user_profile.work_address (gold passes the address verbatim)."""
    prof = (context or {}).get("user_profile") or {}
    k = norm_key(value)
    if k in _HOME_PHRASES and prof.get("home_address"):
        return str(prof["home_address"])
    if k in _WORK_PHRASES and prof.get("work_address"):
        return str(prof["work_address"])
    return None


# contacts search queries (gold convention measured on train, scripts in EXPERIMENTS 3.x): the name as spoken,
# generic peer titles dropped ("anh Tuấn" -> "Tuấn", "bạn Minh" -> "Minh"), kinship/role kept ("Bác Ba", "Sếp Dũng",
# "Mẹ"), trailing possessives dropped ("bố em" -> "Bố"), first letters capitalised (ASR output is lower-case).
_PEER_TITLES = {"anh", "chị", "chi", "em", "bạn", "ban"}
_POSSESSIVES = {"em", "anh", "chị", "tôi", "toi", "mình", "minh", "tao", "tui", "con", "cháu", "chau", "ấy", "ay", "ơi", "oi", "nhé", "nha"}
_LOWER_IN_NAME = {"giáo", "thủ", "quỹ", "bên", "của", "ở", "và", "kho", "phòng", "đội", "nhóm", "trưởng", "phó", "ban"}


def person_query_params(tool: dict) -> set[str]:
    """String params of a read tool that resolves `*.contacts` references (contacts.search.query-like)."""
    if tool.get("kind") != "read":
        return set()
    props = (tool.get("parameters") or {}).get("properties") or {}
    app = tool.get("app") or tool.get("name", "").split(".")[0]
    if not (app == "contacts" or "liên hệ" in (tool.get("description") or "").lower()):
        return set()
    return {k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string"
            and any(w in (v.get("description") or "").lower() for w in ("tên", "quan hệ", "name"))}


def normalize_person_query(q: str) -> str:
    toks = q.strip().split()
    if not toks or any(ch.isdigit() for ch in q):
        return q.strip()
    if len(toks) > 1 and toks[0].lower() in _PEER_TITLES:
        toks = toks[1:]
    while len(toks) > 1 and toks[-1].lower() in _POSSESSIVES:
        toks = toks[:-1]
    out = []
    for i, t in enumerate(toks):
        if i > 0 and t.lower() in _LOWER_IN_NAME:
            out.append(t.lower())
        else:
            out.append(t[0].upper() + t[1:] if t[0].islower() else t)
    return " ".join(out)


def tz_offset_of(context: dict) -> Optional[str]:
    ct = (context or {}).get("current_time")
    if isinstance(ct, str):
        m = re.search(r"([+-]\d{2}:\d{2}|Z)$", ct)
        if m:
            return "+00:00" if m.group(1) == "Z" else m.group(1)
    return None


# --------------------------------------------------------------------------- #
# Snapping
# --------------------------------------------------------------------------- #

def ground_arguments(tool: dict, args: dict, store: EntityStore, context: dict,
                     snap_threshold: float = 0.8) -> tuple[dict, list[str]]:
    """Return (grounded_args, notes). Never raises."""
    notes: list[str] = []
    props = (tool.get("parameters") or {}).get("properties") or {}
    refs = ref_params(tool)
    fuzzy = fuzzy_params(tool)
    person_q = person_query_params(tool)
    tz = tz_offset_of(context)
    out: dict[str, Any] = {}
    for k, v in (args or {}).items():
        if v in _EMPTY:
            notes.append(f"{k}: bỏ giá trị rỗng")   # zaic_tools treats None/""/[]/{} as "not given"; gold omits them
            continue
        schema = props.get(k) if isinstance(props.get(k), dict) else {}
        v = coerce_value(schema, v, tz)
        if isinstance(v, str):
            # only address-like params get the profile address; e.g. bike_rental.find_bikes.near keeps the spoken street
            addr = place_from_profile(v, context) if (norm_key(k) in _ADDRESS_PARAMS or (
                k == "near" and not str(tool.get("name", "")).startswith("bike_rental."))) else None
            if addr is not None:
                notes.append(f"{k}: '{v}' -> địa chỉ trong user_profile")
                v = addr
            elif k in person_q:
                nv = normalize_person_query(v)
                if nv != v:
                    notes.append(f"{k}: '{v}' -> '{nv}' (tên liên hệ)")
                    v = nv
        if k in fuzzy or not isinstance(v, str):
            out[k] = v
            continue
        if k in refs:
            cands = store.ids_for(refs[k])
            if cands:
                rid, s = best_id(v, cands)
                if rid is not None and rid != v and s >= snap_threshold:
                    notes.append(f"{k}: '{v}' -> '{rid}' ({s:.2f})")
                    v = rid
                elif rid is None or s < snap_threshold:
                    notes.append(f"{k}: '{v}' không khớp ID nào ({s:.2f})")
            out[k] = v
            continue
        # non-fuzzy free string (name, address): snap to an exact known string if near-identical.
        # Never for timestamps / numbers / phones: "06:00" vs "09:00" is a different value (measured bug).
        if v not in store.strings and snappable(v):
            best, best_s = None, 0.0
            for s in store.strings:
                sc = similarity(v, s)
                if sc > best_s:
                    best, best_s = s, sc
            if best is not None and best_s >= max(snap_threshold, 0.9) and snappable(best) \
                    and 0.6 <= len(best) / max(1, len(v)) <= 1.6:
                notes.append(f"{k}: '{v}' -> '{best}' ({best_s:.2f})")
                v = best
        out[k] = v
    return out, notes
