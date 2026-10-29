"""Local re-implementation of the LifeMate metric.

Per sample:
  gold empty            -> 1 if submitted empty else 0
  hash mismatch or M=0  -> 0
  otherwise             -> F1 with P = M/(M+U), R = M/|G|
Final = 100/N * sum(score_i)

A submitted call matches a gold call when the name is the same, every
non-fuzzy argument is identical and every fuzzy argument (x-match: fuzzy in the
catalog) has token-F1 > 0.6 after lowercasing + NFC. Each call matches at most
one call; order is ignored (maximum bipartite matching).
"""
from __future__ import annotations

import json
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

FUZZY_THRESHOLD = 0.6


def _tokens(s: Any) -> list[str]:
    return unicodedata.normalize("NFC", str(s)).lower().split()


def token_f1(a: Any, b: Any) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    common = sum((Counter(ta) & Counter(tb)).values())
    if common == 0:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def _eq(a: Any, b: Any) -> bool:
    """JSON-value equality; bool is not a number here, 26 == 26.0."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, str) and isinstance(b, str):
        return unicodedata.normalize("NFC", a) == unicodedata.normalize("NFC", b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)
    return a == b


FuzzyLookup = Callable[[str], set[str]]


def call_matches(sub: dict, gold: dict, fuzzy_of: FuzzyLookup) -> bool:
    if sub.get("name") != gold.get("name"):
        return False
    sa, ga = sub.get("arguments") or {}, gold.get("arguments") or {}
    if set(sa.keys()) != set(ga.keys()):
        return False
    fuzzy = fuzzy_of(gold["name"])
    for k, gv in ga.items():
        sv = sa[k]
        if k in fuzzy:
            if not (token_f1(sv, gv) > FUZZY_THRESHOLD):
                return False
        elif not _eq(sv, gv):
            return False
    return True


def max_matching(sub_calls: list[dict], gold_calls: list[dict], fuzzy_of: FuzzyLookup) -> tuple[int, list[tuple[int, int]]]:
    """Maximum bipartite matching (Kuhn). Returns (M, pairs of (sub_idx, gold_idx))."""
    n_s, n_g = len(sub_calls), len(gold_calls)
    adj = [[j for j in range(n_g) if call_matches(sub_calls[i], gold_calls[j], fuzzy_of)] for i in range(n_s)]
    match_g = [-1] * n_g

    def try_kuhn(i: int, seen: list[bool]) -> bool:
        for j in adj[i]:
            if seen[j]:
                continue
            seen[j] = True
            if match_g[j] == -1 or try_kuhn(match_g[j], seen):
                match_g[j] = i
                return True
        return False

    m = 0
    for i in range(n_s):
        if try_kuhn(i, [False] * n_g):
            m += 1
    pairs = [(match_g[j], j) for j in range(n_g) if match_g[j] != -1]
    return m, pairs


@dataclass
class SampleScore:
    sample_id: str
    score: float
    hash_ok: bool
    M: int
    U: int
    G: int
    P: float
    R: float
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _diff_reasons(sub_calls, gold_calls, pairs, fuzzy_of) -> list[str]:
    matched_s = {i for i, _ in pairs}
    matched_g = {j for _, j in pairs}
    reasons = []
    for j, g in enumerate(gold_calls):
        if j in matched_g:
            continue
        # closest same-name submitted call that is unmatched
        cands = [i for i, s in enumerate(sub_calls) if i not in matched_s and s.get("name") == g.get("name")]
        if cands:
            s = sub_calls[cands[0]]
            sa, ga = s.get("arguments") or {}, g.get("arguments") or {}
            fuzzy = fuzzy_of(g["name"])
            bad = []
            for k in set(sa) | set(ga):
                if k not in sa:
                    bad.append(f"-{k}")
                elif k not in ga:
                    bad.append(f"+{k}")
                elif k in fuzzy:
                    if not token_f1(sa[k], ga[k]) > FUZZY_THRESHOLD:
                        bad.append(f"{k}(fuzzy={token_f1(sa[k], ga[k]):.2f})")
                elif not _eq(sa[k], ga[k]):
                    bad.append(f"{k}:{json.dumps(sa[k], ensure_ascii=False)}!={json.dumps(ga[k], ensure_ascii=False)}")
            reasons.append(f"arg_mismatch {g['name']} [{', '.join(bad)}]")
        else:
            reasons.append(f"missing {g.get('name')}")
    for i, s in enumerate(sub_calls):
        if i in matched_s:
            continue
        if not any(g.get("name") == s.get("name") for j, g in enumerate(gold_calls) if j not in matched_g):
            reasons.append(f"extra {s.get('name')}")
    return reasons


def score_sample(sample_id: str, sub_calls: Optional[list[dict]], sub_hash: Optional[str],
                 gold_calls: list[dict], gold_hash: Optional[str], fuzzy_of: FuzzyLookup) -> SampleScore:
    sub_calls = sub_calls or []
    G = len(gold_calls)
    if G == 0:
        ok = len(sub_calls) == 0
        return SampleScore(sample_id, 1.0 if ok else 0.0, True, 0, len(sub_calls), 0, 0.0, 0.0,
                           [] if ok else ["gold_empty_but_called"])
    M, pairs = max_matching(sub_calls, gold_calls, fuzzy_of)
    U = len(sub_calls) - M
    P = M / (M + U) if (M + U) else 0.0
    R = M / G
    hash_ok = (gold_hash is None) or (sub_hash == gold_hash)
    reasons = _diff_reasons(sub_calls, gold_calls, pairs, fuzzy_of)
    if not sub_calls:
        reasons.insert(0, "called_nothing")
    if not hash_ok:
        reasons.insert(0, "hash_mismatch")
    if M == 0:
        reasons.insert(0, "no_match")
    if not hash_ok or M == 0:
        return SampleScore(sample_id, 0.0, hash_ok, M, U, G, P, R, reasons)
    f1 = 2 * P * R / (P + R)
    return SampleScore(sample_id, f1, hash_ok, M, U, G, P, R, reasons)


def fuzzy_lookup_from_catalog(catalog: dict) -> FuzzyLookup:
    table: dict[str, set[str]] = {}
    for app in catalog.get("apps", []):
        for tool in app.get("tools", []):
            props = (tool.get("parameters") or {}).get("properties") or {}
            table[tool["name"]] = {k for k, v in props.items() if isinstance(v, dict) and v.get("x-match") == "fuzzy"}
    return lambda name: table.get(name, set())


def aggregate(scores: Iterable[SampleScore]) -> dict:
    scores = list(scores)
    n = len(scores)
    if n == 0:
        return {"final": 0.0, "n": 0}
    total = sum(s.score for s in scores)
    buckets: Counter = Counter()
    for s in scores:
        for r in s.reasons:
            buckets[r.split(" ")[0]] += 1
    return {
        "final": 100.0 * total / n,
        "n": n,
        "perfect": sum(1 for s in scores if s.score >= 0.999),
        "zero": sum(1 for s in scores if s.score == 0.0),
        "hash_mismatch": sum(1 for s in scores if not s.hash_ok),
        "mean_P": sum(s.P for s in scores if s.G) / max(1, sum(1 for s in scores if s.G)),
        "mean_R": sum(s.R for s in scores if s.G) / max(1, sum(1 for s in scores if s.G)),
        "reason_buckets": dict(buckets.most_common()),
    }
