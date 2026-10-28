"""Local LifeMate scorer for the train split, following the official formula (portal > Evaluation).

    python score.py submission.jsonl [--ids splits/dev_unseen.txt] [--root train+public_test]

Both state hashes come from replaying calls with zaic_tools, as the organisers do: gold = label.tool_calls,
pred = the submitted tool_calls (the submitted state_hash is only checked for consistency).
"""
import argparse
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "train+public_test"


def token_f1(a, b):
    # shortcut: whitespace tokens after lower + NFC; the official tokenizer is not published, recheck if leaderboard disagrees
    ta = unicodedata.normalize("NFC", str(a).lower()).split()
    tb = unicodedata.normalize("NFC", str(b).lower()).split()
    common = sum((Counter(ta) & Counter(tb)).values())
    if not common:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def call_match(pred, gold, fuzzy):
    """Same tool, same argument names, non-fuzzy values identical, fuzzy values token-F1 > 0.6."""
    if pred.get("name") != gold["name"]:
        return False
    pa, ga = pred.get("arguments") or {}, gold.get("arguments") or {}
    if set(pa) != set(ga):  # assumption: an extra or missing argument breaks the match
        return False
    fz = fuzzy.get(gold["name"], set())
    return all(token_f1(pa[k], v) > 0.6 if k in fz else pa[k] == v for k, v in ga.items())


def n_matched(pred, gold, fuzzy):
    """Size of the maximum one-to-one matching between pred and gold calls (order ignored)."""
    adj = [[j for j, g in enumerate(gold) if call_match(p, g, fuzzy)] for p in pred]
    owner = {}

    def take(i, seen):
        for j in adj[i]:
            if j not in seen:
                seen.add(j)
                if j not in owner or take(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    return sum(take(i, set()) for i in range(len(pred)))


def sample_score(pred_calls, pred_hash, gold_calls, gold_hash, fuzzy):
    if not gold_calls:
        return float(not pred_calls)
    m = n_matched(pred_calls, gold_calls, fuzzy)
    if pred_hash != gold_hash or m == 0:
        return 0.0
    p, r = m / len(pred_calls), m / len(gold_calls)  # len(pred) = M + U
    return 2 * p * r / (p + r)


def load_catalog(root):
    tools = [t for a in json.load(open(root / "train/train_tools_catalog.json", encoding="utf-8"))["apps"] for t in a["tools"]]
    kind = {t["name"]: t["kind"] for t in tools}
    fuzzy = {t["name"]: {k for k, p in t["parameters"]["properties"].items() if p.get("x-match") == "fuzzy"} for t in tools}
    return kind, fuzzy


def category(gold_calls, kind):
    names = [c["name"] for c in gold_calls]
    if not names:
        return "empty"
    if "sys.ask_missing_info" in names:
        return "ask"
    return "read_only" if all(kind.get(n) == "read" for n in names) else "write"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("--ids", help="file with one train id per line (default: every id in pred)")
    ap.add_argument("--root", type=Path, default=ROOT)
    a = ap.parse_args()

    sys.path.insert(0, str(a.root / "train"))
    from zaic_tools import ToolEnv

    def replay_hash(sid, calls):
        env = ToolEnv("train", sid, root=a.root)
        for c in calls:
            env.call(c.get("name", ""), c.get("arguments") or {})
        return env.get_state_hash()

    gold = {d["id"]: d for d in map(json.loads, open(a.root / "train/train.jsonl", encoding="utf-8"))}
    pred = {p["id"]: p for p in map(json.loads, filter(str.strip, open(a.pred, encoding="utf-8")))}
    ids = [l.strip() for l in open(a.ids, encoding="utf-8") if l.strip()] if a.ids else list(pred)
    kind, fuzzy = load_catalog(a.root)

    by_cat = defaultdict(list)
    missing = hash_fail = stale_hash = 0
    for sid in ids:
        g = gold[sid]["label"]["tool_calls"]
        cat = category(g, kind)
        if sid not in pred:
            missing += 1
            by_cat[cat].append(0.0)
            continue
        p = pred[sid].get("tool_calls") or []
        ph, gh = replay_hash(sid, p), replay_hash(sid, g)
        hash_fail += bool(g) and ph != gh
        stale_hash += pred[sid].get("state_hash") not in (None, ph)
        by_cat[cat].append(sample_score(p, ph, g, gh, fuzzy))

    scores = [s for v in by_cat.values() for s in v]
    print(f"score {100 * sum(scores) / len(scores):.2f}  (n={len(scores)}, missing={missing}, hash_fail={hash_fail})")
    for cat, v in sorted(by_cat.items()):
        print(f"  {cat:10} {100 * sum(v) / len(v):6.2f}  n={len(v)}")
    if stale_hash:
        print(f"warning: {stale_hash} submitted state_hash differ from the replayed one (the replayed one is used)")


if __name__ == "__main__":
    main()
