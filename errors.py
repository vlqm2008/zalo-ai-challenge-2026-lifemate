"""Why did the agent lose points? One main reason per non-perfect train sample, plus argument-level diffs.

    python errors.py runs/sft_dev_seen.jsonl runs/sft_dev_unseen.jsonl [--show 3]
"""
import argparse
import json
from collections import Counter, defaultdict

from score import ROOT, call_match, load_catalog, sample_score

ASK = "sys.ask_missing_info"


def reason(pred, gold, hash_ok, kind, fuzzy):
    """The first failure that applies, from the most to the least fundamental."""
    gn, pn = Counter(c["name"] for c in gold), Counter(c["name"] for c in pred)
    if not gold:
        return "acted on a no-op request"
    if not pred:
        return "did nothing"
    if (ASK in gn) != (ASK in pn):
        return "missed ask" if ASK in gn else "asked instead of acting"
    writes = lambda n: sorted(x for x in n.elements() if kind.get(x) == "write")
    if writes(gn) != writes(pn):
        return "missing write" if len(writes(pn)) < len(writes(gn)) else "extra/other write"
    if not hash_ok:
        return "right write tools, wrong arguments"
    unmatched = [c for c in pred if not any(call_match(c, g, fuzzy) for g in gold)]
    if not unmatched:
        return "missing reads"
    if any(c["name"] == ASK for c in unmatched):
        return "ask with wrong params"
    # would match if free text were ignored and other strings compared case-insensitively
    loose = lambda c, g: c["name"] == g["name"] and set(c.get("arguments") or {}) == set(g["arguments"]) and all(
        k in fuzzy.get(g["name"], ()) or str(c["arguments"][k]).lower() == str(v).lower() for k, v in g["arguments"].items())
    if all(any(loose(c, g) for g in gold) for c in unmatched):
        return "state ok, only wording/casing differs"
    return "state ok, extra or different calls"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--show", type=int, default=0, help="print this many examples per reason")
    a = ap.parse_args()

    import sys
    sys.path.insert(0, str(ROOT / "train"))
    from zaic_tools import ToolEnv

    def replay_hash(sid, calls):
        env = ToolEnv("train", sid, root=ROOT)
        for c in calls:
            env.call(c.get("name", ""), c.get("arguments") or {})
        return env.get_state_hash()

    kind, fuzzy = load_catalog(ROOT)
    gold = {d["id"]: d for d in map(json.loads, open(ROOT / "train/train.jsonl", encoding="utf-8"))}
    by_reason, argdiff, total = defaultdict(list), Counter(), 0
    for f in a.runs:
        for p in map(json.loads, open(f, encoding="utf-8")):
            total += 1
            g, pc = gold[p["id"]]["label"]["tool_calls"], p["tool_calls"]
            gh, ph = replay_hash(p["id"], g), replay_hash(p["id"], pc)
            s = sample_score(pc, ph, g, gh, fuzzy)
            if s == 1.0:
                continue
            by_reason[reason(pc, g, gh == ph, kind, fuzzy)].append((p["id"], 1 - s))
            for c in pc:  # same tool as a gold call but no match: which arguments differ
                same = [x for x in g if x["name"] == c["name"]]
                if same and not any(call_match(c, x, fuzzy) for x in same):
                    pa, ga = c.get("arguments") or {}, same[0]["arguments"]
                    argdiff.update(f"{c['name']}: extra {k}" for k in set(pa) - set(ga))
                    argdiff.update(f"{c['name']}: missing {k}" for k in set(ga) - set(pa))
                    argdiff.update(f"{c['name']}: value {k}" for k in set(pa) & set(ga) if k not in fuzzy.get(c["name"], ()) and pa[k] != ga[k])

    print(f"points lost per reason (out of {total} samples = {total} points):")
    for r, v in sorted(by_reason.items(), key=lambda x: -sum(l for _, l in x[1])):
        print(f"  {sum(l for _, l in v):6.1f}  {r:36} n={len(v)}")
        for sid, _ in v[: a.show]:
            d = gold[sid]
            pred = next(p for f in a.runs for p in map(json.loads, open(f, encoding="utf-8")) if p["id"] == sid)
            print(f"      [{sid}] now={d['context']['current_time'][:16]} \"{d['audio_text']}\"")
            print(f"        gold: {json.dumps([[c['name'], c['arguments']] for c in d['label']['tool_calls']], ensure_ascii=False)}")
            print(f"        pred: {json.dumps([[c['name'], c['arguments']] for c in pred['tool_calls']], ensure_ascii=False)}")
    print("argument diffs on matching tool names:", dict(argdiff.most_common(15)))


if __name__ == "__main__":
    main()
