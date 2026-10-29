#!/usr/bin/env python3
"""Group a dev run by failure reason and print concrete samples per bucket (Phase 3 error analysis).

  python scripts/error_buckets.py --run out/dev_oracle [--top 6] [--per 10] [--bucket arg_mismatch]

Reads <run>/dev_scores.jsonl (+ <run>/logs/<id>.json for the LLM thoughts). Buckets = first token of each
reason (hash_mismatch, missing, extra, arg_mismatch, no_match, called_nothing, gold_empty_but_called) and,
for missing/extra/arg_mismatch, the tool name too.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--per", type=int, default=10)
    ap.add_argument("--bucket", default=None, help="only this bucket (prefix match)")
    ap.add_argument("--category", default=None)
    args = ap.parse_args()
    run = Path(args.run)
    rows = [json.loads(l) for l in (run / "dev_scores.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.category:
        rows = [r for r in rows if r.get("category", "").startswith(args.category)]
    coarse, fine = Counter(), Counter()
    members: dict[str, list] = defaultdict(list)
    lost = Counter()
    for r in rows:
        seen = set()
        for reason in r.get("reasons") or []:
            parts = reason.split(" ")
            c = parts[0]
            f = " ".join(parts[:2]) if c in ("missing", "extra", "arg_mismatch") else c
            coarse[c] += 1
            fine[f] += 1
            if f not in seen:
                members[f].append(r)
                seen.add(f)
                lost[f] += 1 - r["score"]
    n = len(rows)
    final = 100 * sum(r["score"] for r in rows) / max(1, n)
    print(f"run={run} n={n} final={final:.2f}  perfect={sum(1 for r in rows if r['score'] >= 0.999)} zero={sum(1 for r in rows if r['score'] == 0)}")
    print("coarse buckets (reason counts):", dict(coarse.most_common()))
    print("\nfine buckets: count | score lost (sum of 1-score over samples in bucket) | bucket")
    for f, c in fine.most_common(40):
        print(f"  {c:4d} | {lost[f]:6.1f} | {f}")
    chosen = [f for f, _ in fine.most_common() if not args.bucket or f.startswith(args.bucket)][: args.top]
    for f in chosen:
        print(f"\n{'=' * 100}\n### bucket: {f}  ({fine[f]} reasons, {len(members[f])} samples)")
        for r in members[f][: args.per]:
            print(f"\n--- {r['sample_id']} [{r.get('category')}] score={r['score']:.2f} M={r['M']} U={r['U']} G={r['G']} hash_ok={r['hash_ok']} finished={r.get('finished_reason')}")
            print(f"  text: {r.get('transcript')}")
            if r.get("oracle") and r.get("oracle") != r.get("transcript"):
                print(f"  gold text: {r.get('oracle')}")
            print(f"  reasons: {r.get('reasons')}")
            print("  GOLD:")
            for c in r.get("gold") or []:
                print("    ", json.dumps(c, ensure_ascii=False))
            print("  SUBMITTED:")
            for c in r.get("sub") or []:
                print("    ", json.dumps(c, ensure_ascii=False))
            lp = run / "logs" / f"{r['sample_id']}.json"
            if lp.exists():
                log = json.loads(lp.read_text(encoding="utf-8"))
                for st in log.get("steps", []):
                    raw = st.get("raw") or ""
                    try:
                        th = json.loads(raw).get("thought", "")
                    except Exception:
                        th = raw[:120]
                    ev = st.get("event") or ("call" if st.get("call") else "")
                    print(f"    step{st.get('step')} {ev}: {th[:160]}" + (f"  | grounding={st['grounding']}" if st.get("grounding") else "") + (f"  | auto_reads={st['auto_reads']}" if st.get("auto_reads") else ""))
                if log.get("hints"):
                    print(f"    hints: {log['hints']}")


if __name__ == "__main__":
    main()
