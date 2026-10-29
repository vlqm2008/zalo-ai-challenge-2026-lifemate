#!/usr/bin/env python3
"""Score the per-sample logs of a (possibly still running) dev run against gold and write <run>/dev_scores.jsonl,
so error analysis can start before run_dev.py finishes. Dev only (gold replay in a second env).

  python scripts/score_logs.py --data ./data --split train --run out/dev_oracle
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifemate.data import load_catalog, load_jsonl  # noqa: E402
from lifemate.runtime import import_toolenv, locate_split  # noqa: E402
from lifemate.scorer import aggregate, fuzzy_lookup_from_catalog, score_sample  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_dev import category, gold_hash_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    root, split, split_dir = locate_split(args.data, args.split)
    catalog = load_catalog(split_dir / f"{split}_tools_catalog.json")
    fuzzy_of = fuzzy_lookup_from_catalog(catalog)
    samples = {s["id"]: s for s in load_jsonl(split_dir / f"{split}.jsonl") if "label" in s}
    ToolEnv = import_toolenv(split_dir)
    run = Path(args.run)
    if not (run / "logs").is_dir():
        print(f"no logs yet under {run}")
        return
    rows, scores = [], []
    by_cat = defaultdict(list)
    for p in sorted((run / "logs").glob("*.json")):
        log = json.loads(p.read_text(encoding="utf-8"))
        sid = log.get("id") or p.stem
        if sid not in samples or "tool_calls" not in log:
            continue
        s = samples[sid]
        gold = s["label"]["tool_calls"]
        sc = score_sample(sid, log["tool_calls"], log.get("state_hash"), gold, gold_hash_for(ToolEnv, split, sid, root, gold), fuzzy_of)
        cat = category(gold, s)
        scores.append(sc)
        by_cat[cat].append(sc.score)
        rows.append(sc.as_dict() | {"category": cat, "wall_s": log.get("timing", {}).get("total"), "gold": gold, "sub": log["tool_calls"],
                                    "transcript": (log.get("transcripts") or [""])[-1], "oracle": s.get("audio_text"),
                                    "finished_reason": log.get("finished_reason")})
    agg = aggregate(scores)
    agg["by_category"] = {k: {"n": len(v), "score": 100 * sum(v) / len(v)} for k, v in sorted(by_cat.items())}
    agg["finished_reasons"] = dict(Counter(r["finished_reason"] for r in rows))
    (run / "dev_scores.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    print(json.dumps({k: agg[k] for k in ("final", "n", "perfect", "zero", "hash_mismatch", "mean_P", "mean_R", "by_category", "finished_reasons")}, ensure_ascii=False, indent=1))
    print("reason_buckets:", agg["reason_buckets"])


if __name__ == "__main__":
    main()
