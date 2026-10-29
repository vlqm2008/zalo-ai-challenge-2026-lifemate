#!/usr/bin/env python3
"""Run the agent on labelled train samples and score them with the local metric.

  python scripts/run_dev.py --data ./data --ids-file splits/dev_ids.txt --oracle      # agent-only loss
  python scripts/run_dev.py --data ./data --ids-file splits/dev_ids.txt               # with ASR

Prints Final score, P/R, reason buckets, latency p50/p95, and writes per-sample rows to
<out-dir>/dev_scores.jsonl for error analysis. Gold hash = replay of the gold calls in a
SECOND env (dev only; at inference exactly one env per sample is created).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifemate.agent import Agent  # noqa: E402
from lifemate.config import Config  # noqa: E402
from lifemate.data import ASK_TOOL_NAME, load_catalog, load_jsonl  # noqa: E402
from lifemate.runtime import build_components, import_toolenv, locate_split  # noqa: E402
from lifemate.scorer import aggregate, fuzzy_lookup_from_catalog, score_sample  # noqa: E402


def gold_hash_for(ToolEnv, split, sid, root, gold_calls):
    env = ToolEnv(split, sid, root=str(root))
    for c in gold_calls:
        env.call(c["name"], c.get("arguments") or {})
    return env.get_state_hash()


def category(gold_calls: list[dict], sample: dict) -> str:
    if not gold_calls:
        return "oos"
    if any(c["name"] == ASK_TOOL_NAME for c in gold_calls):
        return "ask"
    kinds = "multi" if len(gold_calls) > 2 else "single"
    return kinds + ("+hist" if sample.get("history") else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--out-dir", default="out/dev")
    ap.add_argument("--llm-backend", default=None)
    ap.add_argument("--asr-backend", default=None)
    ap.add_argument("--asr-cache", default=None, help="JSON from precompute_asr.py (implies --asr-backend cache)")
    ap.add_argument("--mock-env", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config, {"llm_backend": args.llm_backend, "asr_backend": "cache" if args.asr_cache else args.asr_backend,
                                    "asr_cache_path": args.asr_cache})
    root, split, split_dir = locate_split(args.data, args.split)
    catalog = load_catalog(split_dir / f"{split}_tools_catalog.json")
    fuzzy_of = fuzzy_lookup_from_catalog(catalog)
    samples = list(load_jsonl(split_dir / f"{split}.jsonl"))
    if args.ids_file:
        keep = {l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()}
        samples = [s for s in samples if s["id"] in keep]
    if args.limit:
        samples = samples[: args.limit]
    samples = [s for s in samples if "label" in s]
    if args.mock_env:
        from lifemate.mock_env import MockToolEnv as ToolEnv
    else:
        ToolEnv = import_toolenv(split_dir)

    out_dir = Path(args.out_dir)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    llm, asr = build_components(cfg, oracle_only=args.oracle)
    agent = Agent(cfg, llm, asr, catalog)

    scores, rows, times = [], [], []
    by_cat: dict[str, list] = defaultdict(list)
    scores_f = open(out_dir / "dev_scores.jsonl", "w", encoding="utf-8")   # one row per finished sample (crash-safe)
    for i, sample in enumerate(samples):
        sid = sample["id"]
        gold = sample["label"]["tool_calls"]
        t0 = time.monotonic()
        try:
            env = ToolEnv(split, sid, root=str(root))
            log = agent.run(sample, env, split_dir, oracle=args.oracle)
            sub_calls, sub_hash = log["tool_calls"], log["state_hash"]
        except Exception as e:
            log = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
            sub_calls, sub_hash = [], ""
        dt = time.monotonic() - t0
        times.append(dt)
        try:
            ghash = gold_hash_for(ToolEnv, split, sid, root, gold)
        except Exception as e:
            ghash = None
            log["gold_hash_error"] = str(e)
        s = score_sample(sid, sub_calls, sub_hash, gold, ghash, fuzzy_of)
        cat = category(gold, sample)
        scores.append(s)
        by_cat[cat].append(s.score)
        row = s.as_dict() | {"category": cat, "wall_s": round(dt, 2), "gold": gold, "sub": sub_calls,
                             "transcript": (log.get("transcripts") or [""])[-1], "oracle": sample.get("audio_text"),
                             "finished_reason": log.get("finished_reason")}
        rows.append(row)
        scores_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        scores_f.flush()
        (out_dir / "logs" / f"{sid}.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
        if (i + 1) % 10 == 0:
            print(f"[dev] {i + 1}/{len(samples)} running={100 * sum(x.score for x in scores) / len(scores):.1f}", flush=True)

    agg = aggregate(scores)
    srt = sorted(times)
    agg["latency"] = {"p50": srt[len(srt) // 2], "p95": srt[int(0.95 * (len(srt) - 1))], "max": srt[-1],
                      "a30_est_p95": 2 * srt[int(0.95 * (len(srt) - 1))]} if srt else {}
    agg["by_category"] = {k: {"n": len(v), "score": 100 * sum(v) / len(v)} for k, v in sorted(by_cat.items())}
    agg["finished_reasons"] = dict(Counter(r["finished_reason"] for r in rows))
    agg["mode"] = "oracle" if args.oracle else "asr"
    scores_f.close()
    (out_dir / "summary.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(agg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
