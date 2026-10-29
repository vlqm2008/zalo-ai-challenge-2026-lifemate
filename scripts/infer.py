#!/usr/bin/env python3
"""Run the agent on a split and write submission.jsonl (one line per sample, written as soon
as each sample finishes).

Organisers' run:   python scripts/infer.py --data /data --out /result/submission.jsonl
Dev (train, oracle transcripts, 50 samples):
                   python scripts/infer.py --data ./data --split train --oracle --limit 50 --out out/dev.jsonl

Hard rules respected here: exactly ONE ToolEnv per sample, every call made on it is in the
submission (get_tool_calls()), zaic_tools imported from the split folder, no file under
<split>_environment/ is ever opened by this code.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifemate.agent import Agent  # noqa: E402
from lifemate.config import Config  # noqa: E402
from lifemate.data import load_catalog, load_jsonl  # noqa: E402
from lifemate.runtime import SubmissionWriter, build_components, import_toolenv, locate_split  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data", help="split folder, or the folder that holds the split folders")
    ap.add_argument("--split", default=None, help="split name if --data holds several (train/public_test/...)")
    ap.add_argument("--out", default="/result/submission.jsonl")
    ap.add_argument("--config", default=None, help="JSON config overriding lifemate.config.Config defaults")
    ap.add_argument("--log-dir", default=None, help="per-sample JSON logs (default: <out dir>/logs)")
    ap.add_argument("--oracle", action="store_true", help="DEV ONLY: use audio_text instead of ASR")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids-file", default=None, help="only these sample ids (one per line)")
    ap.add_argument("--no-resume", action="store_true", help="do not skip ids already in --out")
    ap.add_argument("--llm-backend", default=None, choices=["vllm", "hf", "fake"])
    ap.add_argument("--asr-backend", default=None, choices=["hf", "ct2", "cache", "fake"])
    ap.add_argument("--asr-cache", default=None, help="JSON from precompute_asr.py (implies --asr-backend cache)")
    ap.add_argument("--mock-env", action="store_true", help="DEV ONLY: use lifemate.mock_env instead of zaic_tools")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = Config.load(args.config, {"llm_backend": args.llm_backend, "asr_backend": "cache" if args.asr_cache else args.asr_backend,
                                    "asr_cache_path": args.asr_cache})
    root, split, split_dir = locate_split(args.data, args.split)
    print(f"[infer] root={root} split={split} dir={split_dir}", flush=True)
    print(f"[infer] config:\n{cfg.dump()}", flush=True)

    catalog = load_catalog(split_dir / f"{split}_tools_catalog.json")
    samples = list(load_jsonl(split_dir / f"{split}.jsonl"))
    if args.ids_file:
        keep = {l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()}
        samples = [s for s in samples if s["id"] in keep]
    if args.limit:
        samples = samples[: args.limit]

    if args.mock_env:
        from lifemate.mock_env import MockToolEnv as ToolEnv
    else:
        ToolEnv = import_toolenv(split_dir)

    out_path = Path(args.out)
    log_dir = Path(args.log_dir) if args.log_dir else out_path.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SubmissionWriter(out_path, resume=not args.no_resume)

    t_load = time.monotonic()
    llm, asr = build_components(cfg, oracle_only=args.oracle)
    agent = Agent(cfg, llm, asr, catalog)
    print(f"[infer] models loaded in {time.monotonic() - t_load:.1f}s; {len(samples)} samples", flush=True)

    times = []
    for i, sample in enumerate(samples):
        sid = sample["id"]
        if sid in writer.done:
            continue
        t0 = time.monotonic()
        env = None
        log = {"id": sid}
        try:
            env = ToolEnv(split, sid, root=str(root))        # exactly one env per sample
            log = agent.run(sample, env, split_dir, oracle=args.oracle)
            tool_calls, state_hash = log["tool_calls"], log["state_hash"]
        except Exception as e:  # never lose the line: write whatever the env has
            log["error"] = f"{type(e).__name__}: {e}"
            log["traceback"] = traceback.format_exc()
            tool_calls = env.get_tool_calls() if env is not None else []
            try:
                state_hash = env.get_state_hash() if env is not None else ""
            except Exception:
                state_hash = ""
        writer.write(sid, tool_calls, state_hash)
        dt = time.monotonic() - t0
        times.append(dt)
        log["wall_s"] = round(dt, 3)
        (log_dir / f"{sid}.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
        if (i + 1) % 10 == 0 or dt > cfg.sample_budget_s:
            srt = sorted(times)
            p95 = srt[int(0.95 * (len(srt) - 1))]
            print(f"[infer] {i + 1}/{len(samples)} last={dt:.1f}s p50={srt[len(srt) // 2]:.1f}s p95={p95:.1f}s "
                  f"(A30 est ×2: p95≈{2 * p95:.1f}s) calls={len(tool_calls)}", flush=True)
    writer.close()
    if times:
        srt = sorted(times)
        print(f"[infer] done. n={len(times)} mean={sum(times) / len(times):.2f}s p95={srt[int(0.95 * (len(srt) - 1))]:.2f}s "
              f"max={srt[-1]:.2f}s", flush=True)


if __name__ == "__main__":
    main()
