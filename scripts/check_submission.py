#!/usr/bin/env python3
"""Verify a submission.jsonl against a split (Phase 4 acceptance):

  python scripts/check_submission.py --data ./data --split public_test --sub out/public/submission.jsonl [--logs out/public/logs]

Checks: valid JSON per line; exactly one line per id of <split>.jsonl and no extra ids; every line has
`tool_calls` (list of {name, arguments}) and a 64-hex `state_hash`; the sample_submission ids are covered;
the organisers' schema validator (zaic_tools --validate) reports 0 errors. Prints #calls distribution,
% empty, % ask, tool frequency, and the latency summary from the per-sample logs when present.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifemate.data import ASK_TOOL_NAME, load_jsonl  # noqa: E402
from lifemate.runtime import locate_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--logs", default=None, help="per-sample logs dir (default: <sub dir>/logs)")
    args = ap.parse_args()
    root, split, split_dir = locate_split(args.data, args.split)
    ids = [s["id"] for s in load_jsonl(split_dir / f"{split}.jsonl")]
    problems: list[str] = []
    rows: list[dict] = []
    for i, line in enumerate(Path(args.sub).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            problems.append(f"line {i}: empty line")
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"line {i}: invalid JSON ({e})")
            continue
        if not isinstance(r.get("id"), str):
            problems.append(f"line {i}: missing id")
        tc = r.get("tool_calls")
        if not isinstance(tc, list) or not all(isinstance(c, dict) and isinstance(c.get("name"), str) and isinstance(c.get("arguments"), dict) for c in tc):
            problems.append(f"line {i} ({r.get('id')}): tool_calls must be a list of {{name, arguments}}")
        if not (isinstance(r.get("state_hash"), str) and re.fullmatch(r"[0-9a-f]{64}", r["state_hash"])):
            problems.append(f"line {i} ({r.get('id')}): state_hash is not a sha256 hex")
        rows.append(r)
    sub_ids = Counter(r.get("id") for r in rows)
    dup = [k for k, v in sub_ids.items() if v > 1]
    missing = [k for k in ids if k not in sub_ids]
    extra = [k for k in sub_ids if k not in set(ids)]
    if dup:
        problems.append(f"duplicate ids: {dup[:10]} ({len(dup)})")
    if missing:
        problems.append(f"missing ids: {missing[:10]} ({len(missing)})")
    if extra:
        problems.append(f"ids not in {split}: {extra[:10]} ({len(extra)})")
    ss = split_dir / "sample_submission.jsonl"
    if ss.exists():
        ss_ids = [json.loads(l)["id"] for l in ss.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not all(k in sub_ids for k in ss_ids):
            problems.append(f"sample_submission ids not all covered: {[k for k in ss_ids if k not in sub_ids]}")
        print(f"sample_submission.jsonl: {len(ss_ids)} example ids, all covered = {all(k in sub_ids for k in ss_ids)}")

    n = len(rows)
    ncalls = Counter(len(r.get("tool_calls") or []) for r in rows)
    empty = sum(1 for r in rows if not r.get("tool_calls"))
    asks = sum(1 for r in rows if any(c.get("name") == ASK_TOOL_NAME for c in (r.get("tool_calls") or [])))
    print(f"lines={n} ids_in_split={len(ids)} unique_ids={len(sub_ids)}")
    print(f"#calls distribution: {dict(sorted(ncalls.items()))}")
    print(f"empty: {empty} ({100 * empty / max(1, n):.1f} %)   ask_missing_info: {asks} ({100 * asks / max(1, n):.1f} %)")
    freq = Counter(c.get("name") for r in rows for c in (r.get("tool_calls") or []))
    print("top tools:", freq.most_common(15))

    # organisers' validator
    try:
        out = subprocess.run([sys.executable, "-m", "zaic_tools", "--split", split, "--root", str(root), "--validate", str(Path(args.sub).resolve())],
                             cwd=str(root), env={"PYTHONPATH": str(split_dir), "PATH": "/usr/bin:/bin"}, capture_output=True, text=True, timeout=1800)
        tail = (out.stdout.strip().splitlines() or [""])[-1]
        print("zaic_tools --validate:", tail, ("" if out.returncode == 0 else f"(rc={out.returncode} {out.stderr[-300:]})"))
        m = re.search(r"(\d+) schema errors", tail)
        if m and int(m.group(1)) > 0:
            problems.append(f"schema errors: {m.group(1)} (see zaic_tools --validate output above)")
            print(out.stdout[-2000:])
    except Exception as e:
        problems.append(f"could not run zaic_tools --validate: {e}")

    logs = Path(args.logs) if args.logs else Path(args.sub).parent / "logs"
    if logs.is_dir():
        times, steps, reasons = [], [], Counter()
        for p in logs.glob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if d.get("wall_s") is not None:
                times.append(float(d["wall_s"]))
            steps.append(len(d.get("steps") or []))
            reasons[d.get("finished_reason") or d.get("error", "?")[:30]] += 1
        if times:
            srt = sorted(times)
            q = lambda x: srt[min(len(srt) - 1, int(x * (len(srt) - 1)))]  # noqa: E731
            print(f"latency (this machine): n={len(srt)} mean={sum(srt) / len(srt):.1f}s p50={q(.5):.1f}s p95={q(.95):.1f}s max={srt[-1]:.1f}s; "
                  f"steps/sample mean={sum(steps) / max(1, len(steps)):.2f}; finished_reasons={dict(reasons.most_common())}")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("\nOK: submission passes all checks")


if __name__ == "__main__":
    main()
