#!/usr/bin/env python3
"""Transcribe EVERY audio file of a split once (latest request + history turns), in parallel on
all visible GPUs (same recipe as the AIC 2026 ASR pipeline: faster-whisper/CT2 PhoWhisper,
one model per GPU, JSON cache). The cache decouples ASR from agent iteration:

  python scripts/precompute_asr.py --data ./data --split train --backend ct2 --out out/asr_cache_train.json
  python scripts/run_dev.py ... --asr-backend cache --asr-cache out/asr_cache_train.json

With labels present it also reports WER / key-token WER / money exact rate (same as eval_asr).
Output JSON: {"<relative audio path>": "<transcript>", ...}; resumable.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import unicodedata
from pathlib import Path
from queue import Queue

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifemate.asr import CT2WhisperASR, HFWhisperASR  # noqa: E402
from lifemate.config import Config  # noqa: E402
from lifemate.data import load_jsonl, transcript_targets  # noqa: E402
from lifemate.normalize_vi import DIGIT_WORDS, extract_money  # noqa: E402
from lifemate.runtime import locate_split  # noqa: E402


def _norm(s: str) -> list[str]:
    s = unicodedata.normalize("NFC", s or "").lower()
    return re.sub(r"[^\w\s]", " ", s).split()


def _ed(a: list[str], b: list[str]) -> int:
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(b) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[len(b)]


def _key(t: str) -> bool:
    return t.isdigit() or t in DIGIT_WORDS or (t.isascii() and t.isalpha())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--backend", default="ct2", choices=["ct2", "hf"])
    ap.add_argument("--model", default="vinai/PhoWhisper-medium")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--ct2-dir", default=None)
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--prompt", default=None, help='Whisper initial prompt: "domain" (general app vocabulary) or text')
    ap.add_argument("--gpus", type=int, default=None, help="default: all visible GPUs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids-file", default=None, help="only these sample ids (e.g. splits/dev_ids.txt) - the cache is resumable, so run the dev subset first")
    ap.add_argument("--no-metrics", action="store_true")
    args = ap.parse_args()

    root, split, split_dir = locate_split(args.data, args.split)
    out = Path(args.out or f"out/asr_cache_{split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, str] = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}

    samples = list(load_jsonl(split_dir / f"{split}.jsonl"))
    if args.ids_file:
        keep = {l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()}
        samples = [s for s in samples if s["id"] in keep]
    if args.limit:
        samples = samples[: args.limit]
    refs: dict[str, str] = {}
    todo: list[str] = []
    for s in samples:
        for rel, oracle in transcript_targets(s):
            if oracle:
                refs[rel] = oracle
            if rel not in cache and rel not in todo:
                todo.append(rel)
    print(f"[asr] split={split} files={len(todo)} (cached {len(cache)})", flush=True)

    import torch
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_workers = max(1, args.gpus or n_gpu or 1)
    cfg = Config(asr_model=args.model, asr_num_beams=args.beams, asr_compute_type=args.compute_type, asr_ct2_dir=args.ct2_dir,
                 asr_prompt=args.prompt)
    if args.backend == "ct2" and not args.ct2_dir:
        from lifemate.asr import ensure_ct2_model
        cfg.asr_ct2_dir = ensure_ct2_model(args.model, args.compute_type)  # convert once, before the threads start

    q: Queue = Queue()
    for rel in todo:
        q.put(rel)
    lock = threading.Lock()
    t0 = time.monotonic()
    done = [0]

    def worker(idx: int):
        if args.backend == "ct2":
            asr = CT2WhisperASR(cfg, device="cuda" if n_gpu else "cpu", device_index=idx % max(1, n_gpu))
        else:
            asr = HFWhisperASR(cfg)
        while True:
            try:
                rel = q.get_nowait()
            except Exception:
                return
            try:
                text = asr.transcribe(split_dir / rel)
            except Exception as e:
                print(f"[asr] ERROR {rel}: {e}", flush=True)
                text = ""
            with lock:
                cache[rel] = text
                done[0] += 1
                if done[0] % 50 == 0:
                    out.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
                    rate = done[0] / (time.monotonic() - t0)
                    print(f"[asr] {done[0]}/{len(todo)} {rate:.1f} clips/s eta {((len(todo) - done[0]) / max(rate, 1e-6)) / 60:.1f} min", flush=True)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"[asr] wrote {len(cache)} transcripts to {out} in {(time.monotonic() - t0) / 60:.1f} min", flush=True)

    if refs and not args.no_metrics:
        errs = words = kerr = kw = mh = mt = 0
        worst = []
        for rel, ref in refs.items():
            hyp = cache.get(rel, "")
            r, h = _norm(ref), _norm(hyp)
            e = _ed(r, h)
            errs += e
            words += len(r)
            rk, hk = [t for t in r if _key(t)], [t for t in h if _key(t)]
            kerr += _ed(rk, hk)
            kw += len(rk)
            mr = [m.int_value for m in extract_money(ref)]
            if mr:
                mt += 1
                mh += int(mr == [m.int_value for m in extract_money(hyp)])
            worst.append((e / max(1, len(r)), rel, ref, hyp))
        metrics = {"model": args.model, "backend": args.backend, "beams": args.beams, "prompt": args.prompt, "n": len(refs), "WER": errs / max(1, words),
                   "key_token_WER": kerr / max(1, kw), "money_exact_rate": mh / max(1, mt), "money_n": mt}
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        worst.sort(reverse=True)
        tag = f"_{Path(args.ids_file).stem}" if args.ids_file else ""
        metrics["subset"] = Path(args.ids_file).stem if args.ids_file else "all"
        wp = out.with_name(out.stem + tag + "_worst.jsonl")
        wp.write_text("\n".join(json.dumps({"wer": w, "audio": a, "ref": r, "hyp": h}, ensure_ascii=False)
                                for w, a, r, h in worst[:200]), encoding="utf-8")
        (out.with_name(out.stem + tag + "_metrics.json")).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print("worst cases ->", wp)


if __name__ == "__main__":
    main()
