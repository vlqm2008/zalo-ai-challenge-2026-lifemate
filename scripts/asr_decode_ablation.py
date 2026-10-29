#!/usr/bin/env python3
"""Decode-setting ablation for the HF Whisper backend on a small train subset (worst + random clips):
greedy vs beam search, with/without a general domain-vocabulary prompt (prompt_ids). Deterministic settings only.

  python scripts/asr_decode_ablation.py --data ./data --split train --worst out/asr_cache_train_dev_ids_worst.jsonl --n-worst 30 --n-random 30
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifemate.asr import HFWhisperASR  # noqa: E402
from lifemate.config import Config  # noqa: E402
from lifemate.data import load_jsonl, transcript_targets  # noqa: E402
from lifemate.normalize_vi import DIGIT_WORDS, extract_money  # noqa: E402
from lifemate.runtime import locate_split  # noqa: E402

# general domain vocabulary (apps / entities of the catalog), not tied to any sample
DOMAIN_PROMPT = ("Zalo, SMS, ví điện tử, tài khoản ngân hàng, báo thức, nhắc việc, lịch, điều hòa, quạt, tivi kênh VTV1 VTV3 HTV7 SCTV9, "
                 "xăng RON95 E5, dầu diesel, vàng SJC, PNJ, nhẫn, đô la, yên Nhật, euro, Grab, xe máy, ô tô bốn chỗ, bảy chỗ.")


def norm(s):
    return re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFC", s or "").lower()).split()


def ed(a, b):
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(b) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[len(b)]


def key(t):
    return t.isdigit() or t in DIGIT_WORDS or (t.isascii() and t.isalpha())


def score(pairs):
    e = w = ke = kw = mh = mt = 0
    for ref, hyp in pairs:
        r, h = norm(ref), norm(hyp)
        e += ed(r, h); w += len(r)
        rk, hk = [t for t in r if key(t)], [t for t in h if key(t)]
        ke += ed(rk, hk); kw += len(rk)
        mr = [m.int_value for m in extract_money(ref)]
        if mr:
            mt += 1; mh += int(mr == [m.int_value for m in extract_money(hyp)])
    return {"WER": round(e / max(1, w), 4), "key_WER": round(ke / max(1, kw), 4), "money": f"{mh}/{mt}"}


class Variant(HFWhisperASR):
    def __init__(self, cfg, prompt=None):
        super().__init__(cfg)
        self.prompt_ids = self.processor.get_prompt_ids(prompt, return_tensors="pt").to(self.device) if prompt else None
        self.prompt = prompt

    def transcribe(self, path):
        import torch
        audio = self._load_audio(str(path))
        feats = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(self.device, dtype=self.dtype)
        kw = dict(language="vi", task="transcribe", num_beams=self.cfg.asr_num_beams, do_sample=False, max_new_tokens=220)
        if self.prompt_ids is not None:
            kw["prompt_ids"] = self.prompt_ids
        with torch.no_grad():
            ids = self.model.generate(feats, **kw)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if self.prompt and text.startswith(self.prompt):
            text = text[len(self.prompt):].strip()
        return unicodedata.normalize("NFC", text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="vinai/PhoWhisper-medium")
    ap.add_argument("--worst", required=True)
    ap.add_argument("--n-worst", type=int, default=30)
    ap.add_argument("--n-random", type=int, default=30)
    ap.add_argument("--ids-file", default="splits/dev_ids.txt")
    args = ap.parse_args()
    root, split, split_dir = locate_split(args.data, args.split)
    worst = [json.loads(l) for l in Path(args.worst).read_text(encoding="utf-8").splitlines() if l.strip()][: args.n_worst]
    clips = [(d["audio"], d["ref"]) for d in worst]
    keep = {l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()}
    pool = [(rel, o) for s in load_jsonl(split_dir / f"{split}.jsonl") if s["id"] in keep for rel, o in transcript_targets(s) if o]
    rnd = random.Random(0)
    rnd.shuffle(pool)
    seen = {c[0] for c in clips}
    clips += [p for p in pool if p[0] not in seen][: args.n_random]
    print(f"{len(clips)} clips ({len(worst)} worst + {len(clips) - len(worst)} random)")
    variants = [("greedy", 1, None), ("beam5", 5, None), ("greedy+prompt", 1, DOMAIN_PROMPT), ("beam5+prompt", 5, DOMAIN_PROMPT)]
    results = {}
    for name, beams, prompt in variants:
        asr = Variant(Config(asr_model=args.model, asr_num_beams=beams), prompt)
        t0 = time.time()
        pairs = [(ref, asr.transcribe(split_dir / rel)) for rel, ref in clips]
        dt = (time.time() - t0) / len(clips)
        sc = score(pairs)
        sc_w = score(pairs[: len(worst)])
        results[name] = {"all": sc, "worst": sc_w, "s_per_clip": round(dt, 2)}
        print(f"{name:<14} all={sc} worst={sc_w} {dt:.2f}s/clip", flush=True)
        if name == "beam5":
            for (ref, hyp) in pairs[:8]:
                print("    ref:", ref[:90], "\n    hyp:", hyp[:90])
    Path("out/asr_decode_ablation.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
