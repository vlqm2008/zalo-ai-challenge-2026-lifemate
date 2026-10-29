#!/usr/bin/env python3
"""Fine-tune a Whisper-family ASR on the TRAIN split's clips (train_ids only; the dev subset is held out).

  python scripts/finetune_asr.py --data ./data --split train --train-ids splits/train_ids.txt --dev-ids splits/dev_ids.txt \
      --model vinai/PhoWhisper-medium --out models/phowhisper_medium_ft --epochs 3 --freeze-encoder

Decoder-only by default (encoder frozen) so it fits next to the LLM on a 24 GB Apple-Silicon box; --unfreeze-last-k
unfreezes the last k encoder layers. Deterministic (seed 0). Reference text = audio_text as given (cased, punctuated).
Evaluates WER / key-token WER / money exact rate on the dev subset (same metric as precompute_asr.py).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from asr_decode_ablation import ed, norm, key  # noqa: E402
from lifemate.data import load_jsonl, transcript_targets  # noqa: E402
from lifemate.normalize_vi import extract_money  # noqa: E402
from lifemate.runtime import locate_split  # noqa: E402


def load_audio(path):
    import numpy as np
    import soundfile as sf
    a, sr = sf.read(str(path), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != 16000:
        import librosa
        a = librosa.resample(a, orig_sr=sr, target_sr=16000)
    return np.asarray(a, dtype="float32")


def evaluate(model, processor, clips, device, dtype, max_n=None, prompt_ids=None):
    import torch
    model.eval()
    pairs = []
    for rel_path, ref in (clips if max_n is None else clips[:max_n]):
        feats = processor(load_audio(rel_path), sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype=dtype)
        with torch.no_grad():
            ids = model.generate(feats, language="vi", task="transcribe", num_beams=1, do_sample=False, max_new_tokens=220)
        hyp = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        pairs.append((ref, unicodedata.normalize("NFC", hyp)))
    e = w = ke = kw = mh = mt = 0
    for ref, hyp in pairs:
        r, h = norm(ref), norm(hyp)
        e += ed(r, h); w += len(r)
        rk, hk = [t for t in r if key(t)], [t for t in h if key(t)]
        ke += ed(rk, hk); kw += len(rk)
        mr = [m.int_value for m in extract_money(ref)]
        if mr:
            mt += 1; mh += int(mr == [m.int_value for m in extract_money(hyp)])
    model.train()
    return {"WER": round(e / max(1, w), 4), "key_token_WER": round(ke / max(1, kw), 4), "money_exact_rate": round(mh / max(1, mt), 3), "n": len(pairs)}, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--train-ids", default="splits/train_ids.txt")
    ap.add_argument("--dev-ids", default="splits/dev_ids.txt")
    ap.add_argument("--model", default="vinai/PhoWhisper-medium")
    ap.add_argument("--out", default="models/phowhisper_medium_ft")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--unfreeze-last-k", type=int, default=0, help="unfreeze the last k encoder layers (with --freeze-encoder)")
    ap.add_argument("--eval-every", type=int, default=0, help="evaluate on --eval-n dev clips every N steps (0 = per epoch)")
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--decoder-last-k", type=int, default=0, help="train only the last k decoder layers (+ final layer norm); embeddings frozen (memory)")
    ap.add_argument("--cache-encoder", default=None, help="dir: precompute the frozen encoder's outputs (fp16) once and train the decoder from them (needs --freeze-encoder, no --unfreeze-last-k)")
    ap.add_argument("--in-domain-repeat", type=int, default=1, help="repeat the competition training clips N times (keeps the domain dominant next to large external sets)")
    ap.add_argument("--extra-clips", default=None, help="TSV <wav path>\t<text> of external clips (e.g. ViMD) appended to the training clips")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (0 = full epochs)")
    ap.add_argument("--bf16", action="store_true", help="CUDA: bf16 autocast for the forward/backward (fp32 master weights)")
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, logging as hf_logging
    hf_logging.set_verbosity_error()
    torch.backends.cudnn.enabled = False  # Colab after a vLLM install: CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH on Whisper's convs
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    root, split, split_dir = locate_split(args.data, args.split)
    train_ids = {l.strip() for l in Path(args.train_ids).read_text().splitlines() if l.strip()}
    dev_ids = {l.strip() for l in Path(args.dev_ids).read_text().splitlines() if l.strip()}
    assert not (train_ids & dev_ids), "train/dev overlap"
    train_clips, dev_clips = [], []
    for s in load_jsonl(split_dir / f"{split}.jsonl"):
        for rel, text in transcript_targets(s):
            if not text:
                continue
            (train_clips if s["id"] in train_ids else dev_clips if s["id"] in dev_ids else []).append((split_dir / rel, text))
    if args.limit:
        train_clips = train_clips[: args.limit]
    train_clips = train_clips * max(1, args.in_domain_repeat)
    if args.extra_clips:
        extra = [l.split("\t", 1) for l in Path(args.extra_clips).read_text(encoding="utf-8").splitlines() if "\t" in l]
        train_clips += [(Path(a), t.strip()) for a, t in extra if t.strip()]
        print(f"[ft] + {len(extra)} external clips from {args.extra_clips}", flush=True)
    print(f"[ft] train clips {len(train_clips)} (train_ids), dev clips {len(dev_clips)} (held out)", flush=True)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32
    processor = WhisperProcessor.from_pretrained(args.model)
    tok = processor.tokenizer
    tok.set_prefix_tokens(language="vi", task="transcribe", predict_timestamps=False)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=dtype).to(device)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    if args.freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False
        if args.unfreeze_last_k:
            for layer in model.model.encoder.layers[-args.unfreeze_last_k:]:
                for p in layer.parameters():
                    p.requires_grad = True
    if args.decoder_last_k:
        for p in model.model.decoder.parameters():
            p.requires_grad = False
        for layer in model.model.decoder.layers[-args.decoder_last_k:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in model.model.decoder.layer_norm.parameters():
            p.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ft] {args.model} on {device}, trainable params {n_train / 1e6:.0f} M", flush=True)

    base_metrics, _ = evaluate(model, processor, dev_clips, device, dtype, max_n=args.eval_n)
    print(f"[ft] before: dev[:{args.eval_n}] {base_metrics}", flush=True)

    enc_cache = None
    if args.cache_encoder:
        assert args.freeze_encoder and not args.unfreeze_last_k, "--cache-encoder needs a fully frozen encoder"
        enc_cache = Path(args.cache_encoder)
        enc_cache.mkdir(parents=True, exist_ok=True)
        model.model.encoder.to(torch.float16)
        t1 = time.time()
        with torch.no_grad():
            for i in range(0, len(train_clips), 8):
                chunk = train_clips[i: i + 8]
                todo = [(j, p) for j, (p, _) in enumerate(chunk) if not (enc_cache / f"{i + j}.npy").exists()]
                if not todo:
                    continue
                feats = processor([load_audio(p) for _, p in todo], sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype=torch.float16)
                enc = model.model.encoder(feats).last_hidden_state.to(torch.float16).cpu().numpy()
                for (j, _), e in zip(todo, enc):
                    np.save(enc_cache / f"{i + j}.npy", e)
                if (i // 8) % 50 == 0:
                    print(f"[ft] encoder cache {i + len(chunk)}/{len(train_clips)} {(time.time() - t1) / max(1, i + len(chunk)):.2f}s/clip", flush=True)
        model.model.encoder.to(torch.float32)
        print(f"[ft] encoder outputs cached in {enc_cache} ({(time.time() - t1) / 60:.1f} min)", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    steps_per_epoch = (len(train_clips) + args.batch - 1) // args.batch
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, (st + 1) / 50) * max(0.05, 1 - st / max(1, total)))
    model.train()
    step = 0
    t0 = time.time()
    best = None
    for epoch in range(args.epochs):
        order = list(range(len(train_clips)))
        random.Random(args.seed + epoch).shuffle(order)
        run_loss = 0.0
        for bi in range(0, len(order), args.batch):
            idxs = order[bi: bi + args.batch]
            batch = [train_clips[i] for i in idxs]
            lab = tok([t for _, t in batch], padding=True, return_tensors="pt")
            labels = lab.input_ids.masked_fill(lab.attention_mask == 0, -100).to(device)
            if (labels[:, 0] == tok.bos_token_id).all():      # the model prepends decoder_start itself
                labels = labels[:, 1:]
            ac = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (args.bf16 and device == "cuda") else torch.autocast(device_type="cpu", enabled=False)
            with ac:
                if enc_cache is not None:
                    enc = torch.from_numpy(np.stack([np.load(enc_cache / f"{i}.npy") for i in idxs])).to(device, dtype=dtype)
                    out = model(encoder_outputs=(enc,), labels=labels)
                else:
                    feats = processor([load_audio(p) for p, _ in batch], sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype=dtype)
                    out = model(input_features=feats, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            run_loss += float(out.loss)
            if step % 25 == 0:
                print(f"[ft] epoch {epoch + 1} step {step}/{total} loss {run_loss / 25:.3f} {(time.time() - t0) / step:.2f}s/step", flush=True)
                run_loss = 0.0
            if args.eval_every and step % args.eval_every == 0:
                m, _ = evaluate(model, processor, dev_clips, device, dtype, max_n=args.eval_n)
                print(f"[ft] step {step}: dev[:{args.eval_n}] {m}", flush=True)
                if best is None or m["WER"] < best["WER"]:
                    best = m
                    Path(args.out).mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(args.out)
                    processor.save_pretrained(args.out)
                    print(f"[ft] saved {args.out} (step {step})", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
        m, _ = evaluate(model, processor, dev_clips, device, dtype, max_n=args.eval_n)
        print(f"[ft] epoch {epoch + 1} done: dev[:{args.eval_n}] {m}", flush=True)
        if best is None or m["WER"] < best["WER"]:
            best = m
            Path(args.out).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.out)
            processor.save_pretrained(args.out)
            print(f"[ft] saved {args.out}", flush=True)
    model = WhisperForConditionalGeneration.from_pretrained(args.out, dtype=dtype).to(device)
    full, pairs = evaluate(model, processor, dev_clips, device, dtype)
    print(f"[ft] FINAL full dev ({full['n']} clips): {full}  (baseline on first {args.eval_n}: {base_metrics})", flush=True)
    Path(args.out, "dev_metrics.json").write_text(json.dumps({"final_dev": full, "baseline_head": base_metrics, "args": vars(args)}, ensure_ascii=False, indent=2))
    Path(args.out, "dev_pairs.jsonl").write_text("\n".join(json.dumps({"ref": r, "hyp": h}, ensure_ascii=False) for r, h in pairs), encoding="utf-8")


if __name__ == "__main__":
    main()
