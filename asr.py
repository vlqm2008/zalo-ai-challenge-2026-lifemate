"""Transcribe every user turn of a split (the final request and the user turns of the history) with a Whisper-family model.

    python asr.py --split train --ids splits/dev_seen.txt --model vinai/PhoWhisper-large --out runs/asr_pho_dev_seen.json
    python asr.py --split public_test --model vinai/PhoWhisper-large --out runs/asr_pho_public_test.json

Writes {"audio/<id>/<id>.wav": text, ...}; agent.py --asr uses it in place of audio_text.
On train it also prints the word error rate against audio_text.
"""
import argparse
import json
import re
import time
import unicodedata
import wave
from pathlib import Path

import numpy as np

from agent import ROOT


def load_wav(path):
    with wave.open(str(path)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), f"{path}: expected 16 kHz mono 16-bit"
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def words(s):
    return re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFC", s.lower())).split()


def edits(ref, hyp):
    """(word edit distance, number of reference words), after lowercasing and dropping punctuation."""
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)], len(r)


def user_turns(items):
    return [t for d in items for t in [*(h for h in d["history"] if h["role"] == "user"), d]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--ids", help="file with one sample id per line (default: the whole split)")
    ap.add_argument("--model", default="vinai/PhoWhisper-large")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    import torch
    from transformers import pipeline

    # Colab mixes cuDNN libraries once vLLM swaps torch (CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH on Whisper's convs);
    # the two input convs are tiny, so plain CUDA kernels cost nothing
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.enable_cudnn_sdp(False)

    items = [json.loads(l) for l in open(ROOT / a.split / f"{a.split}.jsonl", encoding="utf-8")]
    if a.ids:
        keep = set(open(a.ids, encoding="utf-8").read().split())
        items = [d for d in items if d["id"] in keep]
    turns = user_turns(items)
    cuda = torch.cuda.is_available()
    asr = pipeline("automatic-speech-recognition", model=a.model, dtype=torch.float16 if cuda else torch.float32, device=0 if cuda else "cpu")
    lang = {"language": "vi", "task": "transcribe"} if getattr(asr.model.generation_config, "lang_to_id", None) else {}
    t0 = time.time()
    clips = ({"raw": load_wav(ROOT / a.split / t["audio"]), "sampling_rate": 16000} for t in turns)
    outs = asr(clips, batch_size=a.batch, generate_kwargs={**lang, "num_beams": 1, "do_sample": False})  # greedy: deterministic
    texts = {t["audio"]: o["text"].strip() for t, o in zip(turns, outs)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=0)
    print(f"{len(texts)} clips in {time.time() - t0:.0f}s -> {a.out}")
    if all("audio_text" in t for t in turns):
        e = [edits(t["audio_text"], texts[t["audio"]]) for t in turns]
        print(f"WER {100 * sum(x for x, _ in e) / sum(n for _, n in e):.2f}% ({a.model})")


if __name__ == "__main__":
    main()
