"""ASR backends: HF Whisper-family (PhoWhisper / Whisper), Fake (tests / oracle text)."""
from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Optional

from .config import Config


# General vocabulary of the apps in the catalog (channels, fuel, gold, wallet, chat...). Measured on train only
# (scripts/asr_decode_ablation.py): WER 0.499 -> 0.450 on 60 clips vs no prompt. Not tied to any sample.
DOMAIN_PROMPT = ("Zalo, SMS, ví điện tử, tài khoản ngân hàng, báo thức, nhắc việc, lịch, điều hòa, quạt, tivi kênh VTV1 VTV3 HTV7 SCTV9, "
                 "xăng RON95 E5, dầu diesel, vàng SJC, PNJ, nhẫn, đô la, yên Nhật, euro, Grab, xe máy, ô tô bốn chỗ, bảy chỗ.")


def resolve_prompt(cfg) -> Optional[str]:
    p = getattr(cfg, "asr_prompt", None)
    if not p:
        return None
    return DOMAIN_PROMPT if p == "domain" else p


class BaseASR:
    def transcribe(self, path: str | Path) -> str:
        raise NotImplementedError


class FakeASR(BaseASR):
    """Returns a mapping value, else the sidecar <audio>.txt, else ''."""

    def __init__(self, mapping: Optional[dict] = None):
        self.mapping = mapping or {}

    def transcribe(self, path):
        p = str(path)
        if p in self.mapping:
            return self.mapping[p]
        side = Path(p).with_suffix(".txt")
        if side.exists():
            return side.read_text(encoding="utf-8").strip()
        return ""


def _pick_device() -> str:
    """cuda (A30 / T4 / Blackwell) -> mps (Apple Silicon dev box) -> cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFWhisperASR(BaseASR):
    def __init__(self, cfg: Config):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor, logging as hf_logging

        torch.backends.cudnn.enabled = False  # Colab after a vLLM install: CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH on Whisper's convs

        hf_logging.set_verbosity_error()   # the per-call max_new_tokens/max_length notice floods the logs
        self.cfg = cfg
        self.device = _pick_device()
        want32 = str(cfg.asr_compute_type).startswith("float32")
        dtype = torch.float16 if self.device in ("cuda", "mps") and not want32 else torch.float32
        self.processor = WhisperProcessor.from_pretrained(cfg.asr_model)
        self.model = WhisperForConditionalGeneration.from_pretrained(cfg.asr_model, dtype=dtype).to(self.device)
        self.model.eval()
        self.dtype = dtype
        self._cache: dict[str, str] = {}
        self.prompt = resolve_prompt(cfg)
        self.prompt_ids = self.processor.get_prompt_ids(self.prompt, return_tensors="pt").to(self.device) if self.prompt else None
        print(f"[asr] loaded {cfg.asr_model} on {self.device} ({dtype}) prompt={'yes' if self.prompt else 'no'}", flush=True)

    def _load_audio(self, path: str):
        import numpy as np
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        return np.asarray(audio, dtype="float32")

    def transcribe(self, path):
        import torch

        p = str(path)
        if p in self._cache:
            return self._cache[p]
        audio = self._load_audio(p)
        chunk = 16000 * self.cfg.asr_chunk_length_s
        texts = []
        for start in range(0, max(1, len(audio)), chunk):
            seg = audio[start:start + chunk]
            feats = self.processor(seg, sampling_rate=16000, return_tensors="pt").input_features
            feats = feats.to(self.device, dtype=self.dtype)
            kw = dict(language="vi", task="transcribe", num_beams=self.cfg.asr_num_beams, do_sample=False, max_new_tokens=220)
            if self.prompt_ids is not None:
                kw["prompt_ids"] = self.prompt_ids
            with torch.no_grad():
                ids = self.model.generate(feats, **kw)
            t = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            if self.prompt and t.startswith(self.prompt):
                t = t[len(self.prompt):].strip()
            texts.append(t)
        text = unicodedata.normalize("NFC", " ".join(t for t in texts if t)).strip()
        self._cache[p] = text
        return text


class CT2WhisperASR(BaseASR):
    """faster-whisper (CTranslate2) backend: 3-5x faster than transformers, deterministic with
    beam_size=1 / temperature=0. Same recipe as the AIC 2026 pipeline (PhoWhisper -> CT2)."""

    def __init__(self, cfg: Config, device: str = "cuda", device_index: int = 0):
        from faster_whisper import WhisperModel

        self.cfg = cfg
        ct2_dir = cfg.asr_ct2_dir or ensure_ct2_model(cfg.asr_model, cfg.asr_compute_type)
        self.model = WhisperModel(ct2_dir, device=device, device_index=device_index,
                                  compute_type=cfg.asr_compute_type if device == "cuda" else "int8")
        self._cache: dict[str, str] = {}
        self.prompt = resolve_prompt(cfg)
        print(f"[asr] faster-whisper loaded {ct2_dir} on {device}:{device_index} ({cfg.asr_compute_type}) prompt={'yes' if self.prompt else 'no'}", flush=True)

    def transcribe(self, path):
        p = str(path)
        if p in self._cache:
            return self._cache[p]
        segments, _info = self.model.transcribe(
            p, language="vi", task="transcribe", beam_size=max(1, self.cfg.asr_num_beams), best_of=1,
            temperature=0.0, condition_on_previous_text=False, vad_filter=False, without_timestamps=True,
            initial_prompt=self.prompt,
        )
        text = unicodedata.normalize("NFC", " ".join(s.text.strip() for s in segments)).strip()
        self._cache[p] = text
        return text


def ensure_ct2_model(hf_model: str, compute_type: str = "float16", out_dir: Optional[str] = None) -> str:
    """Convert an HF Whisper checkpoint to CTranslate2 once (needs `ctranslate2` + `transformers`)."""
    import subprocess
    import sys

    out = Path(out_dir or f"models/{hf_model.replace('/', '_')}_ct2_{compute_type}")
    if (out / "model.bin").exists():
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    quant = "float16" if compute_type.startswith("float16") else "int8_float16"
    cmd = [sys.executable, "-m", "ctranslate2.converters.transformers", "--model", hf_model,
           "--output_dir", str(out), "--quantization", quant, "--copy_files", "tokenizer.json", "preprocessor_config.json"]
    print("[asr] converting to CT2:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    return str(out)


class CachedASR(BaseASR):
    """Transcripts precomputed by scripts/precompute_asr.py: JSON {relative or absolute audio path: text}."""

    def __init__(self, cache_path: str):
        import json

        raw = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        self.mapping = {str(k): v for k, v in raw.items()}
        self.by_name = {Path(k).name: v for k, v in raw.items()}
        self.misses: list[str] = []

    def transcribe(self, path):
        p = str(path)
        if p in self.mapping:
            return self.mapping[p]
        for k, v in self.mapping.items():
            if p.endswith(k) or k.endswith(p):
                return v
        if Path(p).name in self.by_name:
            return self.by_name[Path(p).name]
        self.misses.append(p)
        return ""


def build_asr(cfg: Config, fake_mapping: Optional[dict] = None) -> BaseASR:
    if cfg.asr_backend == "fake":
        return FakeASR(fake_mapping)
    if cfg.asr_backend == "cache":
        if not cfg.asr_cache_path:
            raise ValueError("asr_backend=cache needs asr_cache_path")
        return CachedASR(cfg.asr_cache_path)
    if cfg.asr_backend == "ct2":
        return CT2WhisperASR(cfg)
    return HFWhisperASR(cfg)
