"""LLM backends: vLLM (production), HF transformers (fallback), Fake (tests)."""
from __future__ import annotations

import json
import os
import random
from typing import Callable, Optional

from .config import Config


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


class BaseLLM:
    def chat(self, messages: list[dict], json_schema: Optional[dict] = None) -> str:
        raise NotImplementedError


class FakeLLM(BaseLLM):
    """Scripted responses for plumbing tests. `script` is a function(messages) -> str or a list of str."""

    def __init__(self, script):
        self.script = script
        self.calls: list[list[dict]] = []
        self._i = 0

    def chat(self, messages, json_schema=None):
        self.calls.append(messages)
        if callable(self.script):
            return self.script(messages)
        out = self.script[min(self._i, len(self.script) - 1)]
        self._i += 1
        return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)


class VLLMBackend(BaseLLM):
    def __init__(self, cfg: Config):
        import torch
        from vllm import LLM

        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        util = min(0.95, max(0.2, cfg.vllm_memory_gb / total_gb))
        self.cfg = cfg
        self.llm = LLM(
            model=cfg.llm_model,
            dtype=cfg.llm_dtype,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=util,
            enable_prefix_caching=True,
            max_num_seqs=8,            # batch 1 at inference; hybrid (Mamba-style) models like Qwen3.5 need this <= cache blocks
            seed=cfg.seed,
            trust_remote_code=False,
        )
        self.tokenizer = self.llm.get_tokenizer()
        print(f"[llm] vLLM loaded {cfg.llm_model} util={util:.2f} of {total_gb:.1f} GB", flush=True)

    def _sampling(self, schema: Optional[dict]):
        from vllm import SamplingParams

        kwargs = dict(temperature=0.0, max_tokens=self.cfg.max_tokens, seed=self.cfg.seed)
        if schema is not None and self.cfg.use_guided_json:
            try:  # vLLM >= 0.10
                from vllm.sampling_params import StructuredOutputsParams
                kwargs["structured_outputs"] = StructuredOutputsParams(json=schema)
            except ImportError:  # older vLLM
                from vllm.sampling_params import GuidedDecodingParams
                kwargs["guided_decoding"] = GuidedDecodingParams(json=schema)
        return SamplingParams(**kwargs)

    def chat(self, messages, json_schema=None):
        outs = self.llm.chat(
            messages,
            sampling_params=self._sampling(json_schema),
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": self.cfg.enable_thinking},
        )
        return outs[0].outputs[0].text

    def count_tokens(self, messages) -> int:
        try:
            ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            if hasattr(ids, "keys"):          # BatchEncoding in newer transformers
                ids = ids["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            return len(ids)
        except Exception:
            return sum(len(m.get("content", "")) for m in messages) // 3


class HFBackend(BaseLLM):
    """Plain transformers generation (greedy). No grammar enforcement; slower than vLLM.

    Prefix KV-cache reuse across the steps of one sample (like vLLM's prefix caching): the chat grows by
    appending, so the cache is cropped to the longest common token prefix with the previous call and only
    the new tokens are prefilled. Disable with LIFEMATE_NO_PREFIX_CACHE=1 for an A/B check."""

    def __init__(self, cfg: Config):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model)
        dtype = torch.bfloat16 if cfg.llm_dtype == "bfloat16" else torch.float16
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"      # Apple Silicon dev box (no vLLM); same greedy decoding, no grammar
        else:
            device = "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(cfg.llm_model, dtype=dtype if device != "cpu" else None).to(device)
        self.model.eval()
        self.device = device
        self.prefix_cache = os.environ.get("LIFEMATE_NO_PREFIX_CACHE", "") != "1"
        self._cache = None
        self._cache_ids: list[int] = []
        self.last_prompt_tokens = 0
        self.last_new_tokens = 0
        print(f"[llm] HF backend {cfg.llm_model} on {device} ({dtype if device != 'cpu' else 'fp32'}) prefix_cache={self.prefix_cache}", flush=True)

    def _encode(self, messages) -> list[int]:
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=self.cfg.enable_thinking)
        if hasattr(ids, "keys"):          # BatchEncoding in newer transformers
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)

    def chat(self, messages, json_schema=None):
        import torch
        from transformers import DynamicCache

        ids = self._encode(messages)
        n = 0
        if self.prefix_cache and self._cache is not None and self._cache_ids:
            m = min(len(ids), len(self._cache_ids))
            while n < m and ids[n] == self._cache_ids[n]:
                n += 1
            n = min(n, len(ids) - 1)       # always prefill at least the last token
        if n > 0:
            cur = self._cache.get_seq_length()
            if n < cur:
                self._cache.crop(n - cur)     # transformers >= 4.5x: negative = tokens to remove from the end
            cache = self._cache
        else:
            cache = DynamicCache()
        self.last_prompt_tokens = len(ids)
        input_ids = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            out = self.model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), past_key_values=cache,
                                      max_new_tokens=self.cfg.max_tokens, do_sample=False,
                                      temperature=None, top_p=None, top_k=None)
        gen = out[0][len(ids):]
        self.last_new_tokens = int(gen.shape[0])
        if self.prefix_cache:
            self._cache = cache
            self._cache_ids = out[0][: cache.get_seq_length()].tolist()
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def reset(self) -> None:
        """Drop the KV cache (called between samples; the next prompt shares only the system prompt)."""
        self._cache = None
        self._cache_ids = []

    def count_tokens(self, messages) -> int:
        return len(self._encode(messages))


def build_llm(cfg: Config, fake_script: Optional[Callable | list] = None) -> BaseLLM:
    if cfg.llm_backend == "fake":
        return FakeLLM(fake_script or (lambda m: '{"name": "finish", "arguments": {}}'))
    if cfg.llm_backend == "hf":
        return HFBackend(cfg)
    return VLLMBackend(cfg)
