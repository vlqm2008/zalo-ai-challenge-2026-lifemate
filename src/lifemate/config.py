from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # ---- models (total < 6B, LLM < 5B, context <= 16K) ----
    llm_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    llm_backend: str = "vllm"            # vllm | hf | fake
    llm_dtype: str = "bfloat16"          # A30 has no FP8; keep bf16
    max_model_len: int = 16384
    vllm_memory_gb: float = 14.0         # absolute budget so a 32 GB dev card behaves like the 24 GB A30
    enable_thinking: bool = False
    max_tokens: int = 400
    asr_model: str = "vinai/PhoWhisper-medium"
    asr_backend: str = "hf"              # hf | ct2 (faster-whisper) | cache | fake
    asr_num_beams: int = 1               # greedy = deterministic
    asr_chunk_length_s: int = 30
    asr_ct2_dir: Optional[str] = None    # ct2: converted model dir (auto-converted from asr_model if missing)
    asr_compute_type: str = "float16"    # ct2: float16 | int8_float16 (A30 and T4 both fine)
    asr_cache_path: Optional[str] = None # cache: JSON {relative audio path: transcript} from precompute_asr.py
    asr_prompt: Optional[str] = None     # Whisper initial prompt: "domain" = asr.DOMAIN_PROMPT (general app vocabulary), or any text

    # ---- agent loop ----
    max_steps: int = 8                   # LLM turns per sample
    max_repairs: int = 2                 # validate() failures fed back to the LLM
    sample_budget_s: float = 22.0        # soft watchdog (organisers: 30 s hard, on an A30)
    use_thought: bool = True             # short "thought" field before the action
    use_guided_json: bool = True         # vLLM structured output on the action schema
    auto_read_before_write: bool = True  # insert the x-ref read tool before a write if not yet called
    snap_threshold: float = 0.80         # similarity needed to snap a non-fuzzy string to a known value
    few_shot_path: Optional[str] = None  # JSONL of {"messages": [...]} examples prepended after system
    few_shot_mode: str = "static"        # static: every row of few_shot_path | knn: the k rows whose "text" is closest to the transcript
    few_shot_k: int = 2                  # knn: number of retrieved train trajectories
    few_shot_min_sim: float = 0.25       # knn: minimum token-Jaccard to use a neighbour at all
    result_max_chars: int = 2500         # tool result size shown to the LLM
    result_time_hints: bool = False      # append shifted-timestamp hints computed from tool results (normalize_vi.result_time_hints)
    prompt_extras: bool = False          # Phase-3 prompt additions (optional params omitted, short message style); measured separately
    ask_guards: bool = False             # deferred sys.ask_missing_info when a contact is ambiguous (same-name hits) or an amount is ungrounded

    # ---- misc ----
    seed: int = 0
    log_dir: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Optional[str] = None, overrides: Optional[dict] = None) -> "Config":
        data = {}
        if path:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        if overrides:
            data.update({k: v for k, v in overrides.items() if v is not None})
        known = {f for f in cls.__dataclass_fields__}
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        cfg.extra = {k: v for k, v in data.items() if k not in known}
        return cfg

    def dump(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)
