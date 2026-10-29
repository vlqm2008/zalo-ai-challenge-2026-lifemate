#!/usr/bin/env python3
"""LoRA SFT of the agent LLM on trajectories from build_sft.py (TRL + PEFT).

  python scripts/train_lora.py --train data/sft_train.jsonl --out out/lora-qwen3-4b
  python scripts/merge_lora.py ...   (or load the adapter in vLLM with --enable-lora)

NOT executed in the authoring environment (no GPU there): treat as a starting point and
check the TRL version's SFTConfig fields. Memory: Qwen3-4B bf16 + LoRA r=16 + grad
checkpointing fits in ~20 GB at seq len 8K, batch 1, grad accum 16.
"""
from __future__ import annotations

import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


# ChatML with generation markers: for plain-string messages (system first, no tools) it renders byte-identically to
# Qwen3's own template (asserted below), and lets TRL's assistant_only_loss mask everything but the assistant turns.
# Round 1 trained with the loss on the whole sequence (18k-char prompts, ~100-token answers): oracle dev fell from
# 42.85 (base + kNN) to 32.76 -> EXPERIMENTS 5.
GEN_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message.role == 'assistant' %}"
    "{{- '<|im_start|>assistant\n' }}{% generation %}{{- message.content + '<|im_end|>\n' }}{% endgeneration %}"
    "{%- else %}"
    "{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", default=None)
    ap.add_argument("--out", default="out/lora")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"], help="T4 (no bf16): float16")
    ap.add_argument("--qlora", action="store_true", help="4-bit base (bitsandbytes) for 16 GB cards")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--template-from", default=None, help="use this model's chat template (e.g. Qwen/Qwen3-4B-Instruct-2507 plain ChatML for Qwen3.5)")
    ap.add_argument("--target-modules", default="qkvo-mlp", help="'qkvo-mlp' (Qwen3 default) or 'all-linear' (hybrid models such as Qwen3.5)")
    ap.add_argument("--assistant-only", action="store_true",
                    help="loss on the assistant turns only (ChatML template with {%% generation %%} markers; see GEN_TEMPLATE)")
    args = ap.parse_args()

    def load(path):
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        return Dataset.from_list([{"messages": r["messages"]} for r in rows])

    tok = AutoTokenizer.from_pretrained(args.model)
    if args.template_from:
        tok.chat_template = AutoTokenizer.from_pretrained(args.template_from).chat_template
    orig_template = tok.chat_template
    if args.assistant_only:
        probe = json.loads(open(args.train, encoding="utf-8").readline())["messages"]
        a = tok.apply_chat_template(probe, tokenize=False)
        tok.chat_template = GEN_TEMPLATE
        b = tok.apply_chat_template(probe, tokenize=False)
        assert a == b, "GEN_TEMPLATE does not reproduce the model's chat template for this data"
        enc = tok.apply_chat_template(probe, tokenize=True, return_assistant_tokens_mask=True, return_dict=True)
        n_a = sum(enc["assistant_masks"]); print(f"assistant-only loss: {n_a}/{len(enc['input_ids'])} tokens supervised in the first example")
        assert 0 < n_a < len(enc["input_ids"])
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    kw = dict(dtype=dtype, attn_implementation="sdpa")
    if args.qlora:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                                                       bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    peft_cfg = LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05, task_type="CAUSAL_LM",
                          target_modules=("all-linear" if args.target_modules == "all-linear" else
                                          ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    cfg = SFTConfig(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_steps=10, per_device_train_batch_size=1, gradient_accumulation_steps=args.grad_accum,   # warmup_ratio was removed in transformers 5
        gradient_checkpointing=True, bf16=(args.dtype == "bfloat16"), fp16=(args.dtype == "float16"), logging_steps=10, save_strategy="epoch",
        max_length=args.max_len, seed=args.seed, report_to="none",
        assistant_only_loss=args.assistant_only,   # needs the {% generation %} markers of GEN_TEMPLATE
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=load(args.train),
                         eval_dataset=load(args.eval) if args.eval else None, processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(args.out)
    # merge for vLLM (simplest deployment: a plain HF model folder)
    if not args.qlora:   # a 4-bit base cannot be merged directly; reload the bf16/fp16 base and merge the adapter there
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(args.out + "-merged")
        tok.chat_template = orig_template   # ship the model's own template; GEN_TEMPLATE is a training-time device
        tok.save_pretrained(args.out + "-merged")
        print("saved", args.out, "and", args.out + "-merged")
    else:
        print("saved adapter", args.out, "- merge with: PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base, dtype=float16), out).merge_and_unload()")


if __name__ == "__main__":
    main()
