#!/usr/bin/env python3
"""Merge a saved LoRA adapter into the base model (bf16) and save it with the base model's own tokenizer/template.

  python scripts/merge_lora.py --adapter out/lora-v6 --out out/lora-v6-merged
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template-from", default=None, help="ship this model's chat template instead of the base's")
    ap.add_argument("--scale", type=float, default=1.0, help="WiSE-FT-style: merge only this fraction of the LoRA delta")
    args = ap.parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter)
    if args.scale != 1.0:
        for name, mod in model.named_modules():
            if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
                for k in mod.scaling:
                    mod.scaling[k] *= args.scale
    model = model.merge_and_unload()
    model.save_pretrained(args.out)
    tok = AutoTokenizer.from_pretrained(args.base)
    if args.template_from:
        tok.chat_template = AutoTokenizer.from_pretrained(args.template_from).chat_template
    tok.save_pretrained(args.out)
    print("merged", args.adapter, "->", args.out)


if __name__ == "__main__":
    main()
