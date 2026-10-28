"""python test_train.py — runs train_sft.train() end to end (LoRA, masked loss, save, merge) on a tiny random Qwen3, on CPU."""
import tempfile
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

import agent
import train_sft

tok = AutoTokenizer.from_pretrained(agent.MODEL)
data = train_sft.build_dataset(["train_0001", "train_0360", "train_0824", "train_0109"], tok)  # actions, ask, out of scope, history
assert all(any(l != -100 for l in labels) for _, labels in data)

torch.manual_seed(0)
cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
model = Qwen3ForCausalLM(cfg).to(torch.bfloat16)
with tempfile.TemporaryDirectory() as out:
    peft_model = train_sft.train(model, data, out, epochs=2, lr=1e-3, accum=2)
    assert (Path(out) / "adapter_model.safetensors").is_file()
    merged = peft_model.merge_and_unload()
    assert torch.isfinite(merged(input_ids=torch.tensor([data[0][0][:64]])).logits).all()
print("ok")
