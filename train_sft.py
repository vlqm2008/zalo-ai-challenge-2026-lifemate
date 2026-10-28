"""LoRA fine-tune of Qwen3-4B-Instruct-2507 on the gold trajectories of splits/train.txt.

    python train_sft.py --out checkpoints/lora --merge-to checkpoints/merged
    python agent.py --model checkpoints/merged --ids splits/dev_seen.txt --out runs/sft_dev_seen.jsonl

Each train sample becomes the exact conversation agent.py builds at inference, followed by the gold calls replayed one
per assistant turn (results from zaic_tools) and a short closing reply; the loss covers only those last assistant turns.
"""
import argparse
import json
import random
import sys
import time

import agent

MAX_LEN = 12288  # longest train conversation is ~10K tokens; the competition allows 16K
FINAL = "Xong."


def to_chat(item, env):
    """(messages, index of the first gold assistant message): agent.py's prompt + one gold call per turn, then FINAL (not after asking)."""
    msgs = agent.build_messages(item)
    first = len(msgs)
    for c in item["label"]["tool_calls"]:
        msgs.append({"role": "assistant", "content": f"<tool_call>\n{json.dumps({'name': c['name'], 'arguments': c['arguments']}, ensure_ascii=False)}\n</tool_call>"})
        res = env.call(c["name"], c["arguments"])
        if c["name"] == "sys.ask_missing_info":  # the agent stops right after asking
            return msgs, first
        msgs.append({"role": "tool", "content": json.dumps(res, ensure_ascii=False)})
    msgs.append({"role": "assistant", "content": FINAL})
    return msgs, first


def tokenize(msgs, first, tools, tok):
    """input_ids, and labels that are -100 everywhere except the assistant messages from `first` on (<|im_end|> included)."""
    render = lambda m, gen=False: tok.apply_chat_template(m, tools=tools, tokenize=False, add_generation_prompt=gen)
    text, spans = render(msgs), []
    for i in range(first, len(msgs)):
        if msgs[i]["role"] == "assistant":
            a, b = render(msgs[:i], True), render(msgs[:i + 1])
            assert text.startswith(a) and text.startswith(b), "chat template is not append-only"  # also: training prefix == inference prompt
            spans.append((len(a), len(b)))
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    labels = [t if any(s <= o[0] < e for s, e in spans) else -100 for t, o in zip(enc["input_ids"], enc["offset_mapping"])]
    return enc["input_ids"], labels


def build_dataset(ids, tok):
    sys.path.insert(0, str(agent.ROOT / "train"))
    from zaic_tools import ToolEnv
    items = {d["id"]: d for d in map(json.loads, open(agent.ROOT / "train/train.jsonl", encoding="utf-8"))}
    catalog = json.load(open(agent.ROOT / "train/train_tools_catalog.json", encoding="utf-8"))
    data = []
    for sid in ids:
        msgs, first = to_chat(items[sid], ToolEnv("train", sid, root=agent.ROOT))
        data.append(tokenize(msgs, first, agent.tool_specs(catalog, items[sid]["context"]["installed_apps"]), tok))
    long = [len(x) for x, _ in data if len(x) > MAX_LEN]
    print(f"{len(data)} conversations, {len(long)} longer than {MAX_LEN} tokens dropped")
    return [d for d in data if len(d[0]) <= MAX_LEN]


def train(model, data, out, epochs=2, lr=1e-4, accum=16, seed=0):
    """LoRA on every linear layer; batch of one conversation, `accum` conversations per optimizer step."""
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import get_cosine_schedule_with_warmup

    random.seed(seed), torch.manual_seed(seed)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()  # LoRA weights and Adam state in fp32, the frozen base stays bf16
    steps = epochs * len(data) // accum
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, steps // 20), steps)
    dev, t0, step, seen, run = next(model.parameters()).device, time.time(), 0, 0, []
    model.train()
    for epoch in range(epochs):
        order = list(range(len(data)))
        random.Random(seed + epoch).shuffle(order)
        for i in order:
            seen += 1
            ids, labels = (torch.tensor(x, device=dev) for x in data[i])
            keep = (labels[1:] != -100).nonzero().squeeze(-1)  # positions whose next token is trained
            logits = model(input_ids=ids[None], logits_to_keep=keep).logits[0]  # lm_head only where needed: full 150K-vocab logits would not fit
            loss = F.cross_entropy(logits.float(), labels[keep + 1])
            (loss / accum).backward()
            run.append(loss.item())
            if seen % accum == 0 and step < steps:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(), sched.step(), opt.zero_grad()
                step += 1
                if step % 5 == 0 or step == steps:
                    el = time.time() - t0
                    print(f"epoch {epoch + 1} step {step}/{steps} loss {sum(run) / len(run):.4f} lr {sched.get_last_lr()[0]:.2e} {el / 60:.1f}min, ~{el / step * (steps - step) / 60:.0f}min left", flush=True)
                    run = []
        model.save_pretrained(out)  # every epoch, so a Colab disconnect loses at most one
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="where the LoRA adapter is saved (a Drive folder survives disconnects)")
    ap.add_argument("--merge-to", help="also save base + LoRA merged here, for vLLM")
    ap.add_argument("--ids", default="splits/train.txt")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--model", default=agent.MODEL)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    data = build_dataset([l.strip() for l in open(a.ids, encoding="utf-8") if l.strip()], tok)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model = train(model, data, a.out, a.epochs, a.lr, a.accum)
    if a.merge_to:
        merged = model.merge_and_unload().to(torch.bfloat16)
        merged.config.use_cache = True
        merged.save_pretrained(a.merge_to)
        tok.save_pretrained(a.merge_to)
        print("merged ->", a.merge_to)


if __name__ == "__main__":
    main()
