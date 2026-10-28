"""Text-only LifeMate agent (step 2): transcript -> Qwen tool-calling loop on zaic_tools.

    python agent.py --ids splits/dev_seen.txt --out runs/dev_seen.jsonl [--limit 20]
    python score.py runs/dev_seen.jsonl

Train only for now: user turns come from audio_text (ASR replaces it in step 3).
Backend: MLX on a Mac (local tries), vLLM elsewhere (Colab, Docker).
Writes <out> (submission lines) and <out>.trace.jsonl (model turns and tool results, for error analysis).
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "train+public_test"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MAX_STEPS = 10
WEEKDAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)

RULES = """You are LifeMate, a Vietnamese voice assistant that acts by calling tools. Handle ONLY the user's last message; earlier turns are already done (their tool results are still valid, reuse their IDs).

Rules:
1. Call only the tools listed. Use IDs that appear in tool results or earlier turns, never invent them: get them with a read tool first and wait for its result.
2. Call a read tool only when you need its result (an ID, or a fact the request depends on). Create things (reminder, alarm, event, note, task, ride...) directly, without listing first.
3. Call list/status read tools without optional filters (smart_home.get_status {}, calendar.list_events {}, bill_payment.list_bills {}); search tools take the name the user said as query.
4. Pass an optional parameter only when the user states its value (brightness when they give a level). Never add repeat, label, note, color or similar on your own.
5. If the request has several parts, do all of them. If it has a condition ("nếu mai mưa thì..."), check it with a read tool first and do only the branch that holds.
6. Never guess an amount, a time, a person or an ID. If a required parameter cannot be known from the request, the context or tool results (no time given, nobody or several people match, no amount...), call sys.ask_missing_info with the tool name and the missing parameter names, then stop. Do the parts you can do before asking.
7. Call no tool at all when nothing should be done: small talk, thanks, the user cancels or changes their mind, or no installed app can do it.
8. Formats: datetime "YYYY-MM-DDTHH:MM:SS" plus the offset of the current time (e.g. +07:00), date "YYYY-MM-DD", money as an integer in VND ("500 nghìn" = 500000, "1 triệu rưỡi" = 1500000). "năm giờ sáng mai" = 05:00 of tomorrow, "9 giờ kém 15 tối" = 20:45, "chiều" = 12:00-18:00, "tối" = 18:00-23:00. Free text (message content, note, title) short, in Vietnamese, faithful to what the user said.
9. When done, answer in one short Vietnamese sentence without tool calls."""


def context_block(ctx):
    now = datetime.fromisoformat(ctx["current_time"])
    labels = {0: " (hôm nay)", 1: " (ngày mai)", 2: " (ngày kia / mốt)"}
    days = "\n".join(f"- {WEEKDAYS[d.weekday()]} {d:%Y-%m-%d}{labels.get(i, '')}" for i, d in ((i, now + timedelta(days=i)) for i in range(14)))
    return (f"Current time: {ctx['current_time']} ({WEEKDAYS[now.weekday()]}), timezone {ctx['timezone']}\n"
            f"User profile: {json.dumps(ctx['user_profile'], ensure_ascii=False)}\n"
            f"Current location: {json.dumps(ctx['location'])}\n"
            f"Next 14 days:\n{days}")


def tool_specs(catalog, installed):
    allowed = {t for a in installed.values() for t in a["tools"]} | {"sys.ask_missing_info"}
    specs = []
    for t in (t for a in catalog["apps"] for t in a["tools"] if t["name"] in allowed):
        params = t["parameters"]
        props = {k: {a: b for a, b in p.items() if a not in ("x-ref", "x-match")} for k, p in params["properties"].items()}
        specs.append({"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": {**params, "properties": props}}})
    return specs


def build_messages(item):
    msgs = [{"role": "system", "content": RULES + "\n\n" + context_block(item["context"])}]
    for h in item["history"]:
        if h["role"] == "user":
            msgs.append({"role": "user", "content": h["audio_text"]})
            continue
        if h.get("tool_calls"):
            msgs.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": c} for c in h["tool_calls"]]})
            msgs += [{"role": "tool", "content": json.dumps(r, ensure_ascii=False)} for r in h.get("tool_results", [])]
        msgs.append({"role": "assistant", "content": h.get("text", "")})
    msgs.append({"role": "user", "content": item["audio_text"]})
    return msgs


def parse_calls(text):
    calls = []
    for raw in CALL_RE.findall(text):
        try:
            c = json.loads(raw)
            calls.append((c["name"], c.get("arguments") or {}))
        except (ValueError, KeyError, TypeError):
            calls.append((None, raw))
    return calls


def unseen_ids(name, args, xrefs, seen):
    """x-ref arguments (IDs) whose value never appeared in a tool result or earlier call: the model made them up."""
    vals = [(k, v) for k in xrefs.get(name, ()) if k in args for v in (args[k] if isinstance(args[k], list) else [args[k]])]
    return [v for k, v in vals if json.dumps(v, ensure_ascii=False) not in seen]


def run_agent(item, env, catalog, llm):
    """Drive one sample to the end. Only calls that pass the checks reach env.call, so rejected attempts are not submitted."""
    msgs, tools = build_messages(item), tool_specs(catalog, item["context"]["installed_apps"])
    xrefs = {t["name"]: [k for k, p in t["parameters"]["properties"].items() if "x-ref" in p] for a in catalog["apps"] for t in a["tools"]}
    seen = json.dumps(item["history"], ensure_ascii=False)  # IDs the model may legitimately use; tool results get appended
    trace, done, max_prompt = [], set(), 0
    for _ in range(MAX_STEPS):
        text, n_prompt = llm(llm.tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False))
        max_prompt = max(max_prompt, n_prompt)
        msgs.append({"role": "assistant", "content": text})
        trace.append(text)
        calls, dups, stop, read_ran = parse_calls(text), 0, False, False
        for name, args in calls:
            key = json.dumps([name, args], sort_keys=True, ensure_ascii=False)
            if not isinstance(name, str) or not isinstance(args, dict):
                res = {"ok": False, "error": "tool call phải là JSON {\"name\": ..., \"arguments\": {...}}"}
            elif key in done and env.kind(name) == "read":  # a repeated write can be legit (redeem twice), a repeated read never is
                res, dups = {"ok": False, "error": "call này đã chạy rồi, không gọi lại"}, dups + 1
            elif read_ran and env.kind(name) != "read":  # it was written before the read result existed, so it may rest on a guess
                res = {"ok": False, "error": "chưa chạy: xem kết quả lệnh đọc trước rồi mới gọi"}
            elif not (v := env.validate(name, args))["ok"]:
                res = {"ok": False, "error": v["error"]}
            elif bad := unseen_ids(name, args, xrefs, seen):
                res = {"ok": False, "error": f"ID {bad} không có trong kết quả tool nào: gọi tool đọc để lấy ID đúng, hoặc sys.ask_missing_info"}
            else:
                res = env.call(name, args)
                done.add(key)
                read_ran |= env.kind(name) == "read"
                stop |= name == "sys.ask_missing_info"  # gold always ends the turn after asking
                seen += json.dumps(res, ensure_ascii=False)
            msgs.append({"role": "tool", "content": json.dumps(res, ensure_ascii=False)})
            trace.append(res)
        if not calls or stop or dups == len(calls):  # finished, asked, or only repeating itself
            break
    return trace, max_prompt


class MLXModel:
    """Greedy decoding on Apple Silicon. One KV cache for the whole run: each prompt only prefills what differs from the previous one."""

    def __init__(self, model_id):
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
        self.model, self.tok = load(model_id)
        self.cache, self.ids = make_prompt_cache(self.model), []

    def __call__(self, prompt, max_tokens=768):
        from mlx_lm import stream_generate
        ids = self.tok.encode(prompt)
        keep = 0
        while keep < min(len(ids) - 1, len(self.ids)) and ids[keep] == self.ids[keep]:
            keep += 1
        for c in self.cache:
            c.trim(c.offset - keep)  # drops the old tail, generated tokens included
        self.ids = ids
        text = "".join(r.text for r in stream_generate(self.model, self.tok, ids[keep:], max_tokens=max_tokens, prompt_cache=self.cache))
        return text, len(ids)


class VLLMModel:
    """Greedy decoding on an NVIDIA GPU (Colab, the A30 at scoring); vLLM's prefix cache reuses the prompt shared between turns."""

    def __init__(self, model_id):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model_id, max_model_len=16384, enable_prefix_caching=True, seed=0, gpu_memory_utilization=0.85)  # 16K = competition limit
        self.tok, self.params = self.llm.get_tokenizer(), SamplingParams(temperature=0, max_tokens=768)

    def __call__(self, prompt):
        out = self.llm.generate([prompt], self.params, use_tqdm=False)[0]
        return out.outputs[0].text, len(out.prompt_token_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT / "train"))
    from zaic_tools import ToolEnv

    items = {d["id"]: d for d in map(json.loads, open(ROOT / "train/train.jsonl", encoding="utf-8"))}
    catalog = json.load(open(ROOT / "train/train_tools_catalog.json", encoding="utf-8"))
    ids = [l.strip() for l in open(a.ids, encoding="utf-8") if l.strip()][: a.limit]
    llm = MLXModel(a.model) if sys.platform == "darwin" else VLLMModel(a.model)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fo, open(a.out.with_suffix(".trace.jsonl"), "w", encoding="utf-8") as ft:
        for i, sid in enumerate(ids, 1):
            t = time.time()
            env = ToolEnv("train", sid, root=ROOT)
            trace, max_prompt = run_agent(items[sid], env, catalog, llm)
            calls = env.get_tool_calls()
            fo.write(json.dumps({"id": sid, "tool_calls": calls, "state_hash": env.get_state_hash()}, ensure_ascii=False) + "\n")
            ft.write(json.dumps({"id": sid, "secs": round(time.time() - t, 1), "max_prompt_tokens": max_prompt, "trace": trace}, ensure_ascii=False) + "\n")
            fo.flush(), ft.flush()
            print(f"[{i}/{len(ids)}] {sid} {time.time() - t:5.1f}s prompt={max_prompt} calls={len(calls)}", flush=True)


if __name__ == "__main__":
    main()
