#!/usr/bin/env python3
"""Synthetic training samples written by a teacher LLM (Claude) and checked on the train environment.

Development only: the rules allow LLM-generated training data but no external model at inference, and
nothing here reads public_test. For each base sample of train the teacher sees its context, history and
app state and writes new spoken requests with their gold tool calls. A request is kept only if every call
runs on a fresh ToolEnv of the base sample, every ID argument was returned by an earlier call or by
HISTORY (the assistant never sees the state), and its type matches its calls (ask ends with
sys.ask_missing_info, oos has none). Output lines are in train.jsonl format, id <base>_s<k> plus base_id,
for scripts/build_sft.py --extra (which drops the bases outside its --ids-file, so dev2 stays clean).

  ANTHROPIC_API_KEY=... python scripts/synth_data.py --data ./train+public_test --out out/synth.jsonl --limit 20
  python scripts/synth_data.py --data ./train+public_test --out /dev/null --check-gold     # checks on the gold, no API
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifemate.data import ASK_TOOL_NAME, load_catalog, load_jsonl, ref_params, resolve_installed_tools, transcript_targets  # noqa: E402
from lifemate.prompts import render_context, render_history, render_tools, system_prompt  # noqa: E402
from lifemate.runtime import import_toolenv, locate_split  # noqa: E402

# one request per slot and base sample: aimed at the dev2 losses (ask/act decisions, multi-step), plus some out-of-scope
SLOTS = {
    "act": "a clear request (one or two tasks) the installed tools can fully do; if HISTORY is not empty, it refers to something "
           "from HISTORY (\"người đó\", \"cái lịch đó\", an item returned there)",
    "multi": "two or three tasks in one sentence, at least one conditional on what a read tool returns (\"nếu ... thì ..., "
             "không thì ...\") or needing a lookup before the write",
    "ask": "a request where one required parameter cannot be determined from the request, CONTEXT, HISTORY or tool results "
           "(the user does not say the amount/time/which one, or the state holds two matching items such as two contacts with "
           "the same name); the calls do everything else first and end with sys.ask_missing_info (rule 7)",
    "pair": "the same intent as your \"ask\" request, but this time the user states the missing detail, so everything is done "
            "without asking",
    "oos": "a request the assistant must not act on: no installed app can do it, or the user takes it back mid-sentence "
           "(\"à thôi khỏi\"), or only chats; calls is []",
}
MAX_CALLS = 7   # the agent runs at most 8 steps including finish

TEACHER = """You write training data for a Vietnamese voice assistant that acts on the user's phone by calling tools.
Each sample is what the user says in the LATEST turn (transcribed speech) and the exact gold sequence of tool calls the assistant must make for it. The gold calls follow the assistant's rules below exactly; the grader compares the calls and the final app state, so every ID, time and amount must be right.

# Assistant rules (the gold calls follow them)
{rules}

# Writing the request
- Natural spoken Vietnamese as a speech recognizer transcribes it: casual, sometimes long-winded, regional particles (nha, nhé, nghen, hen, rứa, mô, chơ, nớ), forms of address that fit the user profile and HISTORY, occasional self-corrections ("5 giờ, à không, 5 rưỡi") and English or brand names as spoken.
- Numbers mostly as words ("hai trăm nghìn", "tám giờ rưỡi tối mai"), sometimes digits.
- Refer to things as a user would (names, rooms, titles), never by ID, and only to things that exist in the APP STATE (or clearly do not, for ask and out-of-scope requests).
- Each request differs from the original request and from the others in intent, not just wording.

# Writing the calls
- The assistant cannot see the APP STATE, only tool results: before passing an ID, the calls use the read tool that returns it (rule 2), unless the ID is already in HISTORY.
- Compute times from CONTEXT.current_time (+07:00) and amounts exactly. Use only the installed tools (plus sys.ask_missing_info) and their schema parameter names.
- arguments_json is the JSON object of arguments as a string, for example "{{\\"query\\": \\"Lan\\"}}".

# Tool catalog (every app; a sample may only use its installed tools)
{catalog}

# Real samples with their gold calls (results shortened)
{examples}"""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["samples"], "properties": {"samples": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["type", "request", "calls"], "properties": {
        "type": {"type": "string", "enum": list(SLOTS)},
        "request": {"type": "string"},
        "calls": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["name", "arguments_json"],
                                             "properties": {"name": {"type": "string"}, "arguments_json": {"type": "string"}}}}}}}}}

# $/MTok (input, output) for the running cost line; shortcut: copied 2026-10, check the pricing page before a big run
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}


def gold_type(calls: list) -> str:
    return "oos" if not calls else "ask" if calls[-1]["name"] == ASK_TOOL_NAME else "act"


def check(base: dict, typ: str, calls: list, tools: dict, env, said: str = "") -> tuple[list | None, str | None]:
    """Run the calls on env (fresh, of the base sample): (tool_results, None) if the sample is usable, else (None, reason).
    said: the request text (an ID the user reads out, such as a tracking number, counts as known)."""
    names = [c["name"] for c in calls]
    if (typ == "oos") != (not calls):
        return None, "oos must have no calls"
    if (typ == "ask") != (bool(names) and names[-1] == ASK_TOOL_NAME) or ASK_TOOL_NAME in names[:-1]:
        return None, "sys.ask_missing_info must be the last call, and only in ask"
    if len(calls) > MAX_CALLS:
        return None, "too many calls"
    if len({json.dumps(c, sort_keys=True, ensure_ascii=False) for c in calls}) < len(calls):
        return None, "repeated call"
    seen = json.dumps(base.get("history") or [], ensure_ascii=False)   # IDs the assistant can know
    results = []
    for c in calls:
        tool, args = tools.get(c["name"]), c["arguments"]
        if tool is None:
            return None, f"not installed: {c['name']}"
        if c["name"] == ASK_TOOL_NAME:
            target, params = tools.get(args.get("tool_name")), args.get("params")
            if not target or not isinstance(params, list) or len(params) != 1 or \
                    params[0] not in ((target.get("parameters") or {}).get("properties") or {}):
                return None, "ask needs an installed tool_name and exactly one of its parameters"
        for k in ref_params(tool):
            vals = args.get(k) if isinstance(args.get(k), list) else [args.get(k)]
            if any(v is not None and json.dumps(v, ensure_ascii=False) not in seen and str(v) not in said for v in vals):
                return None, f"ungrounded id {c['name']}.{k}"
        res = env.call(c["name"], args)
        if not res.get("ok") or res.get("done") is False:
            return None, f"failed {c['name']}"
        results.append(res)
        seen += json.dumps(res, ensure_ascii=False)
    return results, None


def render_example(s: dict, catalog: dict) -> str:
    texts = [t or "" for _, t in transcript_targets(s)]
    calls = s["label"]["tool_calls"]
    res = s["label"].get("tool_results") or []
    lines = [f"## {s['id']}", f"current_time: {s['context'].get('current_time')}",
             "HISTORY:\n" + (render_history(s.get("history") or [], texts[:-1])[:1500] or "[]"),
             f"REQUEST: {texts[-1]}", "GOLD CALLS:"]
    for i, c in enumerate(calls):
        r = json.dumps(res[i] if i < len(res) else {}, ensure_ascii=False)
        lines.append(f"{c['name']} {json.dumps(c['arguments'], ensure_ascii=False)} -> {r[:300]}")
    return "\n".join(lines + (["(no calls)"] if not calls else []))


def pick_examples(samples: list[dict], k: int = 2) -> list[dict]:
    """The k shortest samples of each behaviour: ask, out-of-scope, multi-step, uses HISTORY, other."""
    def kind(s):
        calls = s["label"]["tool_calls"]
        return gold_type(calls) if gold_type(calls) != "act" else "multi" if len(calls) >= 3 else "hist" if s.get("history") else "single"
    size = lambda s: len(json.dumps(s, ensure_ascii=False))  # noqa: E731
    return [s for b in ("ask", "oos", "multi", "hist", "single") for s in sorted((s for s in samples if kind(s) == b), key=size)[:k]]


def base_message(s: dict, tools: list[dict], state: dict, slots: list[str]) -> str:
    texts = [t or "" for _, t in transcript_targets(s)]
    gold = "\n".join(f"{c['name']} {json.dumps(c['arguments'], ensure_ascii=False)}" for c in s["label"]["tool_calls"]) or "(no calls)"
    task = "\n".join(f"{i + 1}. type \"{t}\": {SLOTS[t]}" for i, t in enumerate(slots))
    return (f"# CONTEXT\n{render_context(s['context'])}\n\n# INSTALLED TOOLS\n{', '.join(t['name'] for t in tools)}\n\n"
            f"# HISTORY (earlier turns, already applied to the app state)\n{render_history(s.get('history') or [], texts[:-1]) or '[]'}\n\n"
            f"# APP STATE (hidden from the assistant)\n{json.dumps(state, ensure_ascii=False)}\n\n"
            f"# ORIGINAL REQUEST of this sample (write different ones)\n{texts[-1]}\n{gold}\n\n"
            f"# TASK\nWrite {len(slots)} samples, in this order:\n{task}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ids-file", default=None, help="base samples (default: all of train; train2 ids go first)")
    ap.add_argument("--examples-ids", default="splits/train2_ids.txt", help="the prompt's real examples come only from these ids")
    ap.add_argument("--out", required=True, help="jsonl, appended; <out>.done records every finished base sample (resume)")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--limit", type=int, default=None, help="at most this many new base samples in this run")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--check-gold", action="store_true", help="only run the checks on the gold calls (no API): how strict they are")
    args = ap.parse_args()

    root, split, split_dir = locate_split(args.data, "train")
    ToolEnv = import_toolenv(split_dir)
    catalog = load_catalog(split_dir / f"{split}_tools_catalog.json")
    samples = list(load_jsonl(split_dir / f"{split}.jsonl"))
    ids = lambda p: {l.strip() for l in Path(p).read_text().splitlines() if l.strip()}  # noqa: E731
    first = ids(args.examples_ids)
    keep = ids(args.ids_file) if args.ids_file else None
    bases = [s for s in samples if keep is None or s["id"] in keep]
    bases.sort(key=lambda s: (s["id"] not in first, hashlib.sha1(s["id"].encode()).hexdigest()))   # mixed order, train2 first

    def tools_of(s):
        return {t["name"]: t for t in resolve_installed_tools(s, catalog)}

    if args.check_gold:
        why = Counter()
        for s in bases:
            calls = s["label"]["tool_calls"]
            _, reason = check(s, gold_type(calls), calls, tools_of(s), ToolEnv(split, s["id"], root=root), s.get("audio_text") or "")
            why[reason or "ok"] += 1
        print(f"gold checks on {len(bases)} samples: {why.most_common(12)}")
        return

    import anthropic

    done_path = Path(args.out + ".done")
    done = {json.loads(l)["base_id"] for l in done_path.read_text(encoding="utf-8").splitlines()} if done_path.exists() else set()
    todo = [s for s in bases if s["id"] not in done][: args.limit]
    examples = pick_examples([s for s in samples if s["id"] in first])
    system_text = TEACHER.format(rules=system_prompt(True), catalog=render_tools(list(catalog["_tools"].values())),
                                 examples="\n\n".join(render_example(s, catalog) for s in examples))
    client = anthropic.Anthropic(max_retries=5)
    # server-side fallback re-runs a refused request on another model; Haiku 5.5 has none
    fallback = {} if "haiku" in args.model else {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}

    def generate(s):
        tools = list(tools_of(s).values())
        env0 = ToolEnv(split, s["id"], root=root)
        state = {k: v for k, v in env0.get_state().items() if k != "_meta"}
        slots = ["act", "multi", "ask", "pair"] + (["oos"] if int(hashlib.sha1(s["id"].encode()).hexdigest(), 16) % 4 == 0 else [])
        with client.beta.messages.stream(
            model=args.model, max_tokens=64000,
            system=[{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": base_message(s, tools, state, slots)}],
            output_config={"effort": args.effort, "format": {"type": "json_schema", "schema": SCHEMA}}, **fallback,
        ) as stream:
            msg = stream.get_final_message()
        if msg.stop_reason == "refusal":
            return [], ["refusal"], msg.usage
        try:
            out = json.loads([b.text for b in msg.content if b.type == "text"][-1])["samples"]
        except (IndexError, json.JSONDecodeError, KeyError):
            return [], [f"unparsable output ({msg.stop_reason})"], msg.usage
        kept, why = [], []
        for k, c in enumerate(out):
            try:
                calls = [{"name": x["name"], "arguments": json.loads(x["arguments_json"])} for x in c["calls"]]
            except json.JSONDecodeError:
                why.append(f"{c['type']}: arguments not JSON")
                continue
            if not all(isinstance(x["arguments"], dict) for x in calls):
                why.append(f"{c['type']}: arguments not an object")
                continue
            results, reason = check(s, c["type"], calls, tools_of(s), ToolEnv(split, s["id"], root=root), c["request"])
            if reason:
                why.append(f"{c['type']}: {reason}")
                continue
            sid = f"{s['id']}_s{k}"
            kept.append({"id": sid, "base_id": s["id"], "synth_type": c["type"], "audio": f"synthetic/{sid}.wav",
                         "audio_text": c["request"].strip(), "context": s["context"], "history": s.get("history") or [],
                         "label": {"tool_calls": calls, "tool_results": results}})
        return kept, why, msg.usage

    lock, tok, why_all, n_kept, n_done = threading.Lock(), Counter(), Counter(), 0, 0
    p_in, p_out = PRICES.get(args.model, (0.0, 0.0))
    print(f"[synth] {len(todo)} base samples to do ({len(done)} done before), model={args.model} effort={args.effort}", flush=True)
    with ThreadPoolExecutor(args.workers) as pool, open(args.out, "a", encoding="utf-8") as fo, \
            open(done_path, "a", encoding="utf-8") as fd:
        futs = {pool.submit(generate, s): s["id"] for s in todo}
        for f in as_completed(futs):
            try:
                kept, why, usage = f.result()
            except (anthropic.RateLimitError, anthropic.APIConnectionError) as e:   # transient: stays undone for the next run
                print(f"[synth] {futs[f]} skipped ({type(e).__name__})", flush=True)
                continue
            except anthropic.APIStatusError as e:
                if e.status_code < 500:
                    raise   # a 4xx repeats on every request: stop instead of paying for it
                print(f"[synth] {futs[f]} skipped (HTTP {e.status_code})", flush=True)
                continue
            with lock:
                for r in kept:
                    fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                fd.write(json.dumps({"base_id": futs[f], "kept": len(kept), "rejected": why}, ensure_ascii=False) + "\n")
                fo.flush()
                fd.flush()
                n_done, n_kept = n_done + 1, n_kept + len(kept)
                why_all.update(why)
                for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"):
                    tok[k] += getattr(usage, k, 0) or 0
                cost = ((tok["input_tokens"] + 1.25 * tok["cache_creation_input_tokens"] + 0.1 * tok["cache_read_input_tokens"]) * p_in
                        + tok["output_tokens"] * p_out) / 1e6
                print(f"[synth] {n_done}/{len(todo)} kept={n_kept} tokens in={tok['input_tokens']} cache_read={tok['cache_read_input_tokens']} "
                      f"out={tok['output_tokens']} cost~${cost:.2f}", flush=True)
    print(f"[synth] done: {n_kept} samples kept from {n_done} base samples; rejected: {why_all.most_common(10)}", flush=True)


if __name__ == "__main__":
    main()
