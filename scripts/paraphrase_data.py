#!/usr/bin/env python3
"""Rewordings of the real train requests, for scripts/build_sft.py --paraphrases.

Development only: an open model writes them (the rules allow LLM-generated training data, no external model at
inference) and nothing here reads public_test. The gold calls stay those of the base sample, so a rewording is
kept only if it still means the same:
1. every string argument of the gold calls that the original request says word for word is said word for word
   again (message texts, names, search words: the gold keeps the original wording);
2. the same model, without the gold, solves it through the agent's rules on a fresh env and either reproduces
   the gold (as synth_data.py's cross-check) or makes exactly the calls it makes on the original request (the
   meaning survived even where the model gets the original wrong).

Output lines {id, k, text, check}; <out>.done lists finished base samples (resume).

  python scripts/paraphrase_data.py --data ./train+public_test --ids-file splits/train2_ids.txt --out out/para.jsonl --limit 20
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lifemate.config import Config  # noqa: E402
from lifemate.data import load_catalog, load_jsonl, resolve_installed_tools, transcript_targets  # noqa: E402
from lifemate.normalize_vi import nfc  # noqa: E402
from lifemate.prompts import action_schema, render_context, render_history  # noqa: E402
from lifemate.runtime import import_toolenv, locate_split  # noqa: E402
from lifemate.scorer import fuzzy_lookup_from_catalog, score_sample  # noqa: E402
from run_dev import gold_hash_for  # noqa: E402
from synth_data import NO_THINK, solve, solver_messages  # noqa: E402

TEACHER = """You reword what a user said to a Vietnamese voice assistant. Write {n} other ways the same person, in the same situation, could say the LATEST REQUEST, as speech recognition would transcribe it.

The assistant must make exactly the same tool calls (GOLD CALLS) for your version as for the original:
- Keep every detail the original gives: who, what, which item, times, dates, amounts, quantities, places, conditions ("nếu ... thì ..."), and the order of the tasks. A number may switch between words and digits if its value stays the same.
- Add nothing the original does not say. If the original leaves something out (so the assistant has to ask, or has to look it up), leave it out too.
- Keep references to earlier turns as references ("người đó", "cái lịch hồi nãy"); do not replace them with what they point to.
- Copy the KEEP WORD FOR WORD phrases exactly (message texts, names, search words).
- Change everything else: word order, sentence shape, fillers, politeness, particles (nha, nhé, nghen, hen, rứa, mô), forms of address that fit the user and HISTORY, shorter or longer. Each version must differ from the original and from the others."""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["requests"],
          "properties": {"requests": {"type": "array", "items": {"type": "string"}}}}


def said(phrase: str, text: str) -> bool:
    """phrase occurs in text as whole words (case and Unicode form ignored)."""
    return re.search(rf"(?<!\w){re.escape(nfc(phrase).lower())}(?!\w)", nfc(text).lower()) is not None


def keep_phrases(request: str, calls: list) -> list[str]:
    """String argument values of the gold calls that the request says word for word."""
    out = []

    def walk(v):
        if isinstance(v, str):
            if len(v) > 1 and v not in out and said(v, request):
                out.append(v)
        elif isinstance(v, (list, dict)):
            for x in (v.values() if isinstance(v, dict) else v):
                walk(x)
    for c in calls:
        walk(c["arguments"])
    return out


def user_message(s: dict) -> str:
    texts = [t or "" for _, t in transcript_targets(s)]
    calls = s["label"]["tool_calls"]
    gold = "\n".join(f"{c['name']} {json.dumps(c['arguments'], ensure_ascii=False)}" for c in calls) or "(no calls: out of scope)"
    keep = "\n".join(f'- "{p}"' for p in keep_phrases(texts[-1], calls)) or "(none)"
    return (f"# CONTEXT\n{render_context(s['context'])}\n\n"
            f"# HISTORY (earlier turns)\n{render_history(s.get('history') or [], texts[:-1]) or '[]'}\n\n"
            f"# LATEST REQUEST\n{texts[-1]}\n\n# GOLD CALLS\n{gold}\n\n# KEEP WORD FOR WORD\n{keep}")


def candidates(s: dict, text: str, n: int) -> tuple[list[str], list[str]]:
    """Teacher JSON -> (rewordings to solve, rejection reasons)."""
    try:
        out = json.loads(text)["requests"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return [], ["unparsable output"]
    orig = transcript_targets(s)[-1][1] or ""
    keep = keep_phrases(orig, s["label"]["tool_calls"])
    seen, kept, why = {nfc(orig).lower().strip(" .!?")}, [], []
    for t in out[:n]:
        t = t.strip() if isinstance(t, str) else ""
        norm = nfc(t).lower().strip(" .!?")
        if not norm or norm in seen:
            why.append("empty or a copy")
        elif not all(said(p, t) for p in keep):
            why.append("lost a word-for-word phrase")
        else:
            kept.append(t)
        seen.add(norm)
    return kept, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ids-file", default=None, help="base samples (default: all of train)")
    ap.add_argument("--out", required=True, help="jsonl, appended; <out>.done records every finished base sample (resume)")
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    ap.add_argument("--n", type=int, default=3, help="rewordings asked per base sample")
    ap.add_argument("--limit", type=int, default=None, help="at most this many new base samples in this run")
    ap.add_argument("--chunk", type=int, default=32, help="base samples per batch (written after each batch)")
    ap.add_argument("--max-model-len", type=int, default=20480, help="the solver prompt is the longest, <= 14k tokens")
    ap.add_argument("--gpu-util", type=float, default=0.95)
    ap.add_argument("--kv-fp8", action="store_true", help="fp8 KV cache, about twice the batch on a 40GB GPU")
    ap.add_argument("--temperature", type=float, default=0.9, help="teacher sampling (the solver is greedy)")
    args = ap.parse_args()

    root, split, split_dir = locate_split(args.data, "train")
    ToolEnv = import_toolenv(split_dir)
    catalog = load_catalog(split_dir / f"{split}_tools_catalog.json")
    keep_ids = {l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()} if args.ids_file else None
    bases = [s for s in load_jsonl(split_dir / f"{split}.jsonl") if keep_ids is None or s["id"] in keep_ids]
    done_path = Path(args.out + ".done")
    done = {json.loads(l)["base_id"] for l in done_path.read_text(encoding="utf-8").splitlines()} if done_path.exists() else set()
    todo = [s for s in bases if s["id"] not in done][: args.limit]

    from vllm import LLM, SamplingParams

    from lifemate.llm import json_output_kwargs

    llm = LLM(model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_util,
              kv_cache_dtype="fp8" if args.kv_fp8 else "auto", enable_prefix_caching=True, seed=0)
    teach = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=2048, seed=0, **json_output_kwargs(SCHEMA))
    system_text = TEACHER.format(n=args.n)
    fuzzy_of, max_chars = fuzzy_lookup_from_catalog(catalog), Config().result_max_chars
    n_kept, n_done, why_all, t0 = 0, 0, Counter(), time.time()
    print(f"[para] {len(todo)} base samples to do ({len(done)} done before), model={args.model}", flush=True)
    with open(args.out, "a", encoding="utf-8") as fo, open(done_path, "a", encoding="utf-8") as fd:
        for i in range(0, len(todo), args.chunk):
            chunk = todo[i: i + args.chunk]
            outs = llm.chat([[{"role": "system", "content": system_text}, {"role": "user", "content": user_message(s)}] for s in chunk],
                            teach, use_tqdm=False, chat_template_kwargs=NO_THINK)
            runs, why = [], {s["id"]: [] for s in chunk}
            for s, o in zip(chunk, outs):
                texts, why[s["id"]] = candidates(s, o.outputs[0].text, args.n)
                tools = list(resolve_installed_tools(s, catalog))
                sp = SamplingParams(temperature=0.0, max_tokens=512, seed=0, **json_output_kwargs(action_schema(tools)))
                # k = 0 is the original request: what the solver makes of it is the reference for its rewordings
                for k, t in enumerate([transcript_targets(s)[-1][1] or ""] + texts):
                    runs.append({"base": s, "k": k, "text": t, "msgs": solver_messages({**s, "audio_text": t}, tools),
                                 "env": ToolEnv(split, s["id"], root=root), "sp": sp})
            solve(llm, runs, max_chars)
            ref = {r["base"]["id"]: json.dumps(r["calls"], sort_keys=True, ensure_ascii=False) for r in runs if r["k"] == 0}
            for s in chunk:
                gold = s["label"]["tool_calls"]
                ghash, kept = gold_hash_for(ToolEnv, split, s["id"], root, gold), 0
                for r in (r for r in runs if r["base"] is s and r["k"] > 0):
                    sc = score_sample(s["id"], r["calls"], r["env"].get_state_hash(), gold, ghash, fuzzy_of)
                    calls = json.dumps(r["calls"], sort_keys=True, ensure_ascii=False)
                    # extra reads are fine; an empty reference proves nothing
                    check = "gold" if (sc.score == 1.0 if not gold else sc.hash_ok and sc.M == len(gold)) else \
                        "same as original" if r["calls"] and calls == ref[s["id"]] else None
                    if check:
                        fo.write(json.dumps({"id": s["id"], "k": r["k"], "text": r["text"], "check": check}, ensure_ascii=False) + "\n")
                        kept += 1
                    else:
                        why[s["id"]].append("solver differs")
                fd.write(json.dumps({"base_id": s["id"], "kept": kept, "rejected": why[s["id"]]}, ensure_ascii=False) + "\n")
                n_kept += kept
                why_all.update(why[s["id"]])
            fo.flush()
            fd.flush()
            n_done += len(chunk)
            el = time.time() - t0
            print(f"[para] {n_done}/{len(todo)} kept={n_kept} {el / 60:.0f} min, eta {el / n_done * (len(todo) - n_done) / 60:.0f} min", flush=True)
    print(f"[para] done: {n_kept} rewordings kept from {n_done} base samples; rejected: {why_all.most_common(6)}", flush=True)


if __name__ == "__main__":
    main()
