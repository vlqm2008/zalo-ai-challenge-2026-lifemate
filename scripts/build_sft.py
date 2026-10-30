#!/usr/bin/env python3
"""Turn train labels into SFT chat trajectories in EXACTLY the format the agent uses at
inference (same system prompt, same user message, same tool-result rendering), so the
fine-tuned model sees no distribution shift.

  python scripts/build_sft.py --data ./data --ids-file splits/train_ids.txt --out out/sft_train.jsonl

Each line: {"id": ..., "messages": [{"role": "system"|"user"|"assistant", "content": ...}, ...]}
Assistant turns are the gold actions as JSON without a "thought" field -> train and run the
fine-tuned model with use_thought=false. Also writes a few-shot file (--few-shot-out) with
the K shortest trajectories, usable by the un-tuned model via Config.few_shot_path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifemate.config import Config  # noqa: E402
from lifemate.data import ASK_TOOL_NAME, load_catalog, load_jsonl, resolve_installed_tools, transcript_targets  # noqa: E402
from lifemate.normalize_vi import build_hints  # noqa: E402
from lifemate.prompts import FINISH, SYSTEM_PROMPT, build_user_message, render_history, render_result, render_tools  # noqa: E402
from lifemate.runtime import locate_split  # noqa: E402


def trajectory(sample: dict, catalog: dict, cfg: Config, asr: dict | None = None) -> list[dict] | None:
    label = sample.get("label")
    if not label:
        return None
    context = sample.get("context") or {}
    targets = transcript_targets(sample)
    texts = [(asr[p] if asr and p in asr else t) or "" for p, t in targets]   # asr: {audio path: transcript} replaces the oracle text
    transcript = texts[-1]
    if not transcript:
        return None
    tools = resolve_installed_tools(sample, catalog)
    hints = build_hints(transcript, context.get("current_time") or "")
    user = build_user_message(context, render_history(sample.get("history") or [], texts[:-1]), transcript, hints, render_tools(tools))
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    calls = label.get("tool_calls") or []
    results = label.get("tool_results") or []
    for i, c in enumerate(calls):
        action = {"name": c["name"], "arguments": c.get("arguments") or {}}
        msgs.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
        res = results[i] if i < len(results) else {"ok": True}
        msgs.append({"role": "user", "content": render_result(c["name"], res, cfg.result_max_chars) +
                     "\n\nHành động tiếp theo (hoặc finish nếu đã làm xong mọi việc)?"})
        if c["name"] == ASK_TOOL_NAME:
            msgs.pop()  # the agent stops right after asking; no further turn to learn
            return msgs
    msgs.append({"role": "assistant", "content": json.dumps({"name": FINISH, "arguments": {}}, ensure_ascii=False)})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--out", default="out/sft_train.jsonl")   # never write under data/ (organisers' package)
    ap.add_argument("--few-shot-out", default="configs/few_shot.jsonl")
    ap.add_argument("--few-shot-k", type=int, default=4)
    ap.add_argument("--few-shot-strategy", default="diverse", choices=["diverse", "shortest"])
    ap.add_argument("--few-shot-keep-tools", action="store_true", help="keep the TOOLS block in few-shot user turns")
    ap.add_argument("--index-out", default="configs/few_shot_index.jsonl", help="kNN few-shot index: every trajectory (TOOLS stripped) + request text")
    ap.add_argument("--asr-aug", default=None, help="ASR cache JSON {audio path: text} of clips the ASR model never trained on: each "
                    "sample whose transcripts differ from the oracle text becomes an extra trajectory <id>_asr with the same gold")
    ap.add_argument("--extra", default=None, help="jsonl of extra samples in train.jsonl format with a base_id (scripts/synth_data.py); "
                    "with --ids-file only those whose base sample is in it, so synthetic data of dev bases never reaches training")
    ap.add_argument("--paraphrases", default=None, help="jsonl {id, k, text}: synthetic rewordings of the latest request; each one "
                    "becomes an extra trajectory <id>_p<k> with the same gold (synthetic data for development only, EXPERIMENTS 6)")
    args = ap.parse_args()
    cfg = Config()
    root, split, split_dir = locate_split(args.data, args.split)
    catalog = load_catalog(split_dir / f"{split}_tools_catalog.json")
    samples = list(load_jsonl(split_dir / f"{split}.jsonl"))
    if args.ids_file:
        keep = {l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()}
        samples = [s for s in samples if s["id"] in keep]
    else:
        keep = None
    rows = []
    for s in samples:
        msgs = trajectory(s, catalog, cfg)
        if msgs:
            rows.append({"id": s["id"], "messages": msgs})
    n_orig = len(rows)
    if args.asr_aug:
        asr = json.loads(Path(args.asr_aug).read_text(encoding="utf-8"))
        for s in samples:
            targets = transcript_targets(s)
            # every user turn must have a transcript (no oracle/ASR mix), and an identical copy teaches nothing
            if not all(p in asr for p, _ in targets) or all(asr[p] == t for p, t in targets):
                continue
            msgs = trajectory(s, catalog, cfg, asr)
            if msgs:
                rows.append({"id": f'{s["id"]}_asr', "messages": msgs})
        print(f"{n_orig} original + {len(rows) - n_orig} ASR-transcript trajectories")
    if args.extra:
        n0 = len(rows)
        for s in load_jsonl(args.extra):
            if keep is None or s.get("base_id") in keep:
                msgs = trajectory(s, catalog, cfg)
                if msgs:
                    rows.append({"id": s["id"], "messages": msgs})
        print(f"{len(rows) - n0} extra trajectories from {args.extra}")
    n_aug = len(rows)
    if args.paraphrases:
        from paraphrase_data import drift

        by_id, n_drift = {s["id"]: s for s in samples}, 0
        for line in Path(args.paraphrases).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            pr = json.loads(line)
            base = by_id.get(pr["id"])
            if not base or not pr.get("text"):
                continue
            if drift(base, pr["text"]):   # files written before paraphrase_data.py checked this
                n_drift += 1
                continue
            clone = dict(base)
            clone["audio_text"] = {**base["audio_text"], "text": pr["text"]} if isinstance(base.get("audio_text"), dict) else pr["text"]
            msgs = trajectory(clone, catalog, cfg)
            if msgs:
                rows.append({"id": f'{pr["id"]}_p{pr.get("k", 0)}', "messages": msgs})
        print(f"{n_aug} original + {len(rows) - n_aug} paraphrased trajectories ({n_drift} dropped: numbers, times or gold words differ)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")   # trailing newline: files are concatenated
    print(f"wrote {len(rows)} trajectories to {args.out}")
    # few-shot: system prompt removed (the agent adds its own). "shortest" = the K shortest trajectories;
    # "diverse" = the shortest trajectory of each behaviour: oos (finish only), ask after a read, read->write,
    # conditional read -> write, multi-task (>=3 calls) - in that order, up to K.
    by_id = {s["id"]: s for s in samples}
    rows = rows[:n_orig]   # few-shot and kNN index: original trajectories only (augmented ids are not in by_id)

    def behaviour(r):
        calls = by_id[r["id"]]["label"]["tool_calls"]
        names = [c["name"] for c in calls]
        kinds = [catalog["_tools"].get(n, {}).get("kind") for n in names]
        if not calls:
            return "oos"
        if names[-1] == ASK_TOOL_NAME:
            return "ask_after_read" if len(calls) > 1 else "ask"
        if any(a == "read" and b == "write" for a, b in zip(kinds, kinds[1:])) and len(calls) <= 3:
            return "read_write"
        if len(calls) >= 3:
            return "multi"
        return "other"

    size = lambda r: sum(len(m["content"]) for m in r["messages"])  # noqa: E731
    if args.few_shot_strategy == "diverse":
        chosen = []
        for b in ("oos", "ask_after_read", "read_write", "multi", "ask", "other"):
            cands = sorted((r for r in rows if behaviour(r) == b), key=size)
            if cands:
                chosen.append(cands[0])
        short = chosen[: args.few_shot_k]
    else:
        short = sorted(rows, key=size)[: args.few_shot_k]
    def strip_tools(m):
        # the real prompt already lists TOOLS; in an example they only cost tokens (5-8k chars each)
        if args.few_shot_keep_tools or m["role"] != "user" or "## TOOLS" not in m["content"]:
            return m
        head, _, _ = m["content"].partition("## TOOLS")
        return {"role": "user", "content": head + "## TOOLS\n(như danh sách TOOLS của lượt thật, lược bỏ trong ví dụ)\n\nHãy trả lời hành động đầu tiên."}

    fs = [{"id": r["id"], "messages": [strip_tools(m) for m in r["messages"] if m["role"] != "system"]} for r in short]
    idx = [{"id": r["id"], "text": by_id[r["id"]].get("audio_text") or "",
            "messages": [strip_tools(m) for m in r["messages"] if m["role"] != "system"]} for r in rows]
    Path(args.index_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.index_out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in idx), encoding="utf-8")
    print(f"wrote kNN few-shot index with {len(idx)} trajectories to {args.index_out} (few_shot_mode=knn)")
    Path(args.few_shot_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.few_shot_out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in fs), encoding="utf-8")
    print(f"wrote {len(fs)} few-shot examples to {args.few_shot_out} (set few_shot_path in config to use)")


if __name__ == "__main__":
    main()
