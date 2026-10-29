"""The agent loop: ASR -> hints -> LLM actions -> grounding -> validate -> env.call."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from .asr import BaseASR
from .config import Config
from .data import (ASK_TOOL_NAME, find_read_tool_for_ref, history_tool_results, ref_params, resolve_installed_tools,
                   sort_ask_params, transcript_targets)
from .grounding import EntityStore, ground_arguments
from .llm import BaseLLM
from .normalize_vi import build_hints, extract_numbers, norm_tokens, result_time_hints
from .prompts import (FINISH, action_schema, build_user_message, parse_action, render_history, render_result,
                      render_tools, system_prompt)


def _canon(name: str, args: dict) -> str:
    return name + "|" + json.dumps(args, ensure_ascii=False, sort_keys=True)


class Agent:
    def __init__(self, cfg: Config, llm: BaseLLM, asr: BaseASR, catalog: dict):
        self.cfg = cfg
        self.llm = llm
        self.asr = asr
        self.catalog = catalog
        self.few_shot: list[dict] = []
        self.few_shot_index: list[tuple[set, list[dict], str]] = []   # knn: (tokens of the request, messages, id)
        if cfg.few_shot_path:
            for line in Path(cfg.few_shot_path).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if cfg.few_shot_mode == "knn":
                    self.few_shot_index.append((set(norm_tokens(row.get("text") or "")), row["messages"], row.get("id", "")))
                else:
                    self.few_shot.extend(row["messages"])

    def retrieve_few_shot(self, transcript: str) -> tuple[list[dict], list[tuple[str, float]]]:
        """kNN few-shot: the k train trajectories whose request is closest (token Jaccard) to this transcript."""
        if not self.few_shot_index:
            return list(self.few_shot), []
        q = set(norm_tokens(transcript))
        scored = []
        for toks, msgs, rid in self.few_shot_index:
            sim = len(q & toks) / max(1, len(q | toks))
            if sim >= self.cfg.few_shot_min_sim:
                scored.append((sim, rid, msgs))
        scored.sort(key=lambda x: (-x[0], x[1]))
        chosen = scored[: max(0, self.cfg.few_shot_k)]
        out: list[dict] = []
        for _, _, msgs in reversed(chosen):   # most similar example last (closest to the real turn)
            out.extend(msgs)
        return out, [(rid, round(sim, 3)) for sim, rid, _ in chosen]

    # ------------------------------------------------------------------ #
    def transcribe_all(self, sample: dict, split_dir: Path, oracle: bool) -> list[str]:
        texts = []
        for audio_rel, oracle_text in transcript_targets(sample):
            if oracle and oracle_text is not None:
                texts.append(oracle_text)
            else:
                texts.append(self.asr.transcribe(split_dir / audio_rel))
        return texts

    # ------------------------------------------------------------------ #
    def run(self, sample: dict, env: Any, split_dir: Path, oracle: bool = False) -> dict:
        cfg = self.cfg
        t0 = time.monotonic()
        log: dict[str, Any] = {"id": sample.get("id"), "steps": [], "notes": [], "timing": {}}
        if hasattr(self.llm, "reset"):
            self.llm.reset()
        context = sample.get("context") or {}

        # 1. ASR
        texts = self.transcribe_all(sample, split_dir, oracle)
        transcript = texts[-1]
        log["transcripts"] = texts
        log["timing"]["asr"] = round(time.monotonic() - t0, 3)

        # 2. Tools, hints, prompt
        tools = resolve_installed_tools(sample, self.catalog)
        tool_map = {t["name"]: t for t in tools}
        hints = build_hints(transcript, context.get("current_time") or "")
        log["hints"] = hints
        history_text = render_history(sample.get("history") or [], texts[:-1])
        user_msg = build_user_message(context, history_text, transcript, hints, render_tools(tools))
        shots, retrieved = self.retrieve_few_shot(transcript)
        if retrieved:
            log["few_shot"] = retrieved
        messages: list[dict] = [{"role": "system", "content": system_prompt(cfg.prompt_extras)}] + shots + \
                               [{"role": "user", "content": user_msg}]
        schema = action_schema(tools, cfg.use_thought) if cfg.use_guided_json else None

        store = EntityStore()
        store.add_context(context)
        # IDs returned by tools in HISTORY are valid and gold reuses them without re-reading (train: 311 vs 2)
        for hname, hres in history_tool_results(sample):
            if _result_ok(hres):
                store.add_result(hname, hres)
        for turn in sample.get("history") or []:
            for c in (turn.get("tool_calls") or []) if isinstance(turn, dict) else []:
                store.add_used_arguments(c.get("arguments") or {})
        # amounts said or shown earlier in words ("mỗi người 180 nghìn" in a note or an assistant reply) also count as known
        def _strs(o):
            if isinstance(o, str):
                yield o
            elif isinstance(o, dict):
                for x in o.values():
                    yield from _strs(x)
            elif isinstance(o, list):
                for x in o:
                    yield from _strs(x)
        for txt in list(texts[:-1]) + [t.get("text") or "" for t in (sample.get("history") or []) if isinstance(t, dict)] + \
                [s for t in (sample.get("history") or []) if isinstance(t, dict) for s in _strs(t.get("tool_results") or [])]:
            for ns in extract_numbers(txt or ""):
                try:
                    v = ns.int_value
                except Exception:
                    continue
                store.numbers.add(v)
                if 1 <= v <= 999:
                    store.numbers.add(v * 1000)   # "180" next to "nghìn"/"k" written apart
        text_has_number = bool(extract_numbers(transcript)) or any(ch.isdigit() for ch in transcript)
        pending_ask: Optional[tuple[str, str]] = None   # (tool_name, param) to ask about as the LAST call
        executed: set[str] = set()
        called_tools: set[str] = set()
        repairs = 0
        finished_reason = "max_steps"

        # 3. Loop
        for step in range(cfg.max_steps):
            elapsed = time.monotonic() - t0
            if elapsed > cfg.sample_budget_s:
                finished_reason = "budget"
                break
            ts = time.monotonic()
            # context guard: if the few-shot block would push the prompt past max_model_len, drop it (real turn first)
            if shots and hasattr(self.llm, "count_tokens"):
                try:
                    n_tok = self.llm.count_tokens(messages)
                except Exception:
                    n_tok = 0
                if n_tok > cfg.max_model_len - cfg.max_tokens - 256:
                    del messages[1:1 + len(shots)]
                    log["notes"].append(f"few-shot dropped at step {step}: prompt {n_tok} tokens")
                    shots = []
            try:
                raw = self.llm.chat(messages, schema)
            except Exception as e:  # pragma: no cover
                log["notes"].append(f"llm error: {e}")
                finished_reason = "llm_error"
                break
            step_log: dict[str, Any] = {"step": step, "raw": raw, "llm_s": round(time.monotonic() - ts, 3),
                                        "prompt_tokens": getattr(self.llm, "last_prompt_tokens", None),
                                        "new_tokens": getattr(self.llm, "last_new_tokens", None)}
            log["steps"].append(step_log)

            action = parse_action(raw)
            if action is None:
                repairs += 1
                step_log["event"] = "unparseable"
                if repairs > cfg.max_repairs:
                    finished_reason = "unparseable"
                    break
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "Trả lời không phải JSON hợp lệ. Trả lời lại đúng định dạng {\"thought\":..., \"name\":..., \"arguments\":{...}}."})
                continue

            name = action["name"]
            if name == FINISH:
                finished_reason = "finish"
                step_log["event"] = "finish"
                break
            if name not in tool_map:
                repairs += 1
                step_log["event"] = "unknown_tool"
                if repairs > cfg.max_repairs:
                    finished_reason = "unknown_tool"
                    break
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": f"Tool '{name}' không có trong TOOLS. Chỉ dùng tool trong danh sách, hoặc finish."})
                continue

            tool = tool_map[name]
            args, notes = ground_arguments(tool, action.get("arguments") or {}, store, context, cfg.snap_threshold)

            # 3a. read-before-write: fetch IDs the write needs if that app was never read
            if cfg.auto_read_before_write and tool.get("kind") == "write":
                for pname, ref in ref_params(tool).items():
                    app = ref.split(".")[0]
                    if pname in args and not store.has_app(app):
                        rt = find_read_tool_for_ref(ref, (tool["parameters"]["properties"] or {}).get(pname, {}), tools)
                        # the auto-read is called with no arguments: a read tool with required parameters (e.g.
                        # grocery.search needs query) would be rejected by the env, logged and cost precision
                        if rt and rt not in called_tools and rt in tool_map and \
                                not ((tool_map[rt].get("parameters") or {}).get("required")):
                            res = self._execute(env, rt, {}, store, executed, called_tools)
                            step_log.setdefault("auto_reads", []).append(rt)
                            messages.append({"role": "user", "content": "Hệ thống đã tự gọi tool đọc để lấy ID.\n" +
                                             render_result(rt, res, cfg.result_max_chars)})
                args, notes2 = ground_arguments(tool, args, store, context, cfg.snap_threshold)
                notes += notes2
            if name == ASK_TOOL_NAME:
                # gold: schema names, one param, schema order
                args["params"] = sort_ask_params(tool_map.get(args.get("tool_name")), args.get("params"))
            step_log["grounding"] = notes

            # 3a'. hallucinated ID guard: an x-ref value that is not among the IDs returned for that collection
            # would run as {"ok": true, "done": false} and still count as a wrong call -> challenge it instead
            bad_ids = []
            for pname, ref in ref_params(tool).items():
                v = args.get(pname)
                if isinstance(v, str) and v and store.has_app(ref.split(".")[0]):
                    known = [rid for rid, _ in store.ids_for(ref)]
                    if known and v not in known:
                        bad_ids.append((pname, v, known))
            # 3a''. gold convention guards -> deferred ask (gold: do everything else first, ask once, last)
            if cfg.ask_guards and tool.get("kind") == "write" and not bad_ids:
                reason = None
                for pname, ref in ref_params(tool).items():
                    v = args.get(pname)
                    if isinstance(v, str) and v and v not in store.resolved_ids and store.is_ambiguous(v) \
                            and not store.disambiguated(v, " ".join(t for t in texts if t)):
                        reason = (pname, f"có từ 2 liên hệ/mục cùng tên ứng với {pname}='{v}', không được đoán")
                        break
                if reason is None and isinstance(args.get("amount"), int) and not isinstance(args.get("amount"), bool) \
                        and not text_has_number and args["amount"] not in store.numbers:
                    reason = ("amount", "người dùng không nói số tiền và không có số tiền nào trong kết quả tool/HISTORY")
                if reason is not None:
                    pname, why = reason
                    if pending_ask is None:
                        pending_ask = (name, pname)
                    repairs += 1
                    step_log["event"] = "deferred_ask"
                    step_log["deferred_ask"] = [name, pname, why]
                    if repairs > cfg.max_repairs:
                        finished_reason = "deferred_ask"
                        break
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content": f"Không gọi {name}: {why}. Hệ thống sẽ tự gọi sys.ask_missing_info về "
                                     f"{pname} ở bước cuối. Hãy làm các việc còn lại của yêu cầu (nếu có), không gọi lại {name}, rồi finish."})
                    continue
            if bad_ids:
                repairs += 1
                step_log["event"] = "unknown_id"
                step_log["bad_ids"] = [(p_, v_) for p_, v_, _ in bad_ids]
                if repairs > cfg.max_repairs:
                    finished_reason = "unknown_id"
                    break
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "ID không có trong kết quả tool đọc: " +
                                 "; ".join(f"{p_}='{v_}' (ID hợp lệ: {', '.join(k_[:20])})" for p_, v_, k_ in bad_ids) +
                                 ". Dùng đúng ID trong danh sách, hoặc gọi tool đọc phù hợp, hoặc sys.ask_missing_info."})
                continue

            # 3b. duplicate guard
            key = _canon(name, args)
            if key in executed:
                repairs += 1
                step_log["event"] = "duplicate"
                if repairs > cfg.max_repairs:
                    finished_reason = "duplicate"
                    break
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": f"Tool {name} với đúng tham số này đã được gọi rồi. Không lặp lại; làm việc tiếp theo hoặc finish."})
                continue

            # 3c. validate before executing (nothing is executed by validate)
            if hasattr(env, "validate"):
                try:
                    v = env.validate(name, args)
                except Exception as e:  # pragma: no cover
                    v = {"ok": False, "error": str(e)}
                if isinstance(v, dict) and v.get("ok") is False:
                    repairs += 1
                    step_log["event"] = "invalid"
                    step_log["error"] = v.get("error")
                    if repairs > cfg.max_repairs:
                        finished_reason = "invalid"
                        break
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content": f"Tham số của {name} không hợp lệ: {v.get('error')}. "
                                     "Sửa lại tham số; nếu thiếu thông tin thì gọi sys.ask_missing_info."})
                    continue

            # 3d. execute
            res = self._execute(env, name, args, store, executed, called_tools)
            store.add_used_arguments(args)
            step_log["call"] = {"name": name, "arguments": args}
            step_log["ok"] = _result_ok(res)   # ok:false = schema error; done:false = ran but did nothing
            shown = dict(action)
            shown["arguments"] = args
            messages.append({"role": "assistant", "content": json.dumps(shown, ensure_ascii=False)})
            result_text = render_result(name, res, cfg.result_max_chars)
            if cfg.result_time_hints and step_log["ok"]:
                extra = result_time_hints(res, transcript)
                if extra:
                    result_text += "\nGỢI Ý TÍNH GIỜ (đã tính sẵn từ kết quả trên, dùng nguyên văn):\n" + "\n".join(f"- {x}" for x in extra)
                    step_log["result_hints"] = extra
            messages.append({"role": "user", "content": result_text +
                             "\n\nHành động tiếp theo (hoặc finish nếu đã làm xong mọi việc)?"})
            if name == ASK_TOOL_NAME:
                finished_reason = "asked"
                break
            if not step_log["ok"]:
                repairs += 1
                if repairs > cfg.max_repairs:
                    finished_reason = "call_errors"
                    break

        if pending_ask is not None and ASK_TOOL_NAME not in called_tools and ASK_TOOL_NAME in tool_map:
            ask_args = {"tool_name": pending_ask[0], "params": sort_ask_params(tool_map.get(pending_ask[0]), [pending_ask[1]])}
            self._execute(env, ASK_TOOL_NAME, ask_args, store, executed, called_tools)
            log["steps"].append({"step": "final", "event": "deferred_ask_executed", "call": {"name": ASK_TOOL_NAME, "arguments": ask_args}})
            finished_reason = "asked"
        log["finished_reason"] = finished_reason
        log["tool_calls"] = _safe(lambda: env.get_tool_calls(), [])
        log["state_hash"] = _safe(lambda: env.get_state_hash(), "")
        log["timing"]["total"] = round(time.monotonic() - t0, 3)
        return log

    # ------------------------------------------------------------------ #
    @staticmethod
    def _execute(env, name: str, args: dict, store: EntityStore, executed: set, called_tools: set):
        try:
            res = env.call(name, args)
        except Exception as e:  # the env should return ok:false, but never crash the sample
            res = {"ok": False, "error": f"exception: {e}"}
        executed.add(_canon(name, args))
        called_tools.add(name)
        if _result_ok(res):
            store.add_result(name, res)
        return res


def _result_ok(res) -> bool:
    """zaic_tools: {"ok": false} = schema error (call rejected); {"ok": true, "done": false} = ran, changed nothing."""
    return isinstance(res, dict) and res.get("ok", True) is not False and res.get("done", True) is not False


def _safe(fn, default):
    try:
        return fn()
    except Exception:
        return default
