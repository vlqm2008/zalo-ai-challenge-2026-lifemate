"""Loading samples / tool catalog and resolving the tools a sample may call."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator, Optional

ASK_TOOL_NAME = "sys.ask_missing_info"

# Fallback definition if the catalog does not list the meta tool explicitly.
ASK_TOOL_FALLBACK = {
    "name": ASK_TOOL_NAME,
    "kind": "meta",
    "app": "sys",
    "description": "Hỏi lại người dùng khi thiếu thông tin bắt buộc để gọi một tool. "
                   "tool_name là tool định gọi, params là danh sách tham số còn thiếu.",
    "parameters": {
        "type": "object",
        "properties": {
            "tool_name": {"type": "string"},
            "params": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["tool_name", "params"],
    },
    "returns": "",
}


def load_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_catalog(path: str | Path) -> dict:
    cat = json.loads(Path(path).read_text(encoding="utf-8"))
    tools: dict[str, dict] = {}
    for app in cat.get("apps", []):
        for t in app.get("tools", []):
            t = dict(t)
            t.setdefault("app", app.get("id") or t["name"].split(".")[0])
            tools[t["name"]] = t
    # meta tool may live at top level, inside an app, or be absent
    for t in cat.get("tools", []) or []:
        tools.setdefault(t["name"], dict(t))
    if ASK_TOOL_NAME not in tools:
        tools[ASK_TOOL_NAME] = dict(ASK_TOOL_FALLBACK)
    cat["_tools"] = tools
    cat["_apps"] = {a.get("id"): a for a in cat.get("apps", [])}
    return cat


def installed_app_ids(sample: dict) -> list[str]:
    apps = (sample.get("context") or {}).get("installed_apps") or {}
    if isinstance(apps, dict):
        return list(apps.keys())
    if isinstance(apps, list):
        return [a if isinstance(a, str) else a.get("id") or a.get("name") for a in apps]
    return []


def resolve_installed_tools(sample: dict, catalog: dict) -> list[dict]:
    """Tools the sample may call, in installed_apps order (+ sys.ask_missing_info)."""
    apps = (sample.get("context") or {}).get("installed_apps") or {}
    all_tools = catalog["_tools"]
    out: list[dict] = []
    seen: set[str] = set()

    def add(name: str, inline: Optional[dict] = None):
        if name in seen:
            return
        t = all_tools.get(name)
        if t is None and inline:
            t = dict(inline)
            t.setdefault("app", name.split(".")[0])
        if t is None:
            return
        seen.add(name)
        out.append(t)

    if isinstance(apps, dict):
        for app_id, spec in apps.items():
            names = spec.get("tools") if isinstance(spec, dict) else spec
            if names:
                for n in names:
                    if isinstance(n, str):
                        add(n)
                    elif isinstance(n, dict) and "name" in n:
                        add(n["name"], n)
            else:  # app listed without tool names -> every catalog tool of that app
                for t in all_tools.values():
                    if t.get("app") == app_id:
                        add(t["name"])
    elif isinstance(apps, list):
        for app_id in installed_app_ids(sample):
            for t in all_tools.values():
                if t.get("app") == app_id:
                    add(t["name"])
    add(ASK_TOOL_NAME)
    return out


_TOOL_NAME_RE = re.compile(r"\b([a-z][a-z0-9_]*\.[a-z][a-z0-9_]*)\b")


def ref_params(tool: dict) -> dict[str, str]:
    """{param_name: x-ref value} for parameters that must be valid IDs."""
    props = (tool.get("parameters") or {}).get("properties") or {}
    return {k: v["x-ref"] for k, v in props.items() if isinstance(v, dict) and v.get("x-ref")}


def fuzzy_params(tool: dict) -> set[str]:
    props = (tool.get("parameters") or {}).get("properties") or {}
    return {k for k, v in props.items() if isinstance(v, dict) and v.get("x-match") == "fuzzy"}


def find_read_tool_for_ref(ref: str, param_schema: dict, tools: list[dict]) -> Optional[str]:
    """Which read tool returns the IDs an x-ref parameter needs.

    1. a tool name mentioned in the parameter description ("lấy từ smart_home.get_status")
    2. a read tool of the same app whose `returns` mentions the collection ("devices")
    3. the only read tool of that app, if there is exactly one
    """
    app, _, collection = ref.partition(".")
    by_name = {t["name"]: t for t in tools}
    for cand in _TOOL_NAME_RE.findall(param_schema.get("description") or ""):
        if cand in by_name and by_name[cand].get("kind") == "read":
            return cand
    reads = [t for t in tools if t.get("app") == app and t.get("kind") == "read"]
    for t in reads:
        if collection and collection in (t.get("returns") or "") + " " + (t.get("description") or ""):
            return t["name"]
    if len(reads) == 1:
        return reads[0]["name"]
    return None


def sanitize_schema(schema: Any) -> Any:
    """Drop x-* extension keywords so the schema is plain JSON Schema (for guided decoding)."""
    if isinstance(schema, dict):
        return {k: sanitize_schema(v) for k, v in schema.items() if not k.startswith("x-")}
    if isinstance(schema, list):
        return [sanitize_schema(v) for v in schema]
    return schema


def history_turns(sample: dict) -> list[dict]:
    h = sample.get("history") or []
    return h if isinstance(h, list) else []


def history_tool_results(sample: dict) -> list[tuple[str, Any]]:
    """(tool_name, result) pairs from the assistant turns of `history`.

    Real format (data README 2.1): assistant turns are {"role": "assistant", "text": ..., "tool_calls": [...],
    "tool_results": [...]} and those calls were already applied to the sample's state, so the IDs they
    returned are valid and gold reuses them WITHOUT calling the read tool again (train: 311 vs 2).
    """
    out: list[tuple[str, Any]] = []
    for turn in history_turns(sample):
        if not isinstance(turn, dict) or turn.get("role") == "user":
            continue
        calls = turn.get("tool_calls") or []
        results = turn.get("tool_results") or []
        for i, c in enumerate(calls):
            if isinstance(c, dict) and c.get("name"):
                out.append((c["name"], results[i] if i < len(results) else None))
    return out


def _as_text(v: Any) -> Optional[str]:
    if isinstance(v, str):
        return v
    if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
        return " ".join(v)
    return None


def transcript_targets(sample: dict) -> list[tuple[str, Optional[str]]]:
    """(audio_path, oracle_text) for every user turn: history user turns first (in order), latest last.

    Real format: history user turns are {"role": "user", "audio": "audio/<id>/<id>_u1.wav", "audio_text": ...}
    (audio_text only in train; public_test has just the audio), and the latest request is sample["audio"]
    with sample["audio_text"] (train only). oracle_text is None when no transcript is given -> ASR.
    """
    targets: list[tuple[str, Optional[str]]] = []
    for turn in history_turns(sample):
        if isinstance(turn, dict) and turn.get("role", "user") == "user" and turn.get("audio"):
            targets.append((turn["audio"], _as_text(turn.get("audio_text"))))
    targets.append((sample["audio"], _as_text(sample.get("audio_text"))))
    return targets


def sort_ask_params(tool: Optional[dict], params: Any) -> list[str]:
    """sys.ask_missing_info.params: schema names only, deduplicated, in the schema's property order.

    Gold convention (train): always exactly one param, always the schema name, always the last call.
    """
    props = list((((tool or {}).get("parameters") or {}).get("properties") or {}).keys())
    seen: list[str] = []
    for p in (params if isinstance(params, list) else [params]):
        if isinstance(p, str) and p.strip() and p.strip() not in seen:
            seen.append(p.strip())
    return sorted(seen, key=lambda p: (props.index(p) if p in props else len(props), 0))
