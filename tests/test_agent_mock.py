import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifemate.agent import Agent  # noqa: E402
from lifemate.asr import FakeASR  # noqa: E402
from lifemate.config import Config  # noqa: E402
from lifemate.data import load_catalog, resolve_installed_tools  # noqa: E402
from lifemate.llm import FakeLLM  # noqa: E402
from lifemate.mock_env import MOCK_CATALOG, MockToolEnv  # noqa: E402
from lifemate.prompts import action_schema  # noqa: E402
from lifemate.scorer import fuzzy_lookup_from_catalog, score_sample  # noqa: E402

SAMPLE = {
    "id": "s_000123",
    "audio": "audio/s_000123/s_000123.wav",
    "context": {
        "current_time": "2026-11-16T17:00:00+07:00",
        "timezone": "Asia/Ho_Chi_Minh",
        "location": "227 Nguyễn Văn Cừ, Quận 5, TP.HCM",
        "user_profile": {"home_address": "123 Trần Hưng Đạo, Quận 5, TP.HCM"},
        "installed_apps": {"smart_home": {"tools": ["smart_home.get_status", "smart_home.climate_set", "smart_home.light_set"]},
                           "ride": {"tools": ["ride.book"]}, "banking": {"tools": ["banking.transfer", "banking.get_balance"]},
                           "contacts": {"tools": ["contacts.list"]}},
    },
    "history": [],
    "audio_text": "Tôi sắp về nhà, bật điều hòa phòng khách 26 độ và đặt xe đón tôi.",
}

GOLD = [
    {"name": "smart_home.get_status", "arguments": {}},
    {"name": "smart_home.climate_set", "arguments": {"device_id": "ac_living_room", "mode": "cool", "temperature": 26}},
    {"name": "ride.book", "arguments": {"from": "227 Nguyễn Văn Cừ, Quận 5, TP.HCM", "to": "123 Trần Hưng Đạo, Quận 5, TP.HCM"}},
]


def make_catalog(tmp_path):
    p = tmp_path / "cat.json"
    p.write_text(json.dumps(MOCK_CATALOG, ensure_ascii=False), encoding="utf-8")
    return load_catalog(p)


def gold_hash():
    env = MockToolEnv("train", "x")
    for c in GOLD:
        env.call(c["name"], c["arguments"])
    return env.get_state_hash()


def run(script, tmp_path, cfg=None):
    cfg = cfg or Config(llm_backend="fake", asr_backend="fake", use_guided_json=False)
    catalog = make_catalog(tmp_path)
    agent = Agent(cfg, FakeLLM(script), FakeASR(), catalog)
    env = MockToolEnv("train", SAMPLE["id"])
    log = agent.run(SAMPLE, env, tmp_path, oracle=True)
    return log, catalog


def test_happy_path_matches_gold(tmp_path):
    script = [
        {"thought": "cần ID điều hòa", "name": "smart_home.get_status", "arguments": {}},
        {"thought": "bật điều hòa", "name": "smart_home.climate_set", "arguments": {"device_id": "ac_living_room", "mode": "cool", "temperature": "26"}},
        {"thought": "đặt xe về nhà", "name": "ride.book", "arguments": {"from": "227 Nguyễn Văn Cừ, Quận 5, TP.HCM", "to": "123 Trần Hưng Đạo, Quận 5, TP.HCM"}},
        {"thought": "xong", "name": "finish", "arguments": {}},
    ]
    log, catalog = run(script, tmp_path)
    s = score_sample("x", log["tool_calls"], log["state_hash"], GOLD, gold_hash(), fuzzy_lookup_from_catalog(catalog))
    assert s.score == 1.0, (s, log["tool_calls"])
    assert log["tool_calls"][1]["arguments"]["temperature"] == 26  # "26" coerced to int
    assert log["finished_reason"] == "finish"


def test_snap_device_name_and_auto_read(tmp_path):
    # LLM skips the read and uses the device NAME instead of its ID -> auto read + snap
    script = [
        {"thought": "", "name": "smart_home.climate_set", "arguments": {"device_id": "điều hòa phòng khách", "mode": "cool", "temperature": 26}},
        {"thought": "", "name": "ride.book", "arguments": {"from": "227 Nguyen Van Cu, Quan 5, TP.HCM", "to": "123 Trần Hưng Đạo, Quận 5, TP.HCM"}},
        {"thought": "", "name": "finish", "arguments": {}},
    ]
    log, catalog = run(script, tmp_path)
    names = [c["name"] for c in log["tool_calls"]]
    assert names == ["smart_home.get_status", "smart_home.climate_set", "ride.book"]
    assert log["tool_calls"][1]["arguments"]["device_id"] == "ac_living_room"
    assert log["tool_calls"][2]["arguments"]["from"] == "227 Nguyễn Văn Cừ, Quận 5, TP.HCM"  # diacritics restored
    s = score_sample("x", log["tool_calls"], log["state_hash"], GOLD, gold_hash(), fuzzy_lookup_from_catalog(catalog))
    assert s.score == 1.0, s


def test_invalid_then_repair_and_duplicate_guard(tmp_path):
    script = [
        {"thought": "", "name": "smart_home.get_status", "arguments": {}},
        {"thought": "", "name": "smart_home.get_status", "arguments": {}},                     # duplicate -> blocked
        {"thought": "", "name": "smart_home.light_set", "arguments": {"power": "off"}},        # invalid: missing device_id
        {"thought": "", "name": "smart_home.light_set", "arguments": {"device_id": "light_kitchen", "power": "off"}},
        {"thought": "", "name": "finish", "arguments": {}},
    ]
    log, _ = run(script, tmp_path)
    names = [c["name"] for c in log["tool_calls"]]
    assert names == ["smart_home.get_status", "smart_home.light_set"]
    assert any(st.get("event") == "duplicate" for st in log["steps"])
    assert any(st.get("event") == "invalid" for st in log["steps"])


def test_out_of_scope_and_ask(tmp_path):
    log, _ = run([{"thought": "không app nào", "name": "finish", "arguments": {}}], tmp_path)
    assert log["tool_calls"] == []
    script = [{"thought": "", "name": "sys.ask_missing_info", "arguments": {"tool_name": "banking.transfer", "params": ["amount"]}}]
    log, _ = run(script, tmp_path)
    assert log["tool_calls"] == [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "banking.transfer", "params": ["amount"]}}]
    assert log["finished_reason"] == "asked"


def test_action_schema_builds(tmp_path):
    catalog = make_catalog(tmp_path)
    tools = resolve_installed_tools(SAMPLE, catalog)
    schema = action_schema(tools, use_thought=True)
    names = {v["properties"]["name"]["const"] for v in schema["anyOf"]}
    assert "finish" in names and "sys.ask_missing_info" in names and "ride.book" in names
    assert "x-ref" not in json.dumps(schema)
