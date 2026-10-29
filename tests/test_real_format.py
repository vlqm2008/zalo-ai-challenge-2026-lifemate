"""Tests that capture the REAL data format (data/README.md 2.1, zaic_tools semantics) with the mock env."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifemate.agent import Agent  # noqa: E402
from lifemate.asr import FakeASR  # noqa: E402
from lifemate.config import Config  # noqa: E402
from lifemate.data import (history_tool_results, load_catalog, resolve_installed_tools, sort_ask_params,  # noqa: E402
                           transcript_targets)
from lifemate.llm import FakeLLM  # noqa: E402
from lifemate.mock_env import MOCK_CATALOG, MockToolEnv  # noqa: E402
from lifemate.prompts import render_context, render_history  # noqa: E402

# Shape copied from train.jsonl (values invented): history user turns carry audio + audio_text (train only),
# assistant turns carry text + tool_calls + tool_results; context.location is {lat, lng}.
SAMPLE = {
    "id": "train_9999",
    "audio": "audio/train_9999/train_9999.wav",
    "audio_text": "Tắt đèn bếp giúp tôi.",
    "context": {
        "current_time": "2026-11-16T17:00:00+07:00",
        "timezone": "Asia/Ho_Chi_Minh",
        "location": {"lat": 10.7769, "lng": 106.7009},
        "user_profile": {"name": "Anh Nam", "home_address": "12 Nguyễn Huệ, Quận 1, TP.HCM",
                         "work_address": "Khu Công nghệ cao, TP. Thủ Đức, TP.HCM", "city": "TP.HCM"},
        "installed_apps": {"smart_home": {"tools": ["smart_home.get_status", "smart_home.light_set", "smart_home.climate_set"]},
                           "banking": {"tools": ["banking.transfer", "banking.get_balance"]},
                           "contacts": {"tools": ["contacts.list"]}},
    },
    "history": [
        {"role": "user", "audio": "audio/train_9999/train_9999_u1.wav", "audio_text": "Nhà có những đèn nào?"},
        {"role": "assistant", "text": "Có 3 đèn.",
         "tool_calls": [{"name": "smart_home.get_status", "arguments": {}}],
         "tool_results": [{"ok": True, "devices": [
             {"id": "light_living", "name": "đèn phòng khách", "type": "light", "room": "phòng khách", "state": {"power": "on"}},
             {"id": "light_kitchen", "name": "đèn bếp", "type": "light", "room": "bếp", "state": {"power": "on"}}]}]},
    ],
}

PUBLIC_SAMPLE = {  # public_test: no audio_text anywhere
    "id": "public_9999", "audio": "audio/public_9999/public_9999.wav", "context": SAMPLE["context"],
    "history": [{"role": "user", "audio": "audio/public_9999/public_9999_u1.wav"},
                {"role": "assistant", "text": "ok"}],
}


def make_catalog(tmp_path):
    p = tmp_path / "cat.json"
    p.write_text(json.dumps(MOCK_CATALOG, ensure_ascii=False), encoding="utf-8")
    return load_catalog(p)


def test_transcript_targets_follow_history_order():
    t = transcript_targets(SAMPLE)
    assert t == [("audio/train_9999/train_9999_u1.wav", "Nhà có những đèn nào?"),
                 ("audio/train_9999/train_9999.wav", "Tắt đèn bếp giúp tôi.")]
    t = transcript_targets(PUBLIC_SAMPLE)
    assert t == [("audio/public_9999/public_9999_u1.wav", None), ("audio/public_9999/public_9999.wav", None)]


def test_history_tool_results_and_render():
    assert history_tool_results(SAMPLE) == [("smart_home.get_status", SAMPLE["history"][1]["tool_results"][0])]
    text = render_history(SAMPLE["history"], ["Nhà có những đèn nào?"])
    lines = text.splitlines()
    assert json.loads(lines[0]) == {"role": "user", "text": "Nhà có những đèn nào?"}
    a = json.loads(lines[1])
    assert a["role"] == "assistant" and a["tool_calls"][0]["name"] == "smart_home.get_status"
    assert "light_kitchen" in lines[1] and "audio/" not in text
    # public_test: ASR transcript replaces the audio path
    text = render_history(PUBLIC_SAMPLE["history"], ["xin chào"])
    assert json.loads(text.splitlines()[0]) == {"role": "user", "text": "xin chào"}


def test_render_context_shows_real_keys():
    out = render_context(SAMPLE["context"])
    for key in ("current_time", "timezone", "lat", "lng", "home_address", "work_address", "city", "installed_apps"):
        assert key in out
    assert "smart_home.get_status" not in out  # tool lists belong to TOOLS, not CONTEXT


def test_id_from_history_is_used_without_re_reading(tmp_path):
    cfg = Config(llm_backend="fake", asr_backend="fake", use_guided_json=False)
    script = [{"thought": "", "name": "smart_home.light_set", "arguments": {"device_id": "đèn bếp", "power": "off"}},
              {"thought": "", "name": "finish", "arguments": {}}]
    agent = Agent(cfg, FakeLLM(script), FakeASR(), make_catalog(tmp_path))
    env = MockToolEnv("train", SAMPLE["id"])
    log = agent.run(SAMPLE, env, tmp_path, oracle=True)
    assert [c["name"] for c in log["tool_calls"]] == ["smart_home.light_set"]      # no extra get_status
    assert log["tool_calls"][0]["arguments"]["device_id"] == "light_kitchen"         # snapped from history


def test_ask_params_schema_order_and_dedupe(tmp_path):
    catalog = make_catalog(tmp_path)
    tool = catalog["_tools"]["banking.transfer"]   # properties: contact_id, amount, note
    assert sort_ask_params(tool, ["amount", "contact_id", "amount"]) == ["contact_id", "amount"]
    assert sort_ask_params(tool, "amount") == ["amount"]
    cfg = Config(llm_backend="fake", asr_backend="fake", use_guided_json=False)
    script = [{"thought": "", "name": "sys.ask_missing_info", "arguments": {"tool_name": "banking.transfer", "params": ["amount", "contact_id"]}}]
    agent = Agent(cfg, FakeLLM(script), FakeASR(), catalog)
    log = agent.run(SAMPLE, MockToolEnv("train", SAMPLE["id"]), tmp_path, oracle=True)
    assert log["tool_calls"] == [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "banking.transfer", "params": ["contact_id", "amount"]}}]
    assert log["finished_reason"] == "asked"


def test_done_false_is_fed_back_as_failure(tmp_path, monkeypatch):
    cfg = Config(llm_backend="fake", asr_backend="fake", use_guided_json=False)
    seen = []
    script = lambda messages: (seen.append(messages[-1]["content"]) or  # noqa: E731
                               json.dumps([{"name": "smart_home.light_set", "arguments": {"device_id": "light_kitchen", "power": "off"}},
                                           {"name": "finish", "arguments": {}}][min(len(seen) - 1, 1)], ensure_ascii=False))
    agent = Agent(cfg, FakeLLM(script), FakeASR(), make_catalog(tmp_path))
    env = MockToolEnv("train", SAMPLE["id"])
    orig = env.call

    def call(name, args):
        r = orig(name, args)
        return {"ok": True, "done": False, "message": "không tìm thấy device_id"} if name == "smart_home.light_set" else r
    monkeypatch.setattr(env, "call", call)
    log = agent.run(SAMPLE, env, tmp_path, oracle=True)
    assert log["steps"][0]["ok"] is False
    assert "không tìm thấy device_id" in seen[-1]        # the LLM saw the message
    assert len(log["tool_calls"]) == 1                    # the rejected-in-effect call is still in the submission


def test_result_time_hints_and_few_shot_plumbing(tmp_path):
    from lifemate.normalize_vi import result_time_hints
    res = {"ok": True, "events": [{"id": "evt_01", "start": "2026-11-25T08:00:00+07:00", "end": "2026-11-25T09:00:00+07:00"}]}
    lines = result_time_hints(res, "dời lịch đá bóng sáng mai muộn hai tiếng")
    assert any("2026-11-25T10:00:00+07:00" in l for l in lines) and any("2026-11-25T11:00:00+07:00" in l for l in lines)
    assert result_time_hints(res, "tắt đèn bếp") == []            # no span in the request -> nothing
    # few-shot file: its turns are inserted between the system prompt and the real user turn
    fs = tmp_path / "fs.jsonl"
    fs.write_text(json.dumps({"id": "x", "messages": [{"role": "user", "content": "ví dụ"}, {"role": "assistant", "content": '{"name": "finish", "arguments": {}}'}]}) + "\n", encoding="utf-8")
    cfg = Config(llm_backend="fake", asr_backend="fake", use_guided_json=False, few_shot_path=str(fs), result_time_hints=True)
    llm = FakeLLM([{"name": "finish", "arguments": {}}])
    agent = Agent(cfg, llm, FakeASR(), make_catalog(tmp_path))
    agent.run(SAMPLE, MockToolEnv("train", SAMPLE["id"]), tmp_path, oracle=True)
    roles = [m["role"] for m in llm.calls[0]]
    assert roles[:3] == ["system", "user", "assistant"] and roles[-1] == "user" and llm.calls[0][1]["content"] == "ví dụ"


def test_timestamps_are_never_snapped_and_empty_args_dropped(tmp_path):
    from lifemate.grounding import EntityStore, ground_arguments
    catalog = make_catalog(tmp_path)
    store = EntityStore()
    store.add_context(SAMPLE["context"])
    store.add_result("alarm.list", {"ok": True, "alarms": [{"id": "alm_01", "time": "2026-11-23T09:00:00+07:00", "label": "họp sớm"}]})
    tool = {"name": "alarm.create", "kind": "write", "parameters": {"type": "object", "properties": {
        "time": {"type": "string"}, "label": {"type": "string", "x-match": "fuzzy"}, "repeat": {"type": "string"}, "tags": {"type": "array"}}}}
    args, notes = ground_arguments(tool, {"time": "2026-11-23T06:00:00+07:00", "repeat": "", "tags": []}, store, SAMPLE["context"])
    assert args == {"time": "2026-11-23T06:00:00+07:00"}          # 06:00 kept (measured bug: was snapped to 09:00), empties dropped
    # names/addresses with a near-identical known string are still snapped
    tool2 = {"name": "ride.book", "kind": "write", "parameters": {"type": "object", "properties": {"to": {"type": "string"}}}}
    args, _ = ground_arguments(tool2, {"to": "12 Nguyen Hue, Quan 1, TP.HCM"}, store, SAMPLE["context"])
    assert args["to"] == "12 Nguyễn Huệ, Quận 1, TP.HCM"
    from lifemate.prompts import system_prompt
    assert "KHÔNG tự thêm" in system_prompt(True) and "KHÔNG tự thêm" not in system_prompt(False)
    assert '{"thought"' in system_prompt(False)


def test_profile_place_phrases_and_unknown_id_guard(tmp_path):
    from lifemate.grounding import EntityStore, ground_arguments
    store = EntityStore()
    tool = {"name": "ride.book", "kind": "write", "parameters": {"type": "object", "properties": {"from": {"type": "string", "x-match": "fuzzy"}, "to": {"type": "string"}}}}
    args, _ = ground_arguments(tool, {"from": "công ty", "to": "Nhà"}, store, SAMPLE["context"])
    assert args == {"from": "Khu Công nghệ cao, TP. Thủ Đức, TP.HCM", "to": "12 Nguyễn Huệ, Quận 1, TP.HCM"}
    # an ID that is not in the read result is challenged, not executed
    cfg = Config(llm_backend="fake", asr_backend="fake", use_guided_json=False)
    script = [{"thought": "", "name": "smart_home.get_status", "arguments": {}},
              {"thought": "", "name": "smart_home.light_set", "arguments": {"device_id": "light_garage_001", "power": "off"}},
              {"thought": "", "name": "smart_home.light_set", "arguments": {"device_id": "light_kitchen", "power": "off"}},
              {"thought": "", "name": "finish", "arguments": {}}]
    agent = Agent(cfg, FakeLLM(script), FakeASR(), make_catalog(tmp_path))
    log = agent.run({**SAMPLE, "history": []}, MockToolEnv("train", SAMPLE["id"]), tmp_path, oracle=True)
    assert [c["name"] for c in log["tool_calls"]] == ["smart_home.get_status", "smart_home.light_set"]
    assert log["tool_calls"][1]["arguments"]["device_id"] == "light_kitchen"
    assert any(st.get("event") == "unknown_id" for st in log["steps"])


def test_person_query_normalisation():
    from lifemate.grounding import normalize_person_query, person_query_params
    assert normalize_person_query("anh Tuấn") == "Tuấn"
    assert normalize_person_query("bạn minh") == "Minh"
    assert normalize_person_query("bác ba") == "Bác Ba"
    assert normalize_person_query("sếp dũng") == "Sếp Dũng"
    assert normalize_person_query("cô giáo mai") == "Cô giáo Mai"
    assert normalize_person_query("bố em") == "Bố"
    assert normalize_person_query("Mẹ") == "Mẹ"
    assert normalize_person_query("0912345678") == "0912345678"
    tool = {"name": "contacts.search", "kind": "read", "app": "contacts", "description": "Tìm liên hệ theo tên, quan hệ hoặc số điện thoại",
            "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Tên, quan hệ (mẹ, vợ, sếp) hoặc số điện thoại"}}}}
    assert person_query_params(tool) == {"query"}
    assert person_query_params({"name": "news.search", "kind": "read", "app": "news", "description": "Tìm tin",
                                "parameters": {"properties": {"query": {"type": "string"}}}}) == set()


def test_ask_guards_defer_ambiguous_contact_and_ungrounded_amount(tmp_path):
    from lifemate.mock_env import DEFAULT_STATE
    import copy
    state = copy.deepcopy(DEFAULT_STATE)
    state["contacts"]["contacts"] = [{"id": "c_01", "name": "Lan", "relation": "vợ", "phone": "0901"},
                                     {"id": "c_02", "name": "Lan", "relation": "đồng nghiệp", "phone": "0902"}]
    cfg = Config(llm_backend="fake", asr_backend="fake", use_guided_json=False, ask_guards=True)
    sample = {**SAMPLE, "history": [], "audio_text": "Chuyển ví cho Lan ít tiền nhé, rồi bật điều hòa phòng khách."}
    script = [{"thought": "", "name": "contacts.list", "arguments": {}},
              {"thought": "", "name": "banking.transfer", "arguments": {"contact_id": "c_01", "amount": 100000}},   # ambiguous + invented
              {"thought": "", "name": "smart_home.get_status", "arguments": {}},
              {"thought": "", "name": "smart_home.climate_set", "arguments": {"device_id": "ac_living_room", "mode": "cool"}},
              {"thought": "", "name": "finish", "arguments": {}}]
    agent = Agent(cfg, FakeLLM(script), FakeASR(), make_catalog(tmp_path))
    log = agent.run(sample, MockToolEnv("train", sample["id"], state=state), tmp_path, oracle=True)
    names = [c["name"] for c in log["tool_calls"]]
    assert names == ["contacts.list", "smart_home.get_status", "smart_home.climate_set", "sys.ask_missing_info"]
    assert log["tool_calls"][-1]["arguments"] == {"tool_name": "banking.transfer", "params": ["contact_id"]}
    assert log["finished_reason"] == "asked"
    # a unique contact with the amount given in the request is executed normally
    state["contacts"]["contacts"] = [{"id": "c_01", "name": "Lan", "relation": "vợ", "phone": "0901"}]
    sample = {**SAMPLE, "history": [], "audio_text": "Chuyển ví cho Lan năm trăm nghìn."}
    script = [{"thought": "", "name": "contacts.list", "arguments": {}},
              {"thought": "", "name": "banking.transfer", "arguments": {"contact_id": "c_01", "amount": 500000}},
              {"thought": "", "name": "finish", "arguments": {}}]
    agent = Agent(cfg, FakeLLM(script), FakeASR(), make_catalog(tmp_path))
    log = agent.run(sample, MockToolEnv("train", sample["id"], state=state), tmp_path, oracle=True)
    assert [c["name"] for c in log["tool_calls"]] == ["contacts.list", "banking.transfer"]


def test_knn_few_shot_retrieval(tmp_path):
    idx = tmp_path / "idx.jsonl"
    rows = [{"id": "a", "text": "Tắt đèn bếp giúp tôi nhé", "messages": [{"role": "user", "content": "ví dụ A"}, {"role": "assistant", "content": '{"name": "finish", "arguments": {}}'}]},
            {"id": "b", "text": "Chuyển tiền cho mẹ năm trăm nghìn", "messages": [{"role": "user", "content": "ví dụ B"}, {"role": "assistant", "content": '{"name": "finish", "arguments": {}}'}]}]
    idx.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    cfg = Config(llm_backend="fake", asr_backend="fake", use_guided_json=False, few_shot_path=str(idx), few_shot_mode="knn", few_shot_k=1)
    llm = FakeLLM([{"name": "finish", "arguments": {}}])
    agent = Agent(cfg, llm, FakeASR(), make_catalog(tmp_path))
    log = agent.run(SAMPLE, MockToolEnv("train", SAMPLE["id"]), tmp_path, oracle=True)   # "Tắt đèn bếp giúp tôi."
    assert [m["content"] for m in llm.calls[0] if m["role"] == "user"][0] == "ví dụ A"
    assert log["few_shot"][0][0] == "a"
    # nothing similar -> no example
    cfg.few_shot_min_sim = 0.99
    agent = Agent(cfg, FakeLLM([{"name": "finish", "arguments": {}}]), FakeASR(), make_catalog(tmp_path))
    log = agent.run(SAMPLE, MockToolEnv("train", SAMPLE["id"]), tmp_path, oracle=True)
    assert "few_shot" not in log


def test_few_shot_dropped_when_prompt_too_long(tmp_path):
    idx = tmp_path / "idx.jsonl"
    idx.write_text(json.dumps({"id": "a", "text": "Tắt đèn bếp giúp tôi nhé", "messages": [{"role": "user", "content": "ví dụ A"}, {"role": "assistant", "content": '{"name": "finish", "arguments": {}}'}]}) + "\n", encoding="utf-8")
    cfg = Config(llm_backend="fake", asr_backend="fake", use_guided_json=False, few_shot_path=str(idx), few_shot_mode="knn", few_shot_k=1, max_model_len=16384)

    class CountingFake(FakeLLM):
        def count_tokens(self, messages):
            return 20000   # pretend the prompt overflows
    llm = CountingFake([{"name": "finish", "arguments": {}}])
    agent = Agent(cfg, llm, FakeASR(), make_catalog(tmp_path))
    log = agent.run(SAMPLE, MockToolEnv("train", SAMPLE["id"]), tmp_path, oracle=True)
    assert [m["role"] for m in llm.calls[0]] == ["system", "user"] and "ví dụ A" not in json.dumps(llm.calls[0], ensure_ascii=False)
    assert any("few-shot dropped" in n for n in log["notes"])
