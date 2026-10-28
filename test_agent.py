"""python test_agent.py — runs the agent loop with a fake LLM that replays the gold calls; every train sample must come out exact."""
import json
import sys

import agent
from score import load_catalog, sample_score

sys.path.insert(0, str(agent.ROOT / "train"))
from zaic_tools import ToolEnv  # noqa: E402


class GoldLLM:
    """Emits the gold calls one per turn (a first read call twice, to exercise the duplicate guard), then a plain answer."""

    tok = type("Tok", (), {"apply_chat_template": staticmethod(lambda msgs, **kw: json.dumps(msgs, ensure_ascii=False))})

    def __init__(self, calls):
        self.calls, self.turn = calls, 0

    def __call__(self, prompt):
        assert "Current time:" in prompt
        if self.turn >= len(self.calls):
            return "Xong.", 0
        c = self.calls[self.turn]
        self.turn += 1
        block = f"<tool_call>\n{json.dumps(c, ensure_ascii=False)}\n</tool_call>"
        return block * (2 if self.turn == 1 and kind[c["name"]] == "read" else 1), 0


catalog = json.load(open(agent.ROOT / "train/train_tools_catalog.json", encoding="utf-8"))
kind, fuzzy = load_catalog(agent.ROOT)
NOISY = {"train_1696"}  # gold messages chat_6142, an ID no tool ever returned (the env answers "không tìm thấy"); the ID guard rejects it
for item in (i for i in map(json.loads, open(agent.ROOT / "train/train.jsonl", encoding="utf-8")) if i["id"] not in NOISY):
    gold = item["label"]["tool_calls"]
    env, ref = ToolEnv("train", item["id"], root=agent.ROOT), ToolEnv("train", item["id"], root=agent.ROOT)
    for c in gold:
        ref.call(c["name"], c["arguments"])
    agent.run_agent(item, env, catalog, GoldLLM(gold))
    assert sample_score(env.get_tool_calls(), env.get_state_hash(), gold, ref.get_state_hash(), fuzzy) == 1.0, item["id"]
print("ok")
