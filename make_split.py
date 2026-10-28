"""Split train ids into splits/{train,dev_seen,dev_unseen}.txt.

dev_unseen: every sample whose gold calls use a held-out app (stands in for the apps public/private add, never train on it).
dev_seen: a fixed random 10% of the rest. train: everything else.
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
# train-only apps, one per group, so the train part still covers every public_test app
HELD_OUT = {"home_security", "video_meeting", "subscriptions", "bike_rental", "medication", "news"}

data = [json.loads(l) for l in open(HERE / "train+public_test/train/train.jsonl", encoding="utf-8")]
unseen = [d["id"] for d in data if {c["name"].split(".")[0] for c in d["label"]["tool_calls"]} & HELD_OUT]
rest = sorted(d["id"] for d in data if d["id"] not in set(unseen))
random.Random(0).shuffle(rest)
k = len(rest) // 10

(HERE / "splits").mkdir(exist_ok=True)
for name, ids in {"dev_unseen": unseen, "dev_seen": rest[:k], "train": rest[k:]}.items():
    (HERE / f"splits/{name}.txt").write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")
    print(f"{name:10} {len(ids)}")
