"""python test_score.py — checks the scoring rules on hand-made cases."""
from score import n_matched, sample_score, token_f1

FZ = {"sms.send": {"content"}}
send = lambda cid, text: {"name": "sms.send", "arguments": {"contact_id": cid, "content": text}}
search = {"name": "contacts.search", "arguments": {"query": "Lan"}}

assert token_f1("Mai CHẠY bộ lúc 9h", "mai chạy bộ lúc 9h nhé") > 0.6
assert token_f1("hủy lịch", "chạy bộ dời sang 9h") == 0.0
assert n_matched([send("c_01", "mai họp lúc 9h nhé")], [send("c_01", "mai họp lúc 9h")], FZ) == 1  # fuzzy passes
assert n_matched([send("c_02", "mai họp lúc 9h")], [send("c_01", "mai họp lúc 9h")], FZ) == 0  # wrong id
assert n_matched([{"name": "contacts.search", "arguments": {}}], [search], FZ) == 0  # missing argument
assert n_matched([search, search], [search], FZ) == 1  # one gold call matches at most once
# maximum matching, not greedy: pred[0] fits both gold calls, pred[1] only the first
a, b = send("c_01", "anh tới rồi"), send("c_01", "anh tới rồi nha em")
assert n_matched([send("c_01", "anh tới rồi nha"), send("c_01", "anh tới")], [a, b], FZ) == 2

g = [search, send("c_01", "hi")]
assert sample_score([], "h0", [], "h0", FZ) == 1.0  # out of scope, nothing called
assert sample_score([search], "h0", [], "h0", FZ) == 0.0  # out of scope, something called
assert sample_score(g, "h1", g, "h1", FZ) == 1.0
assert sample_score(g, "h0", g, "h1", FZ) == 0.0  # wrong final state
assert abs(sample_score(g + [search], "h1", g, "h1", FZ) - 0.8) < 1e-9  # P=2/3, R=1
assert abs(sample_score([search], "h1", g, "h1", FZ) - 2 / 3) < 1e-9  # P=1, R=1/2
print("ok")
