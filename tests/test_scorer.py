import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifemate.scorer import aggregate, score_sample, token_f1  # noqa: E402

FUZZY = lambda name: {"message", "body"} if name in ("sms.send", "email.send") else set()  # noqa: E731

GOLD = [
    {"name": "smart_home.get_status", "arguments": {}},
    {"name": "smart_home.climate_set", "arguments": {"device_id": "ac_living_room", "mode": "cool", "temperature": 26}},
    {"name": "ride.book", "arguments": {"from": "227 Nguyễn Văn Cừ, Quận 5, TP.HCM", "to": "123 Trần Hưng Đạo, Quận 5, TP.HCM"}},
]


def test_perfect():
    s = score_sample("x", GOLD, "h", GOLD, "h", FUZZY)
    assert s.score == 1.0 and s.M == 3 and s.U == 0


def test_order_ignored_and_extra_call():
    sub = [GOLD[2], GOLD[1], GOLD[0], {"name": "smart_home.get_status", "arguments": {}}]
    s = score_sample("x", sub, "h", GOLD, "h", FUZZY)
    # duplicate read: matched at most once -> U=1, P=3/4, R=1 -> F1=0.857
    assert s.M == 3 and s.U == 1
    assert abs(s.score - (2 * 0.75 * 1 / 1.75)) < 1e-9


def test_missing_read():
    sub = [GOLD[1], GOLD[2]]
    s = score_sample("x", sub, "h", GOLD, "h", FUZZY)
    assert s.M == 2 and abs(s.score - 0.8) < 1e-9


def test_hash_mismatch_zero():
    s = score_sample("x", GOLD, "other", GOLD, "h", FUZZY)
    assert s.score == 0.0 and "hash_mismatch" in s.reasons


def test_no_match_zero():
    sub = [{"name": "smart_home.light_set", "arguments": {"device_id": "x", "power": "on"}}]
    s = score_sample("x", sub, "h", GOLD, "h", FUZZY)
    assert s.score == 0.0 and "no_match" in s.reasons


def test_gold_empty():
    assert score_sample("x", [], "h", [], "h", FUZZY).score == 1.0
    assert score_sample("x", GOLD[:1], "h", [], "h", FUZZY).score == 0.0


def test_fuzzy_argument():
    gold = [{"name": "sms.send", "arguments": {"contact_id": "c_01", "message": "tối nay anh về muộn nhé em"}}]
    sub_ok = [{"name": "sms.send", "arguments": {"contact_id": "c_01", "message": "tối nay anh về muộn nhé"}}]
    sub_bad = [{"name": "sms.send", "arguments": {"contact_id": "c_01", "message": "về muộn"}}]
    assert score_sample("x", sub_ok, "h", gold, "h", FUZZY).score == 1.0
    assert score_sample("x", sub_bad, "h", gold, "h", FUZZY).score == 0.0
    assert token_f1("Hello World", "hello world") == 1.0


def test_int_float_bool():
    gold = [{"name": "t", "arguments": {"n": 26, "b": True}}]
    assert score_sample("x", [{"name": "t", "arguments": {"n": 26.0, "b": True}}], "h", gold, "h", FUZZY).score == 1.0
    assert score_sample("x", [{"name": "t", "arguments": {"n": 26, "b": 1}}], "h", gold, "h", FUZZY).score == 0.0
    assert score_sample("x", [{"name": "t", "arguments": {"n": "26", "b": True}}], "h", gold, "h", FUZZY).score == 0.0


def test_aggregate():
    a = score_sample("a", GOLD, "h", GOLD, "h", FUZZY)
    b = score_sample("b", [], "h", GOLD, "h", FUZZY)
    agg = aggregate([a, b])
    assert abs(agg["final"] - 50.0) < 1e-9 and agg["zero"] == 1
