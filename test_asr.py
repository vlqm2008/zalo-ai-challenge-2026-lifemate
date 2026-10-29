"""python test_asr.py — word error counting used for the ASR comparison."""
from asr import edits

assert edits("Tắt đèn phòng khách.", "tắt đèn phòng khách") == (0, 4)  # case and punctuation ignored
assert edits("nhắn cho mẹ", "nhắn cho mẹ nhé") == (1, 3)  # insertion
assert edits("chuyển năm trăm nghìn", "chuyển 500 nghìn") == (2, 4)  # substitution + deletion
assert edits("", "xin chào") == (2, 0)
print("ok")
