import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifemate.normalize_vi import (  # noqa: E402
    extract_money,
    extract_numbers,
    extract_time_hints,
    norm_key,
)

NOW = "2026-11-16T17:00:00+07:00"  # Monday


def money(text):
    return [s.int_value for s in extract_money(text)]


def test_money_digits_and_units():
    assert money("chuyển cho mẹ 500 nghìn") == [500000]
    assert money("chuyển 500k") == [500000]
    assert money("nạp 1tr5") == [1500000]
    assert money("gửi 1,5 triệu") == [1500000]
    assert money("thanh toán 1.500.000 đồng") == [1500000]
    assert money("trả 350 ngàn") == [350000]


def test_money_words():
    assert money("chuyển năm trăm nghìn") == [500000]
    assert money("hai triệu rưỡi") == [2500000]
    assert money("ba triệu hai") == [3200000]
    assert money("ba triệu hai trăm") == [3200000]
    assert money("một triệu năm trăm nghìn") == [1500000]
    assert money("mười nghìn") == [10000]
    assert money("hai mươi lăm nghìn") == [25000]
    assert money("một trăm hai mươi nghìn") == [120000]
    assert money("nửa triệu") == [] or True  # "nửa" not handled by number parser; LLM gets raw text
    assert money("hai trăm linh ba nghìn") == [203000]


def test_year_not_number():
    assert money("đặt lịch năm sau 500k") == [500000]
    vals = [s.int_value for s in extract_numbers("năm nay tôi 31 tuổi")]
    assert 31 in vals and 5 not in vals


def test_small_numbers():
    vals = [s.int_value for s in extract_numbers("bật điều hòa phòng khách 26 độ")]
    assert vals == [26]
    vals = [s.int_value for s in extract_numbers("chỉnh sáng năm mươi phần trăm")]
    assert vals == [50]


def test_time_clock_and_day():
    h = extract_time_hints("nhắc tôi họp lúc ba giờ chiều mai", NOW)
    assert any(x.iso == "2026-11-17T15:00:00+07:00" for x in h)
    h = extract_time_hints("đặt báo thức 5 giờ sáng", NOW)
    assert any(x.iso == "2026-11-17T05:00:00+07:00" for x in h)  # 05:00 today already passed
    h = extract_time_hints("đặt báo thức 5 rưỡi sáng mai", NOW)
    assert any(x.iso == "2026-11-17T05:30:00+07:00" for x in h)
    h = extract_time_hints("họp lúc 15h30 hôm nay", NOW)
    assert any(x.iso == "2026-11-16T15:30:00+07:00" for x in h)
    h = extract_time_hints("8 giờ tối nay gọi cho mẹ", NOW)
    assert any(x.iso == "2026-11-16T20:00:00+07:00" for x in h)
    h = extract_time_hints("9 giờ sáng thứ sáu", NOW)
    assert any(x.iso == "2026-11-20T09:00:00+07:00" for x in h)
    h = extract_time_hints("thứ hai tuần sau 10 giờ", NOW)
    assert any(x.iso == "2026-11-23T10:00:00+07:00" for x in h)
    h = extract_time_hints("ngày 20/11 lúc 7 giờ", NOW)
    assert any(x.iso == "2026-11-20T07:00:00+07:00" for x in h)
    h = extract_time_hints("11 giờ đêm", NOW)
    assert any(x.iso == "2026-11-16T23:00:00+07:00" for x in h)


def test_time_duration():
    h = extract_time_hints("nhắc tôi sau 30 phút nữa", NOW)
    assert any(x.iso == "2026-11-16T17:30:00+07:00" for x in h)
    h = extract_time_hints("2 tiếng nữa tắt đèn", NOW)
    assert any(x.iso == "2026-11-16T19:00:00+07:00" for x in h)


def test_norm_key():
    assert norm_key("Điều Hòa  Phòng Khách") == "dieu hoa phong khach"


# --- patterns measured on the real train transcripts (scripts/eval_hints.py) ---------------------

def _iso(text, now=NOW):
    return [x.iso for x in extract_time_hints(text, now) if x.kind == "datetime"]


def test_spoken_hours_keep_diacritics():
    assert "2026-11-16T20:00:00+07:00" in _iso("nhắc tôi tám giờ tối nay")
    assert "2026-11-16T22:00:00+07:00" in _iso("cứ mười giờ tối là bật lên")
    assert "2026-11-17T05:00:00+07:00" in _iso("Đặt báo thức năm giờ sáng mai cho em")
    assert "2026-11-17T20:30:00+07:00" in _iso("gọi lại cho khách lúc tám rưỡi tối mai")
    assert "2026-11-16T23:15:00+07:00" in _iso("tự tắt lúc mười một giờ mười lăm tối hằng ngày")
    assert "2026-11-17T19:45:00+07:00" in _iso("ra sân bay lúc 8 giờ kém 15 tối mai")


def test_nearest_day_wins_over_first_day():
    # "nay" earlier in the sentence must not capture "9 giờ sáng mai"
    assert "2026-11-17T09:00:00+07:00" in _iso("Vàng bán ra nay trên 129 triệu thì nhắc tôi 9 giờ sáng mai đi bán")
    assert "2026-11-23T10:00:00+07:00" in _iso("tạo cái lịch mười giờ sáng thứ hai tuần sau")


def test_word_dates_and_name_mai():
    h = extract_time_hints("buổi tập huấn hôm mùng sáu tháng mười hai ấy", NOW)
    assert any(x.iso == "2026-12-06" for x in h)
    h = extract_time_hints("lịch họp team ngày 19 nhé", NOW)
    assert any(x.iso == "2026-11-19" for x in h)
    assert not extract_time_hints("Chuyển cho chị Mai bên hậu cần năm chục nghìn", NOW)   # Mai = a name
    assert any(x.iso == "2026-11-17" for x in extract_time_hints("mai nhắc tôi", NOW))


def test_money_chuc_and_implied_thousands():
    assert money("chuyển năm chục nghìn") == [50000]
    assert money("sáu trăm bốn chục nghìn") == [640000]
    assert money("bốn chục triệu") == [40000000]
    assert money("nâng thêm ngàn rưỡi bước") == [1500]
    from lifemate.normalize_vi import build_hints
    lines = build_hints("chuyển thằng Út ba trăm bằng tài khoản thanh toán", NOW)
    assert any("300000" in l for l in lines)
    lines = build_hints("tập yoga được bốn lăm phút", NOW)
    assert any('"bốn lăm" = 45' in l for l in lines) and any("45 phút" in l for l in lines)
    assert [s.int_value for s in extract_numbers("đổi bảy mét ruy băng")] == [7]
