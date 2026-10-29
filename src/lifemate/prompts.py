"""Prompt construction and action schema for the agent loop."""
from __future__ import annotations

import json
from typing import Any, Optional

from .data import ASK_TOOL_NAME, sanitize_schema

FINISH = "finish"

SYSTEM_PROMPT_TEMPLATE = """Bạn là trợ lý thực hiện yêu cầu bằng giọng nói trên thiết bị của người dùng.
Yêu cầu đã được chuyển thành văn bản bằng nhận dạng giọng nói nên có thể sai vài từ (nhất là tên riêng và số).
Nhiệm vụ: làm đúng và đủ những gì người dùng yêu cầu ở LƯỢT MỚI NHẤT bằng cách gọi tool của các app đã cài, MỖI LƯỢT ĐÚNG MỘT HÀNH ĐỘNG, trả lời CHỈ BẰNG JSON:
{{"thought": "<1-2 câu ngắn>", "name": "<tên tool hoặc finish>", "arguments": {{...}}}}

Quy tắc bắt buộc:
1. Chỉ gọi tool có trong mục TOOLS. Không bịa tool, không thêm tham số ngoài schema, không bỏ tham số required. Giá trị enum phải đúng y chuỗi trong schema.{R1_EXTRA}
2. Tham số ID (có x-ref) phải là id lấy từ kết quả tool đọc tương ứng (mô tả tham số ghi "lấy từ <tool>") hoặc từ kết quả tool trong HISTORY. ID đã có trong HISTORY thì dùng luôn, KHÔNG gọi lại tool đọc. Chưa có ID thì gọi tool đọc đó trước. Tuyệt đối không đoán ID.{R2_EXTRA}
3. Liên hệ: gọi contacts.search với query là tên/xưng hô đúng như người dùng nói (ví dụ "Sếp Dũng", "bác Ba"). Kết quả đúng 1 liên hệ thì dùng id đó; 0 hoặc từ 2 liên hệ trở lên (trùng tên) thì không đoán mà hỏi lại theo quy tắc 7.
4. CONTEXT: user_profile.home_address = "nhà"; user_profile.work_address = "công ty"/"chỗ làm"/"cơ quan"; user_profile.city = thành phố hiện tại; location chỉ là toạ độ GPS (lat, lng), không phải địa chỉ. Địa chỉ, tên, chuỗi lấy từ CONTEXT hoặc kết quả tool phải chép nguyên văn, không sửa chính tả, không rút gọn. "chỗ hẹn"/"chỗ họp" là location của sự kiện tương ứng trong lịch.
5. Số tiền và thời gian: dùng giá trị đã tính sẵn trong mục HINTS. Số tiền là số nguyên, không đơn vị. Thời gian ISO 8601 đầy đủ giây và múi giờ giống CONTEXT.current_time (ví dụ 2026-11-17T15:00:00+07:00); ngày dạng YYYY-MM-DD; "mai" = ngày sau current_time. Số tiền của hoá đơn, món ăn, giao dịch... lấy từ kết quả tool đọc, không tự nghĩ.
6. Nội dung tin nhắn, email, ghi chú, tiêu đề (x-match: fuzzy): giữ đúng lời người dùng, ngắn gọn, không thêm bớt, không viết lại cho lịch sự.{R6_EXTRA}
7. Thiếu thông tin: nếu một tham số bắt buộc (hoặc cả nhóm "bắt buộc một trong") không suy ra được từ yêu cầu, CONTEXT, HISTORY hay kết quả tool (không nói số tiền, không nói giờ, liên hệ trùng tên hoặc không tìm thấy...) thì: làm xong mọi việc khác trước, vẫn gọi các tool đọc cần cho việc đó (ví dụ contacts.search), rồi gọi sys.ask_missing_info với tool_name là tool định gọi và params là danh sách ĐÚNG MỘT tên tham số còn thiếu (đúng tên trong schema). Đây luôn là hành động cuối cùng. Không hỏi khi có thể suy ra hoặc tham số không bắt buộc.{R7_EXTRA}
8. Nhóm "bắt buộc một trong" (x-one-of, ví dụ contact_id hoặc phone_number): truyền đúng MỘT tham số của nhóm. Người dùng đọc số điện thoại thì dùng phone_number/phone với chuỗi số (giữ số 0 đầu), không cần tìm danh bạ.
9. Nếu không app nào đã cài làm được yêu cầu, người dùng chỉ trò chuyện/hỏi chung chung, hoặc rút lại yêu cầu ("thôi", "khỏi"), finish ngay mà không gọi tool nào. Phần bị rút lại thì bỏ, phần còn giữ thì vẫn làm. Người dùng sửa lại ý ("à không", "nhầm") thì theo ý sau cùng.{R9_EXTRA}
10. Không gọi lại một tool với cùng tham số. Không gọi tool đọc nếu kết quả không cần cho bước tiếp theo hoặc cho điều kiện người dùng nêu; không "kiểm tra lại" sau khi làm xong. Yêu cầu có điều kiện ("nếu mai mưa thì...") thì gọi tool đọc để kiểm tra điều kiện rồi mới quyết định.{R10_EXTRA}
11. Một câu có thể chứa nhiều việc; làm lần lượt từng việc rồi mới finish. "hết đèn"/"tất cả" áp dụng cho mọi thiết bị phù hợp trong kết quả tool đọc.
12. Kết quả có "ok": false là lỗi schema (sửa tham số theo thông báo); "done": false là tool chạy nhưng không làm được (ID sai, không tìm thấy...): đọc message, sửa lại hoặc hỏi theo quy tắc 7, không lặp lại y nguyên."""


PROMPT_EXTRAS = {
    "R1_EXTRA": " Chỉ truyền tham số mà người dùng nói rõ hoặc schema bắt buộc; tham số tuỳ chọn (label, note, location, repeat, payment, date_from/date_to, bộ lọc room/type/floor...) KHÔNG tự thêm khi người dùng không nhắc tới — để tool dùng mặc định. Tool đọc dạng danh sách (list_*, get_status, get_balance, list_events, list_bills...) gọi với arguments {} rồi tự lọc trong kết quả; chỉ truyền tham số đọc khi nó là từ khoá tìm kiếm người dùng nói (query, date của dự báo...) hoặc bắt buộc.",
    "R2_EXTRA": " Xem mô tả tham số để biết ID lấy từ tool đọc nào: ví dụ chuyển khoản ngân hàng cần beneficiary_id từ banking.list_beneficiaries và from_account_id từ banking.get_balance (không tìm danh bạ); chuyển ví/nhắn tin/gọi cần contact_id từ contacts.search. \"người đó\", \"cái đó\", \"số đó\", \"khoản đó\" trỏ về đối tượng vừa nhắc trong HISTORY (kết quả tool gần nhất): dùng ID/giá trị đó, không hỏi lại.",
    "R7_EXTRA": " tool_name luôn là tool hành động định gọi (phone.call, sms.send, ewallet.transfer...), không phải tool đọc; params là tham số của tool hành động đó (ví dụ contact_id), không phải query. Tìm kiếm không có kết quả: thử lại ĐÚNG MỘT lần với từ khoá ngắn hơn (bỏ số tập/thứ tự, bỏ xưng hô) hoặc dùng tool đọc của đúng app (người thụ hưởng ngân hàng nằm ở banking.list_beneficiaries, không ở danh bạ); chỉ hỏi lại khi vẫn không có.",
    "R9_EXTRA": " Thông tin đã có trong HISTORY (giá vàng, thời tiết, số dư vừa xem) thì trả lời từ HISTORY, không gọi lại tool đọc. Yêu cầu không thể thực hiện (không có app phù hợp, giá trị vô lý như độ sáng âm, dữ liệu không có trong app) thì finish ngay, không gọi tool để \"thử\". Không gọi tool ghi (lưu, tạo, đặt) mà người dùng không yêu cầu.",
    "R10_EXTRA": " Chỉ gọi tool đọc khi cần ID/dữ liệu cho tham số của bước sau, khi người dùng hỏi thông tin đó, hoặc để kiểm tra điều kiện; không xem báo giá (get_quotes) hay danh sách trước khi đặt nếu không được yêu cầu. Với contacts.search, query là tên như người dùng nói, bỏ \"anh/chị/em/bạn\" đứng trước tên.",
    "R6_EXTRA": " Khi người dùng chỉ nói ý (\"báo giờ mới\", \"báo là đã đóng tiền\"), viết một câu ngắn thân mật chứa đúng thông tin chính (việc gì, giờ mới dạng HH:MM, số tiền), không chào hỏi, không ký tên, không giải thích.",
}


def system_prompt(extras: bool = False) -> str:
    """Baseline rules, optionally with the measured Phase-3 additions (config.prompt_extras)."""
    return SYSTEM_PROMPT_TEMPLATE.format(**{k: (v if extras else "") for k, v in PROMPT_EXTRAS.items()})


SYSTEM_PROMPT = system_prompt(False)


def render_tools(tools: list[dict], include_returns: bool = True) -> str:
    lines = []
    for t in tools:
        item: dict[str, Any] = {
            "name": t["name"],
            "kind": t.get("kind", ""),
            "description": t.get("description", ""),
            "parameters": t.get("parameters", {}),
        }
        if include_returns and t.get("returns"):
            item["returns"] = t["returns"]
        lines.append(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines)


def render_context(context: dict) -> str:
    """Real keys (data README 2.1): current_time, timezone, location{lat,lng}, user_profile{name, home_address,
    work_address, city}, installed_apps{app: {tools: [...]}} (shown as the list of app ids; tools are in TOOLS)."""
    ctx: dict[str, Any] = {}
    for k, v in (context or {}).items():
        if k == "installed_apps":
            ctx[k] = list(v.keys()) if isinstance(v, dict) else v
        else:
            ctx[k] = v
    return json.dumps(ctx, ensure_ascii=False, indent=1)


HISTORY_RESULT_MAX_CHARS = 1800


def render_history(history: list, transcripts: list[Optional[str]]) -> str:
    """One JSON line per turn. User turns: {"role": "user", "text": <transcript>} (the audio path is replaced
    by the transcript given in `transcripts`, in order; falls back to the turn's own audio_text/text).
    Assistant turns keep text, tool_calls and tool_results (compacted when long) because gold reuses the
    IDs returned there without re-reading."""
    if not history:
        return "(không có)"
    out = []
    ti = 0
    for turn in history:
        if not isinstance(turn, dict):
            out.append(str(turn))
            continue
        role = turn.get("role") or ("user" if turn.get("audio") else "assistant")
        if role == "user":
            txt = None
            if turn.get("audio"):
                txt = transcripts[ti] if ti < len(transcripts) else None
                ti += 1
            if txt is None:
                txt = turn.get("audio_text") or turn.get("text") or ""
            out.append(json.dumps({"role": "user", "text": txt}, ensure_ascii=False))
        else:
            item: dict[str, Any] = {"role": "assistant"}
            if turn.get("text"):
                item["text"] = turn["text"]
            if turn.get("tool_calls"):
                item["tool_calls"] = turn["tool_calls"]
            if turn.get("tool_results"):
                res = turn["tool_results"]
                if len(json.dumps(res, ensure_ascii=False)) > HISTORY_RESULT_MAX_CHARS:
                    res = _compact(res, HISTORY_RESULT_MAX_CHARS)
                item["tool_results"] = res
            out.append(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(out)


def build_user_message(context: dict, history_text: str, transcript: str, hints: list[str], tools_text: str) -> str:
    hints_text = "\n".join(f"- {h}" for h in hints) if hints else "- (không có)"
    return (
        f"## CONTEXT\n{render_context(context)}\n\n"
        f"## HISTORY (các lượt trước của cùng cuộc hội thoại; tool_calls ở đây đã được thực hiện, ID trong tool_results vẫn dùng được)\n{history_text}\n\n"
        f"## YÊU CẦU MỚI NHẤT (từ giọng nói)\n\"{transcript}\"\n\n"
        f"## HINTS (đã chuẩn hóa bằng code, ưu tiên dùng)\n{hints_text}\n\n"
        f"## TOOLS (mỗi dòng một tool, JSON Schema của parameters)\n{tools_text}\n\n"
        "Hãy trả lời hành động đầu tiên."
    )


def render_result(name: str, result: Any, max_chars: int = 2500) -> str:
    """Compact rendering of a tool result; long lists of records are tabulated instead of truncated."""
    text = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= max_chars:
        return f"KẾT QUẢ {name}:\n{text}"
    # try to compact lists of dicts
    compact = _compact(result, max_chars)
    text = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    if len(text) > max_chars:
        text = text[: max_chars - 20] + " ...(cắt bớt)"
    return f"KẾT QUẢ {name} (rút gọn):\n{text}"


_KEY_PRIORITY = ("id", "name", "title", "type", "room", "floor", "state", "status", "relation", "phone", "balance",
                 "amount", "time", "start", "end", "date", "due_date", "paid", "provider", "address", "location",
                 "price", "email", "enabled", "repeat", "label", "content", "condition", "rain", "done", "message")


def _compact(obj: Any, budget: int) -> Any:
    if isinstance(obj, dict):
        return {k: _compact(v, budget) for k, v in obj.items()}
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        keys = [k for k in _KEY_PRIORITY if any(k in x for x in obj)]
        if not keys:
            keys = list(obj[0].keys())[:6]
        return [{k: x[k] for k in keys if k in x} for x in obj]
    if isinstance(obj, list):
        return [_compact(x, budget) for x in obj]
    return obj


def action_schema(tools: list[dict], use_thought: bool = True) -> dict:
    """JSON schema of one action: {"thought"?, "name", "arguments"} with per-tool argument schemas."""
    variants = []
    for t in tools:
        params = sanitize_schema(t.get("parameters") or {"type": "object"})
        if not isinstance(params, dict):
            params = {"type": "object"}
        params = dict(params)
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        params["additionalProperties"] = False
        if t["name"] == ASK_TOOL_NAME:
            props = dict(params.get("properties") or {})
            names = [x["name"] for x in tools if x["name"] != ASK_TOOL_NAME]
            if names:
                props["tool_name"] = {"type": "string", "enum": names}
            params["properties"] = props
        variants.append(_variant(t["name"], params, use_thought))
    variants.append(_variant(FINISH, {"type": "object", "properties": {}, "additionalProperties": False}, use_thought))
    return {"anyOf": variants}


def _variant(name: str, params: dict, use_thought: bool) -> dict:
    props: dict[str, Any] = {}
    if use_thought:
        props["thought"] = {"type": "string", "maxLength": 300}
    props["name"] = {"const": name}
    props["arguments"] = params
    req = (["thought"] if use_thought else []) + ["name", "arguments"]
    return {"type": "object", "properties": props, "required": req, "additionalProperties": False}


def parse_action(text: str) -> Optional[dict]:
    """Parse the model output into {"name", "arguments", "thought"}; tolerant to stray text."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    for cand in (text, _first_json_object(text)):
        if not cand:
            continue
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            obj.setdefault("arguments", {})
            if not isinstance(obj["arguments"], dict):
                obj["arguments"] = {}
            return obj
    return None


def _first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None
