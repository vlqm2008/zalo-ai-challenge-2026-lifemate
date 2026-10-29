"""A tiny stand-in for zaic_tools.ToolEnv, ONLY for plumbing tests without the real data.

Same public API: validate(name, args), call(name, args), get_tool_calls(), get_state_hash().
The real environment's behaviour (error messages, which reads exist, hash details) will
differ; everything agent-side is written against the documented API, not this mock.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

MOCK_CATALOG: dict = {
    "schema_version": "mock",
    "apps": [
        {"id": "smart_home", "name_vi": "Nhà thông minh", "tools": [
            {"name": "smart_home.get_status", "kind": "read", "description": "Danh sách thiết bị và trạng thái",
             "parameters": {"type": "object", "properties": {}}, "returns": "devices: id, name, type, room, floor, state"},
            {"name": "smart_home.light_set", "kind": "write", "description": "Bật/tắt/chỉnh sáng đèn",
             "parameters": {"type": "object", "properties": {
                 "device_id": {"type": "string", "description": "ID thiết bị, lấy từ smart_home.get_status", "x-ref": "smart_home.devices"},
                 "power": {"type": "string", "enum": ["on", "off"]},
                 "brightness": {"type": "integer"}}, "required": ["device_id"]}, "returns": ""},
            {"name": "smart_home.climate_set", "kind": "write", "description": "Điều khiển điều hòa",
             "parameters": {"type": "object", "properties": {
                 "device_id": {"type": "string", "description": "ID thiết bị, lấy từ smart_home.get_status", "x-ref": "smart_home.devices"},
                 "mode": {"type": "string", "enum": ["cool", "heat", "off"]},
                 "temperature": {"type": "integer"}}, "required": ["device_id", "mode"]}, "returns": ""},
        ]},
        {"id": "contacts", "name_vi": "Danh bạ", "tools": [
            {"name": "contacts.list", "kind": "read", "description": "Danh bạ",
             "parameters": {"type": "object", "properties": {}}, "returns": "contacts: id, name, relation, phone"},
        ]},
        {"id": "banking", "name_vi": "Ngân hàng", "tools": [
            {"name": "banking.get_balance", "kind": "read", "description": "Số dư",
             "parameters": {"type": "object", "properties": {}}, "returns": "balance"},
            {"name": "banking.transfer", "kind": "write", "description": "Chuyển tiền",
             "parameters": {"type": "object", "properties": {
                 "contact_id": {"type": "string", "description": "ID liên hệ, lấy từ contacts.list", "x-ref": "contacts.contacts"},
                 "amount": {"type": "integer"},
                 "note": {"type": "string", "x-match": "fuzzy"}}, "required": ["contact_id", "amount"]}, "returns": ""},
        ]},
        {"id": "ride", "name_vi": "Gọi xe", "tools": [
            {"name": "ride.book", "kind": "write", "description": "Đặt xe",
             "parameters": {"type": "object", "properties": {
                 "from": {"type": "string"}, "to": {"type": "string"}}, "required": ["from", "to"]}, "returns": ""},
        ]},
        {"id": "sys", "name_vi": "Hệ thống", "tools": [
            {"name": "sys.ask_missing_info", "kind": "meta", "description": "Hỏi lại thông tin thiếu",
             "parameters": {"type": "object", "properties": {
                 "tool_name": {"type": "string"}, "params": {"type": "array", "items": {"type": "string"}}},
                 "required": ["tool_name", "params"]}, "returns": ""},
        ]},
    ],
}

DEFAULT_STATE: dict = {
    "smart_home": {"devices": [
        {"id": "ac_living_room", "name": "điều hòa phòng khách", "type": "ac", "room": "phòng khách", "floor": 1,
         "state": {"power": "off", "mode": "off", "temperature": 27}},
        {"id": "light_living", "name": "đèn phòng khách", "type": "light", "room": "phòng khách", "floor": 1,
         "state": {"power": "on", "brightness": 40}},
        {"id": "light_kitchen", "name": "đèn bếp", "type": "light", "room": "bếp", "floor": 1,
         "state": {"power": "on", "brightness": 80}},
        {"id": "light_bed2", "name": "đèn phòng ngủ", "type": "light", "room": "phòng ngủ", "floor": 2,
         "state": {"power": "off", "brightness": 50}},
    ]},
    "contacts": {"contacts": [
        {"id": "c_01", "name": "Hương", "relation": "bạn", "phone": "0970195783"},
        {"id": "c_02", "name": "Mẹ", "relation": "mẹ", "phone": "0912345678"},
    ]},
    "banking": {"balance": 12000000, "transfers": []},
    "ride": {"bookings": []},
}

FUZZY_FIELDS = {"note"}


class MockToolEnv:
    def __init__(self, split: str, sample_id: str, root: str = ".", state: dict | None = None,
                 installed: list[str] | None = None):
        self.split, self.sample_id, self.root = split, sample_id, root
        self.state = copy.deepcopy(state or DEFAULT_STATE)
        self.installed = installed or [a["id"] for a in MOCK_CATALOG["apps"]]
        self._log: list[dict] = []
        self._tools = {t["name"]: t for a in MOCK_CATALOG["apps"] for t in a["tools"]}

    # -- schema checks -------------------------------------------------- #
    def validate(self, name: str, args: dict) -> dict:
        t = self._tools.get(name)
        if t is None:
            return {"ok": False, "error": f"tool không tồn tại: {name}"}
        if t["name"].split(".")[0] not in self.installed and name != "sys.ask_missing_info":
            return {"ok": False, "error": f"app chưa cài: {name}"}
        props = t["parameters"].get("properties", {})
        for r in t["parameters"].get("required", []):
            if r not in args:
                return {"ok": False, "error": f"thiếu tham số bắt buộc: {r}"}
        for k, v in args.items():
            if k not in props:
                return {"ok": False, "error": f"tham số không có trong schema: {k}"}
            typ = props[k].get("type")
            if typ == "integer" and (isinstance(v, bool) or not isinstance(v, int)):
                return {"ok": False, "error": f"{k} phải là integer"}
            if typ == "string" and not isinstance(v, str):
                return {"ok": False, "error": f"{k} phải là string"}
            if typ == "array" and not isinstance(v, list):
                return {"ok": False, "error": f"{k} phải là array"}
            if "enum" in props[k] and v not in props[k]["enum"]:
                return {"ok": False, "error": f"{k} phải thuộc {props[k]['enum']}"}
            ref = props[k].get("x-ref")
            if ref:
                app, coll = ref.split(".")
                ids = {x["id"] for x in self.state.get(app, {}).get(coll, [])}
                if v not in ids:
                    return {"ok": False, "error": f"{k} không phải ID hợp lệ: {v}"}
        return {"ok": True}

    # -- execution ------------------------------------------------------ #
    def call(self, name: str, args: dict) -> dict:
        self._log.append({"name": name, "arguments": copy.deepcopy(args)})
        v = self.validate(name, args)
        if not v["ok"]:
            return v
        if name == "smart_home.get_status":
            return {"ok": True, "devices": copy.deepcopy(self.state["smart_home"]["devices"])}
        if name in ("smart_home.light_set", "smart_home.climate_set"):
            for d in self.state["smart_home"]["devices"]:
                if d["id"] == args["device_id"]:
                    for k, val in args.items():
                        if k != "device_id":
                            d["state"][k] = val
                    if name == "smart_home.climate_set":
                        d["state"]["power"] = "off" if args["mode"] == "off" else "on"
                    return {"ok": True, "device": copy.deepcopy(d)}
        if name == "contacts.list":
            return {"ok": True, "contacts": copy.deepcopy(self.state["contacts"]["contacts"])}
        if name == "banking.get_balance":
            return {"ok": True, "balance": self.state["banking"]["balance"]}
        if name == "banking.transfer":
            if args["amount"] > self.state["banking"]["balance"]:
                return {"ok": False, "error": "số dư không đủ"}
            self.state["banking"]["balance"] -= args["amount"]
            self.state["banking"]["transfers"].append({"contact_id": args["contact_id"], "amount": args["amount"],
                                                        "note": args.get("note", "")})
            return {"ok": True, "balance": self.state["banking"]["balance"]}
        if name == "ride.book":
            self.state["ride"]["bookings"].append({"from": args["from"], "to": args["to"]})
            return {"ok": True, "booking_id": f"bk_{len(self.state['ride']['bookings'])}"}
        if name == "sys.ask_missing_info":
            return {"ok": True}
        return {"ok": False, "error": "chưa mô phỏng"}

    # -- submission ----------------------------------------------------- #
    def get_tool_calls(self) -> list[dict]:
        return copy.deepcopy(self._log)

    def get_state_hash(self) -> str:
        def strip(o: Any):
            if isinstance(o, dict):
                return {k: strip(v) for k, v in o.items() if k not in FUZZY_FIELDS}
            if isinstance(o, list):
                return [strip(x) for x in o]
            return o
        blob = json.dumps(strip(self.state), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
