"""Observed journey data for later comparisons; this module never selects actions."""
from __future__ import annotations

import copy
import json
import math
import uuid
import re
from datetime import datetime
from pathlib import Path

from .runtime_control import RuntimeControl

STAT_NAMES = ("力量", "体力", "韧性", "专注", "保护")
SETUP_FIELDS = ("character", "journey_records", "support_cards")


def now_iso():
    return datetime.now().astimezone().isoformat()


def field(value=None, source="unobserved", evidence=None, observed_at=None):
    return {"value": value, "source": source, "observed_at": observed_at,
            "evidence": evidence}


def normalize_setup(body):
    """Blank is unknown; an explicit empty list means no items were carried."""
    if not isinstance(body, dict):
        raise ValueError("实际携带配置必须是对象")
    setup = {}
    character = body.get("character")
    if character is not None:
        if not isinstance(character, str) or len(character) > 80:
            raise ValueError("实际角色名称无效")
        setup["character"] = character.strip() or None
    for key in SETUP_FIELDS[1:]:
        items = body.get(key)
        if items is None:
            setup[key] = None
            continue
        if not isinstance(items, list) or len(items) > 20:
            raise ValueError("旅程记录与支援卡必须是最多 20 项的列表")
        normalized = []
        for item in items:
            item = {"name": item} if isinstance(item, str) else item
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise ValueError("每项携带资料必须有名称")
            name = item["name"].strip()
            if not name or len(name) > 120:
                raise ValueError("携带资料名称无效")
            entry = {"name": name}
            for detail in ("id", "rarity", "level", "limit_break", "borrowed", "slot"):
                value = item.get(detail)
                if value is not None:
                    if detail == "borrowed":
                        valid = isinstance(value, bool)
                    elif detail in {"level", "limit_break", "slot"}:
                        valid = type(value) is int and 0 <= value <= 999
                    else:
                        valid = isinstance(value, str) and len(value) <= 120
                    if not valid:
                        raise ValueError(f"携带资料的 {detail} 无效")
                    entry[detail] = value
            normalized.append(entry)
        setup[key] = normalized
    return setup


def screen_observation(items, page, round_num, target_revision=None):
    """Read labelled values only. Gains, suggestions and ambiguous rows stay unknown."""
    rows = []
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            confidence = float(item[1])
            if not math.isfinite(confidence):
                continue
            box = item[2].tolist() if len(item) > 2 and hasattr(item[2], "tolist") else (item[2] if len(item) > 2 else None)
            rows.append({"text": str(item[0]), "confidence": confidence, "box": box})
        except (ValueError, TypeError):
            continue
    values, evidence = {}, {}
    labels = {name: name for name in STAT_NAMES}
    labels.update({"潜质点数": "potential_points", "综合等级": "rank", "RANK": "rank",
                   "古币": "money", "心情": "mood", "耐力": "stamina_pct"})

    def bounds(row):
        try:
            points = row["box"]
            if len(points) != 4 or any(len(point) != 2 for point in points):
                return None
            xs, ys = zip(*points)
            return min(xs), min(ys), max(xs), max(ys)
        except (TypeError, ValueError):
            return None

    for label, key in labels.items():
        candidates = []
        value_pattern = (r"(?:Worst|Bad|Normal|Good|Best)" if key == "mood" else
                         r"\d{1,5}(?:\s*/\s*\d{1,5})?" if key in STAT_NAMES else
                         r"\d{1,5}%?" if key == "stamina_pct" else r"\d{1,5}")
        for i, row in enumerate(rows):
            if row["confidence"] < .85:
                continue
            text = row["text"].strip()
            direct = re.fullmatch(re.escape(label) + r"\s*[:：]?\s*(" + value_pattern + r")", text, re.I)
            if direct:
                candidates.append((direct.group(1), [i]))
            elif text == label and bounds(row):
                left, top, right, bottom = bounds(row)
                height, width = max(1, bottom - top), max(1, right - left)
                for j, other in enumerate(rows):
                    if other["confidence"] < .85 or not re.fullmatch(value_pattern, other["text"].strip(), re.I):
                        continue
                    rect = bounds(other)
                    if not rect:
                        continue
                    ox, oy, ex, ey = rect
                    same_line = abs((oy + ey - top - bottom) / 2) < .75 * height and right <= ox <= right + 3 * width
                    below = bottom <= oy <= bottom + 1.6 * height and abs(ox - left) <= width
                    if same_line or below:
                        candidates.append((other["text"].strip(), [i, j]))
        if len(candidates) == 1:
            raw, indices = candidates[0]
            value = raw if key == "mood" else int(re.match(r"\d+", raw).group())
            if key != "stamina_pct" or value <= 100:
                values[key], evidence[key] = value, indices
    skills = []
    for i, row in enumerate(rows):
        match = re.fullmatch(r"(?:已装备技能|装备技能)\s*[:：]\s*(.+)", row["text"].strip())
        if match and row["confidence"] >= .85:
            codes = re.split(r"[、,，\s]+", match.group(1).strip())
            if codes and all(re.fullmatch(r"(?:[A-H]X[1-8]|XX[1-3])", code, re.I) for code in codes):
                skills.append(([code.upper() for code in codes], [i]))
    if len(skills) == 1:
        values["equipped_skills"], evidence["equipped_skills"] = skills[0]
    appraisal = [(i, re.fullmatch(r"评鉴(?:战)?结果\s*[:：]\s*(胜利|失败|VICTORY|DEFEAT)", row["text"].strip(), re.I))
                 for i, row in enumerate(rows) if row["confidence"] >= .85]
    appraisal = [(i, match) for i, match in appraisal if match]
    if len(appraisal) == 1:
        i, match = appraisal[0]
        values["appraisal_result"] = "victory" if match.group(1).upper() in {"胜利", "VICTORY"} else "defeat"
        evidence["appraisal_result"] = [i]
    dates, turns = [], []
    for i, row in enumerate(rows):
        if row["confidence"] < .85:
            continue
        text = row["text"].strip()
        date = re.fullmatch(r"[@＠]?\s*(\d{1,2})月(上旬|中旬|下旬)", text)
        turn = re.fullmatch(r"(\d{1,2})\s*/\s*45", text)
        if date and 1 <= int(date.group(1)) <= 12:
            dates.append((f"{int(date.group(1))}月{date.group(2)}", [i]))
        if turn and 1 <= int(turn.group(1)) <= 45:
            turns.append((int(turn.group(1)), [i]))
    for key, candidates in (("game_date", dates), ("game_turn", turns)):
        if len(candidates) == 1:
            values[key], evidence[key] = candidates[0]
    return {"observed_at": now_iso(), "page": page, "round": round_num,
            "target_revision": target_revision, "values": values, "value_evidence": evidence,
            "ocr": rows, "screenshot": None}


class JourneyRecordStore:
    def __init__(self, root):
        self.root = Path(root)
        self.directory = self.root / "logs" / "journeys"

    def path(self, journey_id):
        if not isinstance(journey_id, str) or not re.fullmatch(r"[a-f0-9]{32}", journey_id):
            raise ValueError("旅程编号无效")
        return self.directory / f"{journey_id}.json"

    def read(self, journey_id):
        path = self.path(journey_id)
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("journey_id") != journey_id:
            raise ValueError("旅程记录内容损坏，保留原文件等待核对")
        return record

    @staticmethod
    def _quality(record):
        missing = ["actual_setup." + key for key in SETUP_FIELDS
                   if record["actual_setup"][key]["value"] is None]
        missing += ["outcome.final_stats." + name for name in STAT_NAMES
                    if record["outcome"]["final_stats"][name]["value"] is None]
        missing += ["outcome." + key for key in ("equipped_skills", "target_skill_achieved", "appraisal_result")
                    if record["outcome"][key]["value"] is None]
        warnings = []
        if record["entry_point"] != "journey_start":
            warnings.append("未从选角开局完整记录，可能缺少前段决策")
        goals = {tuple(target.get(key) for key in ("character", "playstyle", "combo_code", "training_direction"))
                 for target in record["target_history"]}
        if len(goals) > 1:
            warnings.append("旅程中修改过目标，比较策略时需要按目标版本区分")
        record["data_quality"] = {"missing_fields": missing, "warnings": warnings,
                                  "journey_ended": record["outcome"]["journey_ended"],
                                  "ready_for_comparison": record["outcome"]["journey_ended"] and not missing and not warnings}

    @staticmethod
    def _put(fields, key, value, source, stamp, evidence=None):
        if value is None:
            return
        previous = fields.get(key, {})
        previous_stamp = previous.get("observed_at")
        if previous_stamp and datetime.fromisoformat(previous_stamp).astimezone() > datetime.fromisoformat(stamp).astimezone():
            return
        fields[key] = field(copy.deepcopy(value), source, evidence, stamp)

    def merge_run(self, payload, log_path):
        journey = payload["journey"]
        record = self.read(journey["journey_id"])
        if record is None:
            record = {"schema_version": 1, "journey_id": journey["journey_id"],
                      "started_at": journey["started_at"], "entry_point": journey["entry_point"],
                      "target_history": [], "actual_setup": {key: field() for key in SETUP_FIELDS},
                      "setup_history": [], "segments": [], "latest_observation": None,
                      "exploration": {"included": False, "attempts": 0, "mode": "normal"},
                      "outcome": {"journey_ended": False, "ended_at": None, "final_observation": None,
                                  "execution_mode": "normal",
                                  "final_stats": {name: field() for name in STAT_NAMES},
                                  "equipped_skills": field(), "target_skill_achieved": field(),
                                  "appraisal_result": field(), "rank": field(), "potential_points": field()}}
        record["updated_at"] = now_iso()
        segment = {"run_id": payload["run_id"], "log_path": log_path, "started_at": payload["started_at"],
                   "exit_reason": payload["exit_reason"], "parent_run_id": journey.get("parent_run_id"),
                   "config_hash": payload["strategy"]["config_hash"]}
        segment["strategy_hash"] = payload["strategy"].get("strategy_hash")
        exploration = payload.get("exploration") or {}
        if exploration.get("included"):
            record.setdefault("exploration", {"included": False, "attempts": 0, "mode": "normal"})
            record["exploration"]["included"] = True
            record["exploration"]["attempts"] = int(record["exploration"].get("attempts", 0)) + len(exploration.get("attempts", []))
            record["exploration"]["mode"] = "exploration"
        elif "exploration" not in record:
            record["exploration"] = {"included": False, "attempts": 0, "mode": "normal"}
        # Once any segment contains exploratory choices, keep that fact on the
        # journey summary even if a later resumed segment is routine.
        record["outcome"]["execution_mode"] = (
            "exploration" if record["exploration"].get("included") else
            payload.get("execution_mode", "normal"))
        record["segments"] = [item for item in record["segments"] if item["run_id"] != payload["run_id"]] + [segment]
        for target in payload["target_history"]:
            if not any(item.get("revision") == target.get("revision") for item in record["target_history"]):
                record["target_history"].append(copy.deepcopy(target))
        for setup in payload["setup_observations"]:
            if not any(item["id"] == setup["id"] for item in record["setup_history"]):
                record["setup_history"].append(copy.deepcopy(setup))
                for key in SETUP_FIELDS:
                    self._put(record["actual_setup"], key, setup["values"].get(key), setup["source"],
                              setup["observed_at"], setup.get("evidence"))
        observations = payload["observations"]
        if observations:
            record["latest_observation"] = copy.deepcopy(observations[-1])
        if payload["exit_reason"] == "journey_end":
            record["outcome"]["journey_ended"] = True
            record["outcome"]["ended_at"] = record["outcome"]["ended_at"] or now_iso()
            final = payload.get("final_observation")
            # The last main menu is not a final result. Keep it only as latest_observation.
            if final:
                record["outcome"]["final_observation"] = copy.deepcopy(final)
                values = final["values"]
                evidence = {"run_id": payload["run_id"], "observation_id": final["id"], "screenshot": final["screenshot"]}
                for name in STAT_NAMES:
                    self._put(record["outcome"]["final_stats"], name, values.get(name), "ocr", final["observed_at"], evidence)
                for key in ("rank", "potential_points", "equipped_skills", "appraisal_result"):
                    self._put(record["outcome"], key, values.get(key), "ocr", final["observed_at"], evidence)
                if "equipped_skills" in values and record["target_history"]:
                    achieved = record["target_history"][-1]["combo_code"] in values["equipped_skills"]
                    self._put(record["outcome"], "target_skill_achieved", achieved, "ocr", final["observed_at"], evidence)
        self._quality(record)
        RuntimeControl._atomic_write(self.path(record["journey_id"]), record)
        return record

    def supplement(self, journey_id, body):
        record = self.read(journey_id)
        if record is None:
            raise FileNotFoundError("旅程记录不存在")
        if not record["outcome"]["journey_ended"]:
            raise ValueError("旅程尚未结束，请结束后补录结果")
        if set(body) - {"actual_setup", "final_stats", "equipped_skills", "appraisal_result", "rank", "potential_points"}:
            raise ValueError("包含不支持的结果字段")
        stats = body.get("final_stats", {})
        if not isinstance(stats, dict) or set(stats) - set(STAT_NAMES):
            raise ValueError("最终属性字段无效")
        for key, value in {**stats, **{k: body[k] for k in ("rank", "potential_points") if k in body}}.items():
            if value is not None and (type(value) is not int or not 0 <= value <= 99999):
                raise ValueError("最终属性与点数必须是非负整数")
        skills = body.get("equipped_skills")
        if skills is not None and (not isinstance(skills, list) or len(skills) > 20 or
                any(not isinstance(code, str) or not re.fullmatch(r"(?:[A-H]X[1-8]|XX[1-3])", code, re.I) for code in skills)):
            raise ValueError("最终装备技能请填写已知组合代码")
        appraisal = body.get("appraisal_result")
        if appraisal is not None and (not isinstance(appraisal, str) or appraisal not in {"victory", "defeat"}):
            raise ValueError("评鉴结果无效")
        setup = normalize_setup(body["actual_setup"]) if "actual_setup" in body else None
        stamp = now_iso()
        if setup is not None:
            setup_observation = {"id": uuid.uuid4().hex, "observed_at": stamp, "source": "user_declared", "values": setup}
            record["setup_history"].append(setup_observation)
            for key in SETUP_FIELDS:
                self._put(record["actual_setup"], key, setup.get(key), "user_declared", stamp)
        for name, value in stats.items():
            self._put(record["outcome"]["final_stats"], name, value, "user_declared", stamp)
        for key in ("rank", "potential_points", "appraisal_result", "equipped_skills"):
            if key in body:
                value = [code.upper() for code in body[key]] if key == "equipped_skills" and body[key] is not None else body[key]
                self._put(record["outcome"], key, value, "user_declared", stamp)
        if skills is not None and record["target_history"]:
            self._put(record["outcome"], "target_skill_achieved",
                      record["target_history"][-1]["combo_code"] in [code.upper() for code in skills], "user_declared", stamp)
        record.setdefault("supplements", []).append({"observed_at": stamp, "source": "user_declared", "values": body})
        record["updated_at"] = stamp
        self._quality(record)
        RuntimeControl._atomic_write(self.path(journey_id), record)
        return record

    def apply_screen_scan(self, journey_id, scan):
        """Merge a verified journey-information screen into a finished record.

        A scan is evidence for fields that are actually visible on the panel.
        It never clears unknown values and never replaces a later manual value.
        """
        record = self.read(journey_id)
        if record is None:
            raise FileNotFoundError("旅程记录不存在")
        if not record["outcome"]["journey_ended"]:
            raise ValueError("旅程尚未结束，请结束后再读取旅程信息")
        if not isinstance(scan, dict) or not isinstance(scan.get("scan_id"), str):
            raise ValueError("扫描结果无效")
        stamp = scan.get("observed_at") or now_iso()
        evidence = {"scan_id": scan["scan_id"], "screenshot": scan.get("screenshot"),
                    "before_screenshot": scan.get("before_screenshot")}
        field_evidence = scan.get("evidence") or {}

        def put_screen(fields, key, value):
            if value is None:
                return
            previous = fields.get(key) or {}
            # A human supplement is an explicit correction and remains the
            # authoritative value when the panel is scanned again.
            if previous.get("source") == "user_declared":
                return
            detail = dict(evidence)
            if key in field_evidence:
                detail["ocr_indices"] = copy.deepcopy(field_evidence[key])
            self._put(fields, key, value, "screen_confirmed", stamp, detail)

        setup_values = scan.get("actual_setup") or {}
        normalized_setup = normalize_setup({key: setup_values.get(key) for key in SETUP_FIELDS})
        setup_observation = {"id": uuid.uuid4().hex, "observed_at": stamp,
                             "source": "screen_confirmed", "values": normalized_setup,
                             "evidence": evidence}
        record.setdefault("setup_history", []).append(setup_observation)
        for key in SETUP_FIELDS:
            put_screen(record["actual_setup"], key, normalized_setup.get(key))

        outcome = scan.get("outcome") or {}
        for name, value in (outcome.get("final_stats") or {}).items():
            if name in record["outcome"]["final_stats"]:
                put_screen(record["outcome"]["final_stats"], name, value)
        for key in ("rank", "potential_points", "equipped_skills", "appraisal_result"):
            put_screen(record["outcome"], key, outcome.get(key))

        record.setdefault("screen_scans", []).append({
            "scan_id": scan["scan_id"], "observed_at": stamp,
            "screenshot": scan.get("screenshot"),
            "before_screenshot": scan.get("before_screenshot"),
            "opened_magnifier": bool(scan.get("opened_magnifier")),
            "support_card_slots": copy.deepcopy(scan.get("support_card_slots") or []),
            "ocr": copy.deepcopy(scan.get("ocr") or []),
            "values": {"actual_setup": copy.deepcopy(normalized_setup),
                       "outcome": copy.deepcopy(outcome)},
        })
        record["latest_screen_scan"] = record["screen_scans"][-1]
        record["updated_at"] = now_iso()
        self._quality(record)
        RuntimeControl._atomic_write(self.path(journey_id), record)
        return record
