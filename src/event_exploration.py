"""Empirical event sampling; collected evidence never rewrites verified rules."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from pathlib import Path

from .event_effects import DEFAULT_DIRECTION_WEIGHTS, STAT_ALIASES
from .journey_record import now_iso
from .runtime_control import RuntimeControl


def normalize(text):
    return re.sub(r"[^\w\u4e00-\u9fff]", "", str(text or "")).lower()


def receipt_effects(observation):
    """Extract explicit result numbers, never training previews or options."""
    page = observation.get("page")
    if page not in {"属性结算", "奖励继续", "事件后续"}:
        return {"numeric": {}, "qualitative": [], "ambiguous": []}
    labels = {**STAT_ALIASES, "耐力": "stamina_pct", "潜质点数": "potential_points",
              "潜质点": "potential_points", "古币": "money", "金币": "money", "心情": "mood_delta"}
    candidates, qualitative = {}, []
    rows = [row for row in observation.get("ocr", []) if row.get("confidence", 0) >= .85]
    for row in rows:
        text = re.sub(r"\s+", "", row["text"]).replace("−", "-").replace("＋", "+")
        for label, field in labels.items():
            if page == "事件后续" and not any(word in text for word in ("提升了", "增加了", "减少了", "恢复了", "获得了")):
                continue
            match = re.search(re.escape(label) + r"(?:提升了|增加了|减少了|降低了|恢复了|获得了)?[:：]?([+-]\d{1,5})(?![\d%％/])", text)
            if match:
                candidates.setdefault(field, set()).add(int(match.group(1)))
        if re.search(r"(?:获得|习得|失去|解除|状态|潜能|潜力|增益|BUFF|折扣)", text, re.I):
            qualitative.append(row["text"])
    ambiguous = [field for field, numbers in candidates.items() if len(numbers) != 1]
    return {"numeric": {field: next(iter(numbers)) for field, numbers in candidates.items() if len(numbers) == 1 and field not in ambiguous},
            "qualitative": list(dict.fromkeys(qualitative)), "ambiguous": ambiguous}


def aggregate_effects(frames):
    values, qualitative, ambiguous = {}, [], set()
    for frame in frames:
        effect = receipt_effects(frame)
        for field, number in effect["numeric"].items():
            values.setdefault(field, set()).add(number)
        qualitative.extend(effect["qualitative"])
        ambiguous.update(effect["ambiguous"])
    ambiguous.update(field for field, numbers in values.items() if len(numbers) > 1)
    return {"numeric": {field: next(iter(numbers)) for field, numbers in values.items() if len(numbers) == 1 and field not in ambiguous},
            "qualitative": list(dict.fromkeys(qualitative)), "ambiguous": sorted(ambiguous)}


class EventExplorationStore:
    # A branch can legitimately only change a buff, potential, item, mood, or
    # a later screen.  Requiring numeric OCR forever would keep such an event
    # in exploration mode forever.  Two completed observations are enough to
    # make a stable empirical choice; numeric effects are used when available
    # and qualitative/transition evidence is retained alongside them.
    MIN_OBSERVATIONS = 2

    def __init__(self, root):
        self.path = Path(root) / "runtime" / "event_exploration.json"

    def read(self):
        if not self.path.exists():
            return {"schema_version": 2, "events": {}}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("events"), dict):
            raise ValueError("事件探索记录损坏，保留原文件")
        return data

    @staticmethod
    def event_key(marker, rows, context):
        identity = {"marker": normalize(marker), "options": [normalize(row["text"]) for row in rows], "context": context}
        raw = json.dumps(identity, ensure_ascii=False, sort_keys=True)
        return "explore_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

    @staticmethod
    def _samples(option):
        return [attempt for attempt in option.get("attempts", [])
                if attempt.get("phase") == "observed" and attempt.get("effects", {}).get("numeric")]

    @staticmethod
    def _observations(option):
        return [attempt for attempt in option.get("attempts", [])
                if attempt.get("phase") == "observed"]

    @staticmethod
    def score(effects, direction, weights=None):
        attr = dict(DEFAULT_DIRECTION_WEIGHTS.get(direction, DEFAULT_DIRECTION_WEIGHTS["attack"]))
        if isinstance(weights, dict):
            attr.update(weights)
        attr.update(stamina_pct=.25, potential_points=.5, money=.5, mood_delta=5)
        return round(sum(value * float(attr.get(field, 0)) for field, value in effects.items()), 2)

    def select(self, marker, rows, context, direction, weights=None):
        data = self.read()
        key = self.event_key(marker, rows, context)
        record = data["events"].get(key) or {
            "key": key, "marker": marker, "context": context, "created_at": now_iso(),
            "options": [{"index": i, "text": row["text"], "attempts": []} for i, row in enumerate(rows, 1)]}
        enabled = [option for option, row in zip(record["options"], rows) if row.get("enabled")]
        if not enabled:
            raise ValueError("没有识别到可用事件选项")
        visits = lambda option: sum(attempt.get("phase") == "observed" for attempt in option.get("attempts", []))
        unseen = [option for option in enabled if not visits(option)]
        # Numeric samples improve the score, but are optional.  A qualitative
        # result or an observed transition is still a completed sample.
        incomplete = [option for option in enabled
                      if len(self._observations(option)) < self.MIN_OBSERVATIONS]
        scores = [{"index": option["index"], "score": round(sum(self.score(attempt["effects"]["numeric"], direction, weights)
                  for attempt in self._samples(option)) / len(self._samples(option)), 2),
                  "sample_count": len(self._samples(option))} for option in enabled if self._samples(option)]
        if unseen or incomplete:
            selected = min(unseen or incomplete, key=lambda option: (
                visits(option), len(self._observations(option)), option["index"]))
            mode, reason = "explore", ("未测试分支优先" if unseen else "补充效果样本")
        else:
            if scores:
                best = max(scores, key=lambda item: (item["score"], -item["index"]))
                selected = next(option for option in enabled if option["index"] == best["index"])
                mode, reason = "empirical", "按当前方向的实测数字平均收益选择"
            else:
                # No numeric receipt is available after the bounded sample
                # budget.  Keep the journey moving with a deterministic
                # branch and preserve the qualitative evidence for later
                # manual/table review instead of safe-pausing indefinitely.
                selected = min(enabled, key=lambda option: option["index"])
                mode, reason = "empirical_qualitative", "效果无可量化数字，按稳定回退分支继续旅程"
        record.update(updated_at=now_iso(), enabled_indices=[option["index"] for option in enabled],
                      branch_coverage_complete=not unseen, effects_ready=not incomplete,
                      scores=scores, decision_mode=mode)
        data["events"][key] = record
        RuntimeControl._atomic_write(self.path, data)
        return {"key": key, "index": selected["index"], "mode": mode, "reason": reason,
                "scores": scores, "branch_coverage_complete": not unseen, "effects_ready": not incomplete}

    def put_attempt(self, attempt):
        data = self.read()
        record = data["events"][attempt["key"]]
        option = next(item for item in record["options"] if int(item["index"]) == int(attempt["index"]))
        option.setdefault("attempts", []).append(copy.deepcopy(attempt))
        record["updated_at"] = now_iso()
        record["branch_coverage_complete"] = all(any(item.get("phase") == "observed" for item in opt["attempts"])
                                                  for opt in record["options"])
        record["effects_ready"] = all(len(self._observations(opt)) >= self.MIN_OBSERVATIONS
                                       for opt in record["options"])
        RuntimeControl._atomic_write(self.path, data)


def new_attempt(selection, marker, rows, before, run_id, journey_id, target_revision):
    return {**selection, "id": uuid.uuid4().hex, "marker": marker, "options": copy.deepcopy(rows),
            "phase": "prepared", "started_at": now_iso(), "run_id": run_id,
            "journey_id": journey_id, "target_revision": target_revision,
            "before": copy.deepcopy(before), "frames": [], "effects": {}, "input_ids": []}
