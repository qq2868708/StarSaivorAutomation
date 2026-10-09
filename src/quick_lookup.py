"""Read-only quick lookup for rescuer and Arcana screens.

The two screens share several labels with the journey-information panel, but
they do not have identical semantics.  This module keeps the raw OCR and
screenshots, extracts only conservative fields, and exposes an explicit
mapping to journey-record fields.  It never clicks or changes the game state.
"""
from __future__ import annotations

import copy
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2

from .capture import ScreenCapture, is_window_foreground
from .controller import Controller
from .recognition import HybridOCR
from .runtime_control import RuntimeControl


_CJK = re.compile(r"^[\u3400-\u9fff·]{2,24}$")
_CJK_SHORT = re.compile(r"^[\u3400-\u9fff·]{2,10}$")
_LEVEL = re.compile(r"^Lv\.?\s*(\d{1,3})(?:\s*/\s*(\d{1,3}))?\b", re.I)
_NUMBER = re.compile(r"^[+-]?[\d,]+(?:\.\d+)?%?$")
_COUNT = re.compile(r"^(\d{1,4})\s*/\s*(\d{1,4})$")
_PERCENT = re.compile(r"^[+-]?[\d,.]+%$")
_RARITY = re.compile(r"(?<![A-Z])(SSR|SR|R|N)(?:(?:X|★|☆|星)){0,4}(?![A-Z])", re.I)

_UI_TEXT = {
    "救援者", "救援者介绍", "潜质", "基本", "等级", "共鸣", "等级重置",
    "能力值详细信息", "圣龛能力值信息", "旅程初始信息", "突破极限",
    "阿尔克那", "支援卡", "卡组", "详情", "返回", "关闭", "未习得",
    "旅程信息", "继承星辰档案", "训练效果", "训练失败率", "旅程初始效果",
}
_STAT_LABELS = {
    "攻击力": "attack",
    "生命力": "health",
    "生命值": "health",
    "防御力": "defense",
    "速度": "speed",
    "暴击率": "crit_rate",
    "暴击伤害": "crit_damage",
    "效果命中": "effect_hit",
    "效果抗性": "effect_resistance",
}


def _bounds(box):
    try:
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        if len(xs) != 4 or len(ys) != 4:
            return None
        return min(xs), min(ys), max(xs), max(ys)
    except (TypeError, ValueError):
        return None


def _rows(items: Iterable[Any]) -> list[dict[str, Any]]:
    rows = []
    for item in items or []:
        try:
            text, confidence, box = item[0], item[1], item[2]
            bounds = _bounds(box)
            if not str(text).strip() or bounds is None:
                continue
            rows.append({"text": str(text).strip(), "confidence": float(confidence),
                         "box": box, "bounds": bounds})
        except (IndexError, TypeError, ValueError):
            continue
    return rows


def _center(row):
    left, top, right, bottom = row["bounds"]
    return (left + right) / 2, (top + bottom) / 2


def _texts(rows):
    return " ".join(row["text"] for row in rows)


def _candidate_text(row, *, short=False):
    text = row["text"].replace(" ", "").replace("\n", "").strip()
    if text in _UI_TEXT or text.upper() in {value.upper() for value in _UI_TEXT}:
        return False
    return bool((_CJK_SHORT if short else _CJK).fullmatch(text))


def _normal_number(text):
    raw = text.replace(",", "").replace("，", "").strip()
    if raw.endswith("%"):
        try:
            return float(raw[:-1])
        except ValueError:
            return None
    try:
        value = float(raw)
        return int(value) if value.is_integer() else value
    except ValueError:
        return None


def _nearby_value(rows, label, *, right_only=True):
    """Find one numeric OCR row immediately to the right of a label."""
    lx, ly = _center(label)
    candidates = []
    for index, row in enumerate(rows):
        if row is label or row["confidence"] < 0.78 or not _NUMBER.fullmatch(row["text"]):
            continue
        left, top, right, bottom = row["bounds"]
        cx, cy = _center(row)
        if right_only and left < lx:
            continue
        if abs(cy - ly) > max(36, (bottom - top) * 1.8):
            continue
        distance = abs(cx - lx) + abs(cy - ly) * 1.5
        candidates.append((distance, index, row))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: item[0])
    if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 8:
        return None, None
    _, index, row = candidates[0]
    return _normal_number(row["text"]), index


def _extract_level(rows):
    for index, row in enumerate(rows):
        match = _LEVEL.search(row["text"])
        if match and row["confidence"] >= 0.72:
            result = {"level": int(match.group(1)), "evidence": [index]}
            if match.group(2):
                result["level_max"] = int(match.group(2))
            return result
    return {}


def _extract_rarity(rows):
    for index, row in enumerate(rows):
        match = _RARITY.search(row["text"])
        if match and row["confidence"] >= 0.72:
            return {"rarity": match.group(1).upper(), "evidence": [index]}
    return {}


def _extract_stats(rows):
    stats = {}
    evidence = {}
    for index, row in enumerate(rows):
        label = row["text"].replace(" ", "").strip()
        key = _STAT_LABELS.get(label)
        if not key:
            continue
        value, value_index = _nearby_value(rows, row)
        if value is None:
            continue
        stats[key] = value
        evidence[key] = [index, value_index]
    return stats, evidence


def _extract_effects(rows):
    """Keep effect text grouped by visible section headings."""
    headings = {"旅程初始效果": "journey_start", "训练效果": "training",
                "训练失败率": "failure_rate", "失败率效果": "failure_rate"}
    effects: dict[str, list[dict[str, Any]]] = {key: [] for key in headings.values()}
    current = None
    for index, row in enumerate(rows):
        text = row["text"].strip()
        if text in headings:
            current = headings[text]
            continue
        if current and row["confidence"] >= 0.65:
            if text in _UI_TEXT or _LEVEL.search(text) or _RARITY.fullmatch(text):
                continue
            effects[current].append({"text": text, "evidence": [index]})
    return {key: value for key, value in effects.items() if value}


def _extract_names(rows, *, x_min=0, x_max=1600, y_min=0, y_max=900, short=False):
    result = []
    for index, row in enumerate(rows):
        left, top, right, bottom = row["bounds"]
        if not (x_min <= left <= x_max and y_min <= top <= y_max):
            continue
        if _candidate_text(row, short=short):
            result.append((index, row))
    return result


def parse_rescuer(items) -> dict[str, Any]:
    """Parse the selected rescuer and keep ambiguous roster names as evidence."""
    rows = _rows(items)
    texts = _texts(rows)
    if "救援者" not in texts:
        raise ValueError("当前画面不是救援者界面")

    # On the 1600x900 client the selected card is the upper-left block.  The
    # name is below the title, while the title is one line above it.
    selected = _extract_names(rows, x_min=110, x_max=360, y_min=125, y_max=235, short=True)
    selected.sort(key=lambda pair: (_center(pair[1])[1], _center(pair[1])[0]))
    character = selected[-1] if selected else None
    title_candidates = [pair for pair in selected if character is None or pair[0] != character[0]]
    title = title_candidates[-1][1]["text"] if title_candidates else None

    result: dict[str, Any] = {"page": "rescuer", "character": None, "title": title,
                              "level": None, "rarity": None, "power": None,
                              "stats": {}, "potential": {}, "roster": [],
                              "uncertain": []}
    if character:
        result["character"] = {"value": character[1]["text"], "evidence": [character[0]]}
    else:
        result["uncertain"].append("character")

    level = _extract_level(rows)
    if level:
        result["level"] = level.get("level")
        result["level_max"] = level.get("level_max")
        result.setdefault("evidence", {})["level"] = level.get("evidence")
    rarity = _extract_rarity(rows)
    if rarity:
        result["rarity"] = rarity["rarity"]
        result.setdefault("evidence", {})["rarity"] = rarity["evidence"]

    for index, row in enumerate(rows):
        left, top, right, bottom = row["bounds"]
        if left > 360 or not _NUMBER.fullmatch(row["text"]):
            continue
        if 220 <= top <= 330:
            value = _normal_number(row["text"])
            if value is not None and isinstance(value, (int, float)):
                result["power"] = {"value": value, "evidence": [index]}
                break

    stats, stat_evidence = _extract_stats(rows)
    result["stats"] = stats
    result.setdefault("evidence", {})["stats"] = stat_evidence

    # The right-hand list is useful for later matching, but only names paired
    # with an adjacent level are returned.  A title/illustration alone is not
    # treated as a confirmed character identity.
    roster = []
    roster_names = _extract_names(rows, x_min=1180, x_max=1530, y_min=130, y_max=820, short=True)
    for index, row in roster_names:
        nearest_level = None
        cx, cy = _center(row)
        for level_index, level_row in enumerate(rows):
            if level_index == index:
                continue
            match = _LEVEL.search(level_row["text"])
            if not match:
                continue
            lx, ly = _center(level_row)
            if abs(ly - cy) < 42 and abs(lx - cx) < 180:
                nearest_level = {"level": int(match.group(1)), "evidence": [level_index]}
                break
        if nearest_level:
            cx, cy = _center(row)
            roster.append({"name": row["text"], "evidence": [index],
                           "center": [cx, cy], **nearest_level})
    result["roster"] = roster
    result["ocr"] = [{key: row[key] for key in ("text", "confidence", "box")} for row in rows]
    return result


def parse_arcanum(items) -> dict[str, Any]:
    """Parse a card list/detail page without identifying artwork by guess."""
    rows = _rows(items)
    texts = _texts(rows)
    if not any(term in texts for term in ("阿尔克那", "支援卡", "旅程初始效果", "训练效果")):
        raise ValueError("当前画面不是阿尔克那/支援卡界面")

    level = _extract_level(rows)
    rarity = _extract_rarity(rows)
    effects = _extract_effects(rows)
    names = _extract_names(rows, x_min=80, x_max=1250, y_min=80, y_max=760, short=False)
    names = [(index, row) for index, row in names if row["text"] not in _UI_TEXT]
    # Prefer a long title near the middle of a detail page.  When there is
    # more than one plausible name, expose all candidates and leave the main
    # value unknown for human confirmation.
    candidates = [{"value": row["text"], "evidence": [index]} for index, row in names]
    longest = max(candidates, key=lambda item: len(item["value"]), default=None)
    same_length = [item for item in candidates if longest and len(item["value"]) == len(longest["value"])]
    card_name = longest if longest and len(same_length) == 1 else None
    character = None
    short_names = [(index, row) for index, row in names if _candidate_text(row, short=True)]
    if len(short_names) == 1:
        character = {"value": short_names[0][1]["text"], "evidence": [short_names[0][0]]}

    card: dict[str, Any] = {"name": card_name, "character": character,
                            "level": level.get("level"), "rarity": rarity.get("rarity"),
                            "effects": effects, "candidates": candidates,
                            "uncertain": []}
    if card_name is None:
        card["uncertain"].append("card_name")
    if character is None:
        card["uncertain"].append("character")
    count = next((row for row in rows if _COUNT.fullmatch(row["text"])), None)
    if count:
        match = _COUNT.fullmatch(count["text"])
        card["collection"] = {"owned": int(match.group(1)), "total": int(match.group(2))}
    result = {"page": "arcanum", "cards": [card], "ocr": [
        {key: row[key] for key in ("text", "confidence", "box")} for row in rows],
        "uncertain": [] if card_name or candidates else ["card_name"]}
    return result


def build_match(rescuer: dict[str, Any] | None, arcanum: dict[str, Any] | None,
                *, screenshots: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build an explicit mapping to journey-end fields."""
    rescuer = rescuer or {}
    arcanum = arcanum or {}
    character = (rescuer.get("character") or {}).get("value")
    cards = []
    for card in arcanum.get("cards") or []:
        name = card.get("name")
        name_value = name.get("value") if isinstance(name, dict) else name
        if not name_value:
            continue
        entry = {"name": name_value}
        for key in ("character", "level", "rarity", "effects", "collection"):
            value = card.get(key)
            if isinstance(value, dict) and "value" in value:
                value = value["value"]
            if value is not None:
                entry[key] = copy.deepcopy(value)
        cards.append(entry)

    status = "matched" if character and cards else "partial"
    result = {
        "status": status,
        "character": character,
        "support_cards": cards or None,
        "journey_end_mapping": {
            "actual_setup.character": {
                "value": character, "source": "rescuer_screen",
                "equivalent": True,
            },
            "actual_setup.support_cards": {
                "value": cards or None, "source": "arcanum_screen",
                "equivalent": bool(cards),
            },
            "outcome.final_stats": {
                "value": None, "source": "rescuer_screen",
                "equivalent": False,
                "reason": "救援者详情属性是角色当前面板值，不等于旅程结束结算属性",
            },
        },
        "screenshots": copy.deepcopy(screenshots or {}),
        "uncertain": sorted(set((rescuer.get("uncertain") or []) +
                                 (arcanum.get("uncertain") or []) +
                                 (["support_cards"] if not cards else []))),
        "observed_at": datetime.now().astimezone().isoformat(),
    }
    return result


class QuickLookupStore:
    """Persist scans in ignored runtime state and keep the latest match."""

    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / "runtime" / "quick_lookup.json"

    def read(self):
        if not self.path.exists():
            return {"rescuer": None, "arcanum": None, "match": None, "history": []}
        try:
            value = __import__("json").loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"rescuer": None, "arcanum": None, "match": None, "history": []}
        return value if isinstance(value, dict) else {"rescuer": None, "arcanum": None, "match": None, "history": []}

    def _write(self, value):
        RuntimeControl._atomic_write(self.path, value)
        return value

    def save_scan(self, kind, scan, screenshot=None):
        if kind not in {"rescuer", "arcanum"}:
            raise ValueError("扫描类型无效")
        data = self.read()
        scan = copy.deepcopy(scan)
        scan.update({"scan_id": scan.get("scan_id") or uuid.uuid4().hex,
                     "observed_at": scan.get("observed_at") or datetime.now().astimezone().isoformat(),
                     "screenshot": screenshot})
        data[kind] = scan
        data["match"] = None
        data.setdefault("history", []).append({"kind": kind, "scan_id": scan["scan_id"],
                                                 "observed_at": scan["observed_at"],
                                                 "screenshot": screenshot})
        data["history"] = data["history"][-20:]
        return self._write(data)

    def save_match(self, match):
        data = self.read()
        data["match"] = copy.deepcopy(match)
        return self._write(data)


class QuickLookupScanner:
    def __init__(self, root, window_title="StarSavior", capture=None, ocr=None,
                 controller=None, sleep=time.sleep):
        self.root = Path(root)
        self.capture = capture or ScreenCapture(window_title)
        self.ocr = ocr or HybridOCR()
        self.controller = controller or Controller(auto_refocus=False)
        self.sleep = sleep

    def _capture(self):
        region = self.capture.find_window()
        hwnd = self.capture.get_hwnd()
        if not region or not hwnd:
            raise ValueError("找不到 StarSavior 游戏窗口")
        frame = self.capture.capture_game()
        return region, hwnd, frame

    def _save(self, scan_id, kind, frame, suffix=None):
        folder = self.root / "logs" / "screenshots"
        folder.mkdir(parents=True, exist_ok=True)
        label = f"_{suffix}" if suffix else ""
        path = folder / f"quick_lookup_{scan_id}_{kind}{label}.png"
        if not cv2.imwrite(str(path), frame):
            raise ValueError("快查截图保存失败")
        return (Path("logs") / "screenshots" / path.name).as_posix()

    def scan(self, kind, traverse_rescuer=True, max_rescuers=20):
        if kind not in {"rescuer", "arcanum"}:
            raise ValueError("扫描类型无效")
        region, hwnd, frame = self._capture()
        if not is_window_foreground(hwnd):
            raise ValueError("请把 StarSavior 切到前台后再读取；扫描器不会抢焦点")
        items = self.ocr.recognize_detailed(frame)
        parser = parse_rescuer if kind == "rescuer" else parse_arcanum
        parsed = parser(items)
        scan_id = uuid.uuid4().hex
        screenshot = self._save(scan_id, kind, frame)
        parsed.update({"scan_id": scan_id, "observed_at": datetime.now().astimezone().isoformat(),
                       "screenshot": screenshot})
        if kind == "rescuer" and traverse_rescuer:
            parsed["roster_details"] = self._traverse_rescuers(
                scan_id, region, hwnd, frame, parsed, max_rescuers=max_rescuers)
        return parsed

    def _traverse_rescuers(self, scan_id, region, hwnd, initial_frame,
                           initial, *, max_rescuers=20):
        """Read each visible right-side rescuer row after an explicit scan.

        The caller opens the rescuer page.  We only click OCR-confirmed rows,
        never refocus the window, and stop if the game leaves the foreground.
        A later run can continue with a different scroll position; this pass
        intentionally does not scroll or guess hidden rows.
        """
        self.controller.set_window_handle(hwnd)
        self.controller.set_game_region(region)
        details = []
        seen = set()
        queue = list(initial.get("roster") or [])[:max_rescuers]
        height = max(1, int(initial_frame.shape[0]))
        for slot, entry in enumerate(queue):
            name = str(entry.get("name") or "").strip()
            center = entry.get("center") or []
            if not name or len(center) != 2 or name in seen:
                continue
            if not is_window_foreground(hwnd):
                break
            try:
                x = min(0.96, max(0.70, float(center[0]) / initial_frame.shape[1]))
                y = min(0.92, max(0.16, float(center[1]) / height))
            except (TypeError, ValueError):
                continue
            self.controller.click_at_percent(x, y, fast=True, source="quick_lookup_rescuer")
            self.sleep(0.35)
            if not is_window_foreground(hwnd):
                break
            try:
                frame = self.capture.capture_game()
                rows = self.ocr.recognize_detailed(frame)
                page = parse_rescuer(rows)
            except (ValueError, RuntimeError):
                continue
            selected = (page.get("character") or {}).get("value")
            if not selected:
                continue
            seen.add(selected)
            details.append({
                "slot": slot + 1,
                "character": selected,
                "level": page.get("level"),
                "rarity": page.get("rarity"),
                "power": page.get("power"),
                "stats": page.get("stats") or {},
                "screenshot": self._save(scan_id, "rescuer", frame, f"{slot + 1}"),
                "ocr": page.get("ocr") or [],
            })
        return details
