"""One-shot quick lookup for rescuer and Arcana screens.

The two screens share several labels with the journey-information panel, but
they do not have identical semantics. OCR and full frames are transient;
the persisted lookup contains names and identity crops only, plus scan
progress metadata. The rescuer scan clicks only
OCR-confirmed controls while the user keeps the game in the foreground.
"""
from __future__ import annotations

import copy
import json
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2

from .capture import ScreenCapture, is_window_foreground
from .quick_lookup_log import append_event
from .controller import Controller
from .recognition import HybridOCR
from .runtime_control import RuntimeControl


_CJK = re.compile(r"^[\u3400-\u9fff·]{2,24}$")
_CJK_SHORT = re.compile(r"^[\u3400-\u9fff·]{1,10}$")
_NAME = re.compile(r"^[A-Za-z0-9\u3400-\u9fff·]{2,14}$")
_LEVEL = re.compile(r"^Lv[.,，]?\s*(\d{1,3})(?:\s*/\s*(\d{1,3}))?", re.I)
_NUMBER = re.compile(r"^[+-]?[\d,，]+(?:\.\d+)?%?$")
_COUNT = re.compile(r"^(\d{1,4})\s*/\s*(\d{1,4})$")
_PERCENT = re.compile(r"^[+-]?[\d,.]+%$")
_RARITY = re.compile(r"(?<![A-Z])(SSR|SR|R|N)(?:(?:X|★|☆|星)){0,4}(?![A-Z])", re.I)

# Verified against the 1600x900 list, initial-info modal and Arcana modal.
# Each crop excludes the level/rarity/star overlays and account UI.
_IDENTITY_BOXES = {
    "rescuer": (542, 92, 1063, 638),
    "journey": (240, 145, 535, 320),
    "arcanum": (215, 245, 470, 580),
}
_IDENTITY_IMAGE = re.compile(
    r"^logs/screenshots/quick_lookup_[0-9a-f]+_(?:rescuer|journey|arcanum)_identity(?:_\d+)?\.png$")
_IDENTITY_UNCERTAIN = {"character", "character_full_name", "journey_initial_character",
                       "card_name", "image", "roster_scan", "roster_details"}
_SCAN_METADATA = {"complete", "attempted", "failed", "retries", "stop_reason",
                  "max_rescuers", "mode", "previous_count", "retained",
                  "new_verified", "replaced", "total", "succeeded",
                  "inventory_complete", "success_rate", "accepted",
                  "acceptance_threshold", "failed_characters"}

_UI_TEXT = {
    "救援者", "救援者介绍", "潜质", "基本", "等级", "共鸣", "等级重置",
    "能力值详细信息", "圣龛能力值信息", "旅程初始信息", "突破极限",
    "阿尔克那", "支援卡", "卡组", "详情", "返回", "关闭", "未习得",
    "旅程信息", "继承星辰档案", "训练效果", "训练失败率", "旅程初始效果",
    "查看详情", "旅程初始信息", "关闭", "返回",
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


class QuickLookupInterrupted(RuntimeError):
    """The user stopped a scan or its game input target became unavailable."""


class QuickLookupEntryFailed(RuntimeError):
    """A bounded entry attempt failed, but the list was safely recovered."""


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
            if isinstance(item, dict):
                text = item.get("text", "")
                confidence = item.get("confidence", 0.0)
                box = item.get("box")
            else:
                text, confidence, box = item[0], item[1], item[2]
            bounds = _bounds(box)
            if not str(text).strip() or bounds is None:
                continue
            rows.append({"text": str(text).strip(), "confidence": float(confidence),
                         "box": box, "bounds": bounds})
        except (IndexError, KeyError, TypeError, ValueError):
            continue
    return rows


def _center(row):
    left, top, right, bottom = row["bounds"]
    return (left + right) / 2, (top + bottom) / 2


def _texts(rows):
    return " ".join(row["text"] for row in rows)


def _compact_text(value):
    return re.sub(r"\s+", "", str(value or "")).strip()


def _name_compatible(expected, observed):
    """Return true only when OCR names provide positive identity evidence.

    Game OCR occasionally changes one Han character (or drops a trailing
    character).  Prefix/one-edit matches are accepted, while arbitrary OCR
    text is never promoted to the requested row name.
    """
    expected = _compact_text(expected)
    observed = _compact_text(observed)
    if not expected or not observed:
        return False
    if expected == observed:
        return True
    # Accept at most one missing trailing glyph, backed by a unique roster
    # match. Similar names with different glyphs are not the same character.
    short, long = sorted((expected, observed), key=len)
    return len(short) >= 2 and len(long) - len(short) == 1 and long.startswith(short)


def _page_kind(items):
    """Classify mutually exclusive rescuer screens from OCR landmarks."""
    rows = _rows(items)
    def label(text, bounds):
        left, top, right, bottom = bounds
        return any(_compact_text(row["text"]) == text and row["confidence"] >= .72
                   and left <= _center(row)[0] <= right and top <= _center(row)[1] <= bottom
                   for row in rows)
    training_labels = {row["text"] for row in rows if row["confidence"] >= .72
                       and 180 <= _center(row)[0] <= 540 and 470 <= _center(row)[1] <= 640}
    if (label("潜质", (580, 145, 770, 220)) and
            label("救援者介绍", (750, 145, 930, 220)) and
            len(training_labels & {"力量", "体力", "韧性", "专注", "保护"}) >= 3):
        return "journey_initial"
    if label("查看详情", (1200, 790, 1570, 885)) and label("救援者", (150, 20, 370, 100)):
        return "rescuer_list"
    if (label("旅程初始信息", (1190, 775, 1570, 845)) and
            label("基本", (1120, 100, 1330, 170)) and
            label("救援者", (150, 20, 370, 100))):
        return "rescuer_detail"
    return None


def _roster_click_percent(entry, frame_width, frame_height):
    """Return the avatar hotspot for an OCR-confirmed right-side row."""
    center = entry.get("center") or []
    if len(center) != 2:
        raise ValueError("角色行坐标无效")
    return 0.94, min(0.92, max(0.16, float(center[1]) / max(1, frame_height)))


def _candidate_text(row, *, short=False):
    text = row["text"].replace(" ", "").replace("\n", "").replace("\r", "").strip()
    if text in _UI_TEXT or text.upper() in {value.upper() for value in _UI_TEXT}:
        return False
    pattern = _CJK_SHORT if short else _CJK
    if pattern.fullmatch(text):
        return True
    return bool(short and _NAME.fullmatch(text) and any("\u3400" <= char <= "\u9fff" for char in text))


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
    selected = _extract_names(rows, x_min=110, x_max=360, y_min=164, y_max=203, short=True)
    selected = [pair for pair in selected if pair[1]["confidence"] >= .65]
    character = max(selected, key=lambda pair: pair[1]["confidence"], default=None)
    title_candidates = _extract_names(rows, x_min=110, x_max=400, y_min=137, y_max=163, short=True)
    title = title_candidates[-1][1]["text"] if title_candidates else None

    result: dict[str, Any] = {"page": "rescuer", "character": None, "title": title,
                              "level": None, "rarity": None, "power": None,
                              "stats": {}, "potential": {}, "roster": [],
                              "uncertain": []}
    if character:
        result["character"] = {"value": character[1]["text"], "evidence": [character[0]]}
    else:
        result["uncertain"].append("character")

    level = _extract_level([row for row in rows if row["bounds"][0] < 360
                            and 200 <= row["bounds"][1] <= 250])
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
    roster_names = _extract_names(rows, x_min=1180, x_max=1530, y_min=155, y_max=775, short=True)
    for index, row in roster_names:
        if row["confidence"] < .70:
            continue
        nearest_level = None
        cx, cy = _center(row)
        for level_index, level_row in enumerate(rows):
            if level_index == index:
                continue
            match = _LEVEL.search(level_row["text"])
            if not match or level_row["confidence"] < .70:
                continue
            lx, ly = _center(level_row)
            if 10 <= cy - ly <= 45 and abs(lx - cx) < 180:
                nearest_level = {"level": int(match.group(1)), "evidence": [level_index]}
                break
        if nearest_level:
            cx, cy = _center(row)
            roster.append({"name": row["text"], "evidence": [index],
                           "center": [cx, cy], **nearest_level,
                           "name_evidence": [index],
                           "level_evidence": nearest_level["evidence"],
                           "confidence": row["confidence"]})
    result["roster"] = sorted(roster, key=lambda item: item["center"][1])
    result["ocr"] = [{key: row[key] for key in ("text", "confidence", "box")} for row in rows]
    return result


def _physical_roster(page):
    """Keep every fully visible level-anchored row, including unreadable names."""
    named = {index: entry for entry in page.get("roster", [])
             for index in entry.get("level_evidence", entry.get("evidence", []))}
    result = []
    for index, row in enumerate(_rows(page.get("ocr") or [])):
        match = _LEVEL.match(row["text"])
        if (not match or row["confidence"] < .5
                or not 1180 <= row["bounds"][0] <= 1530
                or not 145 <= row["bounds"][1] <= 750):
            continue
        entry = named.get(index, {})
        result.append({"name": entry.get("name"), "level": int(match[1]),
                       "confidence": entry.get("confidence", 0),
                       "center": entry.get("center") or [1380, _center(row)[1] + 27]})
    # A selected row can lose the level prefix to its highlight animation.
    # Keep high-confidence name rows on the same 90px grid as their neighbours.
    anchors = list(result)
    for row in _rows(page.get("ocr") or []):
        x, y = _center(row)
        if (row["confidence"] < .9 or not _candidate_text(row, short=True)
                or not 1280 <= row["bounds"][0] <= 1450 or not 168 <= y <= 790
                or any(abs(y - entry["center"][1]) < 40 for entry in result)):
            continue
        if sum(abs(abs(y - entry["center"][1]) - 90) <= 15 for entry in anchors) >= 1:
            result.append({"name": row["text"], "level": None,
                           "confidence": row["confidence"], "center": [x, y]})
    result.sort(key=lambda entry: entry["center"][1])
    filled = []
    for entry in result:
        if filled and 165 <= entry["center"][1] - filled[-1]["center"][1] <= 195:
            filled.append({"name": None, "level": None, "confidence": 0,
                           "center": [1380, (entry["center"][1] + filled[-1]["center"][1]) / 2]})
        filled.append(entry)
    return filled


def _same_roster_rows(left, right):
    """Compare ordered UI rows; levels are transient alignment evidence only."""
    if len(left) != len(right) or not left:
        return False
    named_matches = 0
    uncertain_mismatches = 0
    for a, b in zip(left, right):
        if a.get("level") is not None and b.get("level") is not None and a["level"] != b["level"]:
            return False
        if a.get("name") and b.get("name"):
            if not _name_compatible(a["name"], b["name"]):
                if min(a.get("confidence", 1), b.get("confidence", 1)) >= .95:
                    return False
                uncertain_mismatches += 1
            else:
                named_matches += 1
    # This only aligns list positions; it never verifies a character identity.
    return named_matches >= min(2, len(left)) and uncertain_mismatches <= max(1, len(left) // 5)


def _join_roster_pages(previous, current):
    """Join overlapping ordered pages without collapsing same-name variants."""
    if not previous:
        return copy.deepcopy(current)
    for overlap in range(min(len(previous), len(current)), 1, -1):
        if _same_roster_rows(previous[-overlap:], current[:overlap]):
            result = copy.deepcopy(previous)
            for old, new in zip(result[-overlap:], current[:overlap]):
                if new.get("name") and (not old.get("name") or
                                       new.get("confidence", 0) > old.get("confidence", 0)):
                    old["name"] = new["name"]
                    old["confidence"] = new.get("confidence", 0)
            return result + copy.deepcopy(current[overlap:])
    raise RuntimeError("列表相邻页没有足够的有序重叠证据，未确认完整分母")


def _identity_image_path(value):
    """Return the small identity crop only; never persist full UI captures."""
    path = str(value or "").replace("\\", "/")
    return path if _IDENTITY_IMAGE.fullmatch(path) else None


def _name_value(value):
    return str(value.get("value") or "").strip() if isinstance(value, dict) else str(value or "").strip()


def _full_rescuer_name(name, title=None):
    name = _name_value(name)
    title = _name_value(title)
    if title and name and "：" not in name and ":" not in name:
        return f"{title}：{name}"
    return name


def _identity_uncertain(*values):
    return sorted({item for items in values for item in (items or [])
                   if isinstance(item, str) and item in _IDENTITY_UNCERTAIN})


def sanitize_quick_lookup_scan(kind, scan):
    """Keep only identity fields required for journey-role/card matching.

    OCR, levels, rarity, power, stats, collection counts, and full-screen
    screenshots are deliberately excluded from the persisted quick-lookup
    record. The scanner can still use them transiently for verification.
    """
    if not isinstance(scan, dict):
        return {}
    if kind == "rescuer":
        result = {
            "page": "rescuer",
            "character": {"value": _full_rescuer_name(scan.get("character"), scan.get("title"))}
                         if _name_value(scan.get("character")) else None,
            "identity_image": _identity_image_path(scan.get("identity_image")),
            "roster": [],
            "roster_details": [],
            "roster_scan": {key: copy.deepcopy(value) for key, value in
                            (scan.get("roster_scan") or {}).items() if key in _SCAN_METADATA},
            "uncertain": _identity_uncertain(scan.get("uncertain")),
        }
        for row in scan.get("roster") or []:
            name = str(row.get("name") or "").strip() if isinstance(row, dict) else str(row).strip()
            if name:
                result["roster"].append({"name": name})
        for detail in scan.get("roster_details") or []:
            if not isinstance(detail, dict):
                continue
            initial = detail.get("journey_initial") or {}
            character = _full_rescuer_name(detail.get("character"), detail.get("title"))
            if not character:
                character = str(initial.get("journey_initial_character") or "").strip()
            if not character:
                continue
            row = {
                "character": character,
                "list_character": _name_value(detail.get("list_character") or
                                               detail.get("character") or character),
                "verified": bool(detail.get("verified", True)),
                "journey_initial_character": str(
                    detail.get("journey_initial_character") or
                    initial.get("journey_initial_character") or
                    detail.get("list_character") or detail.get("character") or character).strip(),
                "identity_image": _identity_image_path(detail.get("identity_image")),
                "journey_initial_portrait": _identity_image_path(
                    detail.get("journey_initial_portrait") or
                    initial.get("portrait_screenshot")),
                "uncertain": _identity_uncertain(detail.get("uncertain"), initial.get("uncertain")),
            }
            row["journey_initial"] = {
                "journey_initial_character": row["journey_initial_character"],
                "portrait_screenshot": row["journey_initial_portrait"],
            }
            result["roster_details"].append(row)
        if not result.get("identity_image") and result["roster_details"]:
            result["identity_image"] = result["roster_details"][0].get(
                "journey_initial_portrait")
    if kind == "arcanum":
        cards = []
        for card in scan.get("cards") or []:
            if not isinstance(card, dict):
                continue
            name = card.get("name")
            name_value = name.get("value") if isinstance(name, dict) else name
            if not name_value:
                continue
            cards.append({
                "name": str(name_value).strip(),
                "image": _identity_image_path(card.get("image") or scan.get("identity_image")),
                "uncertain": _identity_uncertain(card.get("uncertain")),
            })
        result = {"page": "arcanum", "cards": cards,
                  "uncertain": _identity_uncertain(scan.get("uncertain"))}
    elif kind != "rescuer":
        raise ValueError("扫描类型无效")
    for key in ("scan_id", "observed_at"):
        if isinstance(scan.get(key), str):
            result[key] = scan[key]
    if isinstance(scan.get("incremental"), dict):
        result["incremental"] = {key: copy.deepcopy(value) for key, value in
                                 scan["incremental"].items() if key in _SCAN_METADATA}
    if isinstance(scan.get("scan_summary"), dict):
        result["scan_summary"] = {key: copy.deepcopy(value) for key, value in
                                  scan["scan_summary"].items() if key in _SCAN_METADATA}
    if isinstance(scan.get("last_detail_scan"), dict):
        last = scan["last_detail_scan"]
        result["last_detail_scan"] = {key: last[key] for key in
                                       ("scan_id", "observed_at", "character") if key in last}
        result["last_detail_scan"]["identity_image"] = _identity_image_path(last.get("identity_image"))
    return result


def parse_arcanum(items) -> dict[str, Any]:
    """Parse a card list/detail page without identifying artwork by guess."""
    rows = _rows(items)
    texts = _texts(rows)
    if not any(term in texts for term in ("阿尔克那", "阿尔克纳", "支援卡", "旅程效果", "旅程初始效果", "训练效果")):
        raise ValueError("当前画面不是阿尔克那/支援卡界面")

    level = _extract_level(rows)
    rarity = _extract_rarity(rows)
    effects = _extract_effects(rows)
    header_names = [(index, row) for index, row in enumerate(rows)
                    if row["confidence"] >= .72 and 720 <= row["bounds"][0] <= 1250
                    and 275 <= row["bounds"][1] <= 330
                    and any("\u3400" <= char <= "\u9fff" for char in row["text"])
                    and row["text"] not in _UI_TEXT]
    names = header_names or _extract_names(rows, x_min=80, x_max=1250, y_min=80, y_max=760, short=False)
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


def _arcanum_detail_confirmed(items):
    rows = _rows(items)
    # Confirm the demonstrated detail modal before cropping the card art.
    # List thumbnails use a different layout and cannot share this crop.
    return (any(row["text"] == "旅程效果" and row["confidence"] >= .72 and
                500 <= row["bounds"][0] <= 720 and 240 <= row["bounds"][1] <= 300
                for row in rows) and
            any(_compact_text(row["text"]).lstrip("·•・") in {"旅程初始效果", "支持委托效果", "训练效果"}
                and row["confidence"] >= .72 and
                735 <= row["bounds"][0] <= 1250 and 370 <= row["bounds"][1] <= 415
                for row in rows))


def _arcanum_panel_confirmed(items):
    rows = _rows(items)
    return (any(row["text"] == "旅程效果" and row["confidence"] >= .72
                and 500 <= row["bounds"][0] <= 720 and 240 <= row["bounds"][1] <= 300
                for row in rows)
            and any(row["text"] in {"专属效果", "旅程事件", "篇章"} and row["confidence"] >= .72
                    and 500 <= row["bounds"][0] <= 720 and 300 <= row["bounds"][1] <= 480
                    for row in rows))


def _arcanum_grid_tiles(items):
    """Only locate card artwork on the verified inventory grid."""
    rows = _rows(items)
    header = any(row["text"] in {"阿尔克那", "阿尔克纳"} and row["confidence"] >= .8
                 and 150 <= row["bounds"][0] <= 370 and 20 <= row["bounds"][1] <= 100
                 for row in rows)
    manage = any(row["text"] == "管理" and row["confidence"] >= .8
                 and 100 <= row["bounds"][0] <= 240 and 115 <= row["bounds"][1] <= 190
                 for row in rows)
    if not header or not manage or _arcanum_panel_confirmed(rows):
        return []
    tiles = []
    for row in rows:
        if (row["text"] not in {"SSR", "SR", "R", "N"} or row["confidence"] < .8
                or not 320 <= row["bounds"][0] <= 1450
                or not 260 <= row["bounds"][1] <= 830):
            continue
        x, _ = _center(row)
        y = row["bounds"][1] - 140
        if y < 175:
            continue
        if not any(abs(x - tile[0]) < 40 and abs(y - tile[1]) < 60 for tile in tiles):
            tiles.append((x, y))
    # R/SR/SSR have different glyph bounds; use the shared grid row baseline.
    groups = []
    for x, y in sorted(tiles, key=lambda tile: tile[1]):
        group = next((row for row in groups if abs(y - row[0][1]) < 40), None)
        if group is None:
            groups.append([(x, y)])
        else:
            group.append((x, y))
    normalized = []
    for group in groups:
        baseline = sorted(y for _, y in group)[len(group) // 2]
        normalized.extend((410 + round((x - 410) / 194) * 194, baseline) for x, _ in group)
    return sorted(normalized, key=lambda tile: (tile[1], tile[0]))


def _arcanum_art_patch(frame, tile):
    x, y = (int(value) for value in tile)
    return frame[y + 55:y + 105, x - 90:x - 35].copy()


def _locate_arcanum_art(frame, patch):
    """Follow a static art fragment after Unity auto-scrolls the selected row."""
    if patch.size == 0:
        raise ValueError("卡面定位片段为空")
    region = frame[155:815, 250:1460]
    scores = cv2.matchTemplate(region, patch, cv2.TM_CCOEFF_NORMED)
    _, score, _, point = cv2.minMaxLoc(scores)
    if score < .92:
        raise ValueError("原卡面不在当前可见区域或已变化")
    x, y = point
    height, width = patch.shape[:2]
    scores[max(0, y-height):y+height, max(0, x-width):x+width] = -1
    if cv2.minMaxLoc(scores)[1] >= .92:
        raise ValueError("相同卡面出现多个位置，未唯一定位")
    return (250 + x + width / 2) / 1600, (155 + y + height / 2) / 900


def _arcanum_owned_count(items):
    candidates = []
    for row in _rows(items):
        match = _COUNT.fullmatch(row["text"].replace(" ", ""))
        if (match and row["confidence"] >= .85
                and 1130 <= row["bounds"][0] <= 1300
                and 95 <= row["bounds"][1] <= 145):
            owned, capacity = map(int, match.groups())
            if 0 < owned <= capacity <= 1000:
                candidates.append(owned)
    return candidates[0] if len(candidates) == 1 else None


def _arcanum_patch_score(frame, tile, patch):
    x, y = (round(value) for value in tile)
    window = frame[max(155, y+35):min(815, y+125), max(250, x-110):min(1460, x-15)]
    if patch.size == 0 or window.shape[0] < patch.shape[0] or window.shape[1] < patch.shape[1]:
        return 0.0
    return cv2.minMaxLoc(cv2.matchTemplate(window, patch, cv2.TM_CCOEFF_NORMED))[1]


def _arcanum_patch_matches(frame, tile, patch):
    return _arcanum_patch_score(frame, tile, patch) >= .92


def _same_arcanum_view(before, after):
    frame, rows = before
    next_frame, next_rows = after
    tiles, next_tiles = _arcanum_grid_tiles(rows), _arcanum_grid_tiles(next_rows)
    return bool(tiles and len(tiles) == len(next_tiles)
                and all(abs(x - a) < 12 and abs(y - b) < 12
                        and _arcanum_patch_matches(next_frame, (a, b), _arcanum_art_patch(frame, (x, y)))
                        for (x, y), (a, b) in zip(tiles, next_tiles)))


def _index_arcanum_tiles(frame, items, patches, *, discover=False):
    """Match only artwork fragments; no card stats or account data are persisted."""
    indices = []
    for tile in _arcanum_grid_tiles(items):
        scores = [_arcanum_patch_score(frame, tile, patch) for patch in patches]
        matches = [i for i, score in enumerate(scores) if score >= .92]
        if len(matches) > 1:
            raise ValueError("多个卡面片段匹配同一位置，未确认翻页")
        if not matches:
            if not discover:
                x, y = map(round, tile)
                raise ValueError(
                    f"出现清点之外的卡片，列表可能已更改；"
                    f"标准化位置=({x},{y})，最高卡面匹配分数={max(scores, default=0.0):.4f}")
            matches = [len(patches)]
            patches.append(_arcanum_art_patch(frame, tile))
        if matches[0] in indices:
            raise ValueError("同一卡面在本页出现多个位置，无法唯一计数")
        indices.append(matches[0])
    if not indices:
        raise ValueError("未确认可见卡片，停止滚轮输入")
    return indices


def parse_journey_initial(items, expected_character: str | None = None) -> dict[str, Any]:
    """Parse the character's ``旅程初始信息`` panel conservatively.

    This panel is a source of journey setup evidence.  Values from the
    rescuer detail page are deliberately not copied into the journey outcome
    fields. Raw OCR is used only in memory to verify the selected character.
    """
    rows = _rows(items)
    texts = _texts(rows)
    # The initial-info view is a modal whose title is not rendered as text.
    # Its stable landmarks are the two tabs and the potential-effect rows.
    if _page_kind(rows) != "journey_initial":
        raise ValueError("当前画面不是旅程初始信息界面")

    excluded = _UI_TEXT | {"力量", "体力", "韧性", "专注", "保护", "攻击力",
                           "生命力", "生命值", "防御力", "速度", "等级"}
    candidates = []
    for index, row in enumerate(rows):
        text = row["text"].replace(" ", "").strip()
        left, top, right, bottom = row["bounds"]
        if row["confidence"] < 0.70 or text in excluded:
            continue
        if not _candidate_text(row, short=True):
            continue
        # The identity is printed below the portrait on the left card.  Keep
        # the narrow area so effect names and background labels cannot become
        # the character by accident.
        if 220 <= left <= 510 and 395 <= top <= 447:
            candidates.append({"value": text, "evidence": [index]})

    unique = []
    seen = set()
    for candidate in candidates:
        if candidate["value"] not in seen:
            seen.add(candidate["value"])
            unique.append(candidate)
    expected = str(expected_character or "").replace(" ", "").strip()
    matching = [item for item in unique if _name_compatible(expected, item["value"])] if expected else []
    character = matching[0] if len(matching) == 1 else (
        unique[0] if not expected and len(unique) == 1 else None)

    result: dict[str, Any] = {
        "page": "journey_initial",
        "character": character,
        "character_candidates": unique,
        "journey_initial_character": character["value"] if character else None,
        "uncertain": [] if character else ["journey_initial_character"],
        "ocr": [{key: row[key] for key in ("text", "confidence", "box")} for row in rows],
    }
    return result


def build_match(rescuer: dict[str, Any] | None, arcanum: dict[str, Any] | None,
                *, screenshots: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build an identity-only mapping, including for callers with legacy data."""
    rescuer = sanitize_quick_lookup_scan("rescuer", rescuer or {})
    arcanum = sanitize_quick_lookup_scan("arcanum", arcanum or {})
    character = (rescuer.get("character") or {}).get("value")
    cards = []
    for card in arcanum.get("cards") or []:
        name = card.get("name")
        name_value = name.get("value") if isinstance(name, dict) else name
        if not name_value:
            continue
        cards.append({"name": name_value, "image": card.get("image"),
                      "uncertain": list(card.get("uncertain") or [])})

    roster_mappings = []
    for detail in rescuer.get("roster_details") or []:
        if detail.get("verified") is False or not detail.get("journey_initial"):
            continue
        initial = detail.get("journey_initial") or {}
        initial_character = (initial.get("journey_initial_character") or
                             (initial.get("character") or {}).get("value"))
        roster_mappings.append({
            "rescuer_character": detail.get("character"),
            "journey_initial_character": initial_character,
            "journey_end_character": initial_character,
            "journey_initial": {"journey_initial_character": initial_character},
            "identity_image": detail.get("identity_image"),
            "journey_initial_portrait": detail.get("journey_initial_portrait") or
                                        initial.get("portrait_screenshot"),
            "uncertain": list(detail.get("uncertain") or []),
        })

    roster_scan = rescuer.get("roster_scan") or {}
    traversal_complete = roster_scan.get("complete", True)
    status = "matched" if (character and cards and traversal_complete) else "partial"
    result = {
        "status": status,
        "character": character,
        "support_cards": cards or None,
        "character_mappings": roster_mappings,
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
        "screenshots": {key: path for key, value in (screenshots or {}).items()
                        if key in {"rescuer", "arcanum"} and
                        (path := _identity_image_path(value))},
        "uncertain": sorted(set((rescuer.get("uncertain") or []) +
                                 (arcanum.get("uncertain") or []) +
                                 (["support_cards"] if not cards else []) +
                                 (["roster_scan"] if not traversal_complete else []) +
                                 (["roster_details"] if not roster_mappings else []))),
        "observed_at": datetime.now().astimezone().isoformat(),
    }
    return result


class QuickLookupStore:
    """Persist scans in ignored runtime state and keep the latest match."""

    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / "runtime" / "quick_lookup.json"

    def _legacy_identity_image(self, value, kind):
        """Convert our old captures to crops without retaining a raw backup."""
        if _identity_image_path(value):
            return _identity_image_path(value)
        relative = str(value or "").replace("\\", "/")
        match = re.fullmatch(
            r"logs/screenshots/quick_lookup_([0-9a-f]+)_"
            r"(rescuer|journey_initial|journey_initial_portrait|arcanum)(_\d+)?\.png",
            relative)
        if not match:
            return None
        scan_id, source_kind, suffix = match.groups()
        if {"rescuer": "rescuer", "journey_initial": "journey",
            "journey_initial_portrait": "journey", "arcanum": "arcanum"}[source_kind] != kind:
            return None
        path = self.root / relative
        folder = (self.root / "logs" / "screenshots").resolve()
        if path.resolve().parent != folder or not path.is_file():
            return None
        frame = cv2.imread(str(path))
        if frame is None:
            return None
        height, width = frame.shape[:2]
        if source_kind == "journey_initial_portrait":
            # The old crop was (170,145,560,408) and included SSR/stars.
            if (width, height) != (390, 263):
                return None
            left, top, right, bottom = (70, 0, 365, 175)
        else:
            if abs(width / height - 1600 / 900) > .02:
                return None
            box = _IDENTITY_BOXES[kind]
            left, top, right, bottom = [int(v * (width / 1600 if i % 2 == 0 else height / 900))
                                        for i, v in enumerate(box)]
        target = folder / f"quick_lookup_{scan_id}_{kind}_identity{suffix or ''}.png"
        if not cv2.imwrite(str(target), frame[top:bottom, left:right]):
            raise OSError("快查识别图脱敏保存失败")
        return (Path("logs") / "screenshots" / target.name).as_posix()

    def _prepare_scan(self, kind, scan):
        """Recover legacy full names and safe artwork before dropping OCR."""
        scan = copy.deepcopy(scan or {})
        if kind == "rescuer":
            scan["identity_image"] = self._legacy_identity_image(
                scan.get("identity_image") or scan.get("screenshot"), "rescuer")
            for detail in scan.get("roster_details") or []:
                if not isinstance(detail, dict):
                    continue
                if not detail.get("title") and detail.get("ocr"):
                    try:
                        detail["title"] = parse_rescuer(detail["ocr"]).get("title")
                    except ValueError:
                        pass
                detail["identity_image"] = self._legacy_identity_image(
                    detail.get("identity_image") or detail.get("screenshot"), "rescuer")
                initial = detail.get("journey_initial") or {}
                detail["journey_initial_portrait"] = self._legacy_identity_image(
                    detail.get("journey_initial_portrait") or initial.get("portrait_screenshot") or
                    detail.get("journey_initial_screenshot") or initial.get("screenshot"), "journey")
        elif kind == "arcanum":
            for card in scan.get("cards") or []:
                if isinstance(card, dict):
                    source = (card.get("image") or scan.get("identity_image") or
                              card.get("screenshot") or scan.get("screenshot"))
                    if not _identity_image_path(source) and not _arcanum_detail_confirmed(scan.get("ocr") or []):
                        source = None
                    card["image"] = self._legacy_identity_image(
                        source, "arcanum")
        return sanitize_quick_lookup_scan(kind, scan)

    def _clean_state(self, value):
        rescuer = self._prepare_scan("rescuer", value["rescuer"]) if value.get("rescuer") else None
        arcanum = self._prepare_scan("arcanum", value["arcanum"]) if value.get("arcanum") else None
        history = []
        for item in list(value.get("history") or [])[-20:]:
            if not isinstance(item, dict):
                continue
            row = {key: copy.deepcopy(item[key]) for key in
                   ("kind", "scan_id", "observed_at", "character") if key in item}
            if _identity_image_path(item.get("identity_image")):
                row["identity_image"] = _identity_image_path(item["identity_image"])
            history.append(row)
        match = build_match(rescuer, arcanum) if value.get("match") else None
        if match and isinstance(value["match"].get("observed_at"), str):
            match["observed_at"] = value["match"]["observed_at"]
        return {"rescuer": rescuer, "arcanum": arcanum, "match": match, "history": history}

    def read(self):
        if not self.path.exists():
            return {"rescuer": None, "arcanum": None, "match": None, "history": []}
        try:
            value = __import__("json").loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError("已有快查文件无法读取，未覆盖原记录") from error
        if not isinstance(value, dict):
            raise ValueError("已有快查文件格式无效，未覆盖原记录")
        clean = self._clean_state(value)
        if clean != value:
            RuntimeControl._atomic_write(self.path, clean)
        return clean

    def _write(self, value):
        value = self._clean_state(value)
        RuntimeControl._atomic_write(self.path, value)
        return value

    def redact_existing(self):
        """Migrate records first, then remove only old quick-lookup captures."""
        data = self.read()
        removed = 0
        folder = (self.root / "logs" / "screenshots").resolve()
        for path in folder.glob("quick_lookup_*.png"):
            relative = (Path("logs") / "screenshots" / path.name).as_posix()
            if path.resolve().parent == folder and not _identity_image_path(relative):
                path.unlink()
                removed += 1
        return {"rescuer_count": len((data.get("rescuer") or {}).get("roster_details") or []),
                "arcanum_count": len((data.get("arcanum") or {}).get("cards") or []),
                "removed_captures": removed}

    @staticmethod
    def _detail_key(detail):
        if not isinstance(detail, dict):
            return ""
        character = str(detail.get("character") or "").strip()
        if character:
            return character
        initial = detail.get("journey_initial") or {}
        return str(initial.get("journey_initial_character") or "").strip()

    @classmethod
    def _merge_detail_rows(cls, previous, incoming):
        """Merge verified rows by character while preserving old evidence."""
        merged = copy.deepcopy(list(previous or []))
        positions = {}
        for index, detail in enumerate(merged):
            key = cls._detail_key(detail)
            if key and key not in positions:
                positions[key] = index
        replaced = 0
        appended = 0
        for detail in incoming or []:
            row = copy.deepcopy(detail)
            key = cls._detail_key(row)
            if not key or row.get("verified") is False:
                continue
            index = positions.get(key)
            if index is None:
                # Upgrade legacy short-name rows only when the match is unique.
                # Two complete names with different titles remain separate.
                candidates = [i for i, old in enumerate(merged)
                              if old.get("list_character") == row.get("list_character") and
                              ("：" not in key or "：" not in cls._detail_key(old))]
                if len(candidates) == 1:
                    index = candidates[0]
                    if "：" in cls._detail_key(merged[index]) and "：" not in key:
                        row["character"] = merged[index]["character"]
            if index is not None:
                merged[index] = cls._merge_known(merged[index], row)
                positions[cls._detail_key(merged[index])] = index
                replaced += 1
            else:
                if key:
                    positions[key] = len(merged)
                merged.append(row)
                appended += 1
        return merged, replaced, appended

    @staticmethod
    def _merge_known(previous, incoming):
        """An unread name/image must not erase an earlier confirmed value."""
        result = copy.deepcopy(previous)
        for key, value in incoming.items():
            if value is None or value == "":
                continue
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = QuickLookupStore._merge_known(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    @staticmethod
    def _merge_named_rows(previous, incoming, name_getter):
        """Merge card-like rows by their stable display name."""
        merged = copy.deepcopy(list(previous or []))
        positions = {}
        for index, row in enumerate(merged):
            name = str(name_getter(row) or "").strip()
            if name and name not in positions:
                positions[name] = index
        replaced = 0
        appended = 0
        for row in incoming or []:
            item = copy.deepcopy(row)
            name = str(name_getter(item) or "").strip()
            if name and name in positions:
                merged[positions[name]] = QuickLookupStore._merge_known(merged[positions[name]], item)
                replaced += 1
            else:
                if name:
                    positions[name] = len(merged)
                merged.append(item)
                appended += 1
        return merged, replaced, appended

    def save_scan(self, kind, scan, screenshot=None):
        if kind not in {"rescuer", "arcanum"}:
            raise ValueError("扫描类型无效")
        data = self.read()
        scan = self._prepare_scan(kind, scan)
        scan.update({"scan_id": scan.get("scan_id") or uuid.uuid4().hex,
                     "observed_at": scan.get("observed_at") or datetime.now().astimezone().isoformat()})
        previous = data.get(kind) if isinstance(data.get(kind), dict) else {}
        if kind == "rescuer":
            merged_scan = copy.deepcopy(previous)
            for key, value in scan.items():
                if key not in {"roster", "roster_details", "roster_scan", "uncertain"} and value is not None:
                    merged_scan[key] = copy.deepcopy(value)
            merged_roster, _, _ = self._merge_named_rows(
                previous.get("roster") or [], scan.get("roster") or [],
                lambda row: row.get("name") if isinstance(row, dict) else row)
            merged_scan["roster"] = merged_roster
            merged_scan["uncertain"] = list(dict.fromkeys(
                list(previous.get("uncertain") or []) + list(scan.get("uncertain") or [])))
            details, replaced, appended = self._merge_detail_rows(
                previous.get("roster_details") or [], scan.get("roster_details") or [])
            current_scan = copy.deepcopy(scan.get("roster_scan") or {})
            current_scan.update({
                "mode": "incremental",
                "previous_count": len(previous.get("roster_details") or []),
                "retained": max(0, len(previous.get("roster_details") or []) - replaced),
                "new_verified": appended,
                "replaced": replaced,
            })
            merged_scan["roster_details"] = details
            merged_scan["roster_scan"] = current_scan
            scan = merged_scan
        elif kind == "arcanum":
            merged_scan = copy.deepcopy(previous)
            for key, value in scan.items():
                if key != "cards":
                    merged_scan[key] = copy.deepcopy(value)
            cards, replaced, appended = self._merge_named_rows(
                previous.get("cards") or [], scan.get("cards") or [],
                lambda row: ((row.get("name") or {}).get("value")
                             if isinstance(row.get("name"), dict)
                             else row.get("name")))
            merged_scan["cards"] = cards
            merged_scan["incremental"] = {
                "previous_count": len(previous.get("cards") or []),
                "retained": max(0, len(previous.get("cards") or []) - replaced),
                "new_verified": appended,
                "replaced": replaced,
                "total": len(cards),
            }
            scan = merged_scan
        data[kind] = scan
        data["match"] = None
        data.setdefault("history", []).append({"kind": kind, "scan_id": scan["scan_id"],
                                                 "observed_at": scan["observed_at"]})
        data["history"] = data["history"][-20:]
        return self._write(data)

    def merge_rescuer_detail(self, detail, *, page=None, scan_id=None,
                             screenshot=None, scan_complete=None):
        """Merge one verified rescuer detail into the existing roster scan."""
        if not isinstance(detail, dict) or detail.get("verified") is False:
            raise ValueError("只能补充已确认的救援者记录")
        data = self.read()
        rescuer = copy.deepcopy(data.get("rescuer") or {})
        clean_page = self._prepare_scan("rescuer", page or {})
        if page:
            for key in ("page", "character", "roster", "uncertain"):
                if key in page:
                    if key == "roster":
                        rescuer[key], _, _ = self._merge_named_rows(
                            rescuer.get("roster") or [], clean_page.get("roster") or [],
                            lambda row: row.get("name") if isinstance(row, dict) else row)
                    elif key == "uncertain":
                        rescuer[key] = list(dict.fromkeys(
                            list(rescuer.get("uncertain") or []) +
                            list(clean_page.get("uncertain") or [])))
                    elif clean_page.get(key) is not None:
                        rescuer[key] = copy.deepcopy(clean_page.get(key))
        rescuer.setdefault("page", "rescuer")
        rescuer.setdefault("roster_details", [])
        detail = copy.deepcopy(detail)
        if page and not detail.get("title") and _name_compatible(
                _name_value(detail.get("character")), _name_value(page.get("character"))):
            detail["title"] = page.get("title")
        clean_details = self._prepare_scan("rescuer", {"roster_details": [detail]})["roster_details"]
        if not clean_details:
            raise ValueError("补充记录缺少已确认角色名")
        clean_detail = clean_details[0]
        key = str(clean_detail.get("character") or "").strip()
        if not key:
            key = str((detail.get("journey_initial") or {}).get(
                "journey_initial_character") or "").strip()
        if not key:
            raise ValueError("补充记录缺少已确认角色名")
        detail = clean_detail
        detail["character"] = key
        detail["verified"] = True
        rescuer["roster_details"], replaced, _ = self._merge_detail_rows(
            rescuer["roster_details"], [detail])
        rescuer["last_detail_scan"] = {
            "scan_id": scan_id or uuid.uuid4().hex,
            "observed_at": datetime.now().astimezone().isoformat(),
            "character": key,
            "identity_image": detail.get("journey_initial_portrait"),
        }
        roster_scan = copy.deepcopy(rescuer.get("roster_scan") or {})
        roster_scan.update({"mode": "incremental", "attempted": len(rescuer["roster_details"]),
                            "retained": max(0, len(rescuer["roster_details"]) - 1),
                            "new_verified": 0 if replaced else 1,
                            "replaced": 1 if replaced else 0})
        if scan_complete is not None:
            roster_scan["complete"] = bool(scan_complete)
            if not scan_complete:
                roster_scan["accepted"] = False
        rescuer["roster_scan"] = roster_scan
        data["rescuer"] = rescuer
        data["match"] = None
        data.setdefault("history", []).append({
            "kind": "rescuer_detail",
            "scan_id": rescuer["last_detail_scan"]["scan_id"],
            "observed_at": rescuer["last_detail_scan"]["observed_at"],
            "character": key,
            "identity_image": detail.get("journey_initial_portrait"),
        })
        data["history"] = data["history"][-20:]
        return self._write(data)

    def save_match(self, match):
        data = self.read()
        data["match"] = copy.deepcopy(match)
        return self._write(data)


class QuickLookupScanner:
    def __init__(self, root, window_title="StarSavior", capture=None, ocr=None,
                 controller=None, sleep=time.sleep, clock=time.monotonic, progress=None,
                 should_stop=None, on_rescuer_entry=None, on_arcanum_entry=None):
        self.root = Path(root)
        self.capture = capture or ScreenCapture(window_title)
        self.ocr = ocr or HybridOCR()
        # A 10 ms fast click can land entirely between Unity input frames.
        # Maintenance clicks use the normal hold and a short settle delay.
        self.controller = controller or Controller(click_delay=.15, auto_refocus=False)
        self.sleep = sleep
        self.clock = clock
        self.progress = progress
        self.should_stop = should_stop or (lambda: False)
        self.on_rescuer_entry = on_rescuer_entry
        self.on_arcanum_entry = on_arcanum_entry
        self.completed_count = 0
        self.failed_characters = []
        self.verified_identities = set()
        self.inventory_complete = False
        self.scan_total = None
        self.retry_count = 0
        self.last_observation = None
        self.stage = "waiting_foreground"
        self.phase = None
        self.scan_kind = None
        self.task_id = None
        self.transitions = []
        self.discovered_roster = []

    def _check_active(self, hwnd=None):
        if self.should_stop():
            raise QuickLookupInterrupted("用户已停止数据读取")
        if hwnd is not None and not is_window_foreground(hwnd):
            raise QuickLookupInterrupted("游戏失去前台，读取安全停止")

    def _capture(self):
        region = self.capture.find_window()
        hwnd = self.capture.get_hwnd()
        if not region or not hwnd:
            raise ValueError("找不到 StarSavior 游戏窗口")
        if not is_window_foreground(hwnd):
            raise ValueError("请把 StarSavior 切到前台后再读取；扫描器不会抢焦点")
        frame = self.capture.capture_game()
        return region, hwnd, frame

    def wait_for_foreground(self, timeout=120.0, interval=0.25):
        """Wait for the user to foreground the game without changing focus."""
        deadline = self.clock() + max(0.0, float(timeout))
        while self.clock() < deadline:
            self._check_active()
            region = self.capture.find_window()
            hwnd = self.capture.get_hwnd()
            if region and hwnd and is_window_foreground(hwnd):
                return region, hwnd
            self.sleep(interval)
        raise ValueError("等待游戏前台超时；请保持 StarSavior 前台后重新点击扫描")

    def _save_crop(self, scan_id, kind, frame, box, suffix=None):
        """Save a verified sub-image (used for the initial portrait)."""
        height, width = frame.shape[:2]
        scale_x, scale_y = width / 1600.0, height / 900.0
        left, top, right, bottom = [int(value * (scale_x if index % 2 == 0 else scale_y))
                                    for index, value in enumerate(box)]
        left, top = max(0, left), max(0, top)
        right, bottom = min(width, right), min(height, bottom)
        if right <= left or bottom <= top:
            raise ValueError("快查截图裁剪区域无效")
        folder = self.root / "logs" / "screenshots"
        folder.mkdir(parents=True, exist_ok=True)
        label = f"_{suffix}" if suffix else ""
        path = folder / f"quick_lookup_{scan_id}_{kind}{label}.png"
        if not cv2.imwrite(str(path), frame[top:bottom, left:right]):
            raise ValueError("快查截图保存失败")
        return (Path("logs") / "screenshots" / path.name).as_posix()

    def _read_frame(self, hwnd):
        self._check_active(hwnd)
        frame = self.capture.capture_game()
        if frame.shape[:2] != (900, 1600):
            frame = cv2.resize(frame, (1600, 900), interpolation=cv2.INTER_AREA)
        rows = self.ocr.recognize_detailed(frame)
        self._check_active(hwnd)
        self.last_observation = (frame, rows)
        return frame, rows

    def _wait_page(self, hwnd, kind, *, expected_name=None, timeout=12.0, interval=.15):
        """Require page and identity together on two consecutive captures."""
        self.stage = kind
        self._report_progress(current=expected_name)
        deadline = self.clock() + max(.1, float(timeout))
        last_error = None
        accepted = 0
        previous_signature = None
        while self.clock() < deadline:
            self._check_active(hwnd)
            frame, rows = self._read_frame(hwnd)
            try:
                if _page_kind(rows) == kind:
                    signature = None
                    if kind == "rescuer_list":
                        page = parse_rescuer(rows)
                        signature = tuple((entry["name"], round(entry["center"][1] / 4))
                                          for entry in page["roster"])
                    if expected_name:
                        page = (parse_journey_initial(rows, expected_name)
                                if kind == "journey_initial" else parse_rescuer(rows))
                        self._verify_selected(page, expected_name)
                    accepted = (accepted + 1 if signature == previous_signature else 1)
                    previous_signature = signature
                    if accepted >= 2:
                        self.transitions.append({"page": kind, "character": expected_name,
                                                 "confirmed": True})
                        return frame, rows
                else:
                    accepted = 0
                    previous_signature = None
            except (ValueError, RuntimeError) as error:
                last_error = error
                accepted = 0
                previous_signature = None
            self.sleep(interval)
        observed = _page_kind(self.last_observation[1]) if self.last_observation else None
        raise RuntimeError(f"等待页面转场失败：{kind}，识别页面：{observed or 'unknown'}"
                           + (f"（{last_error}）" if last_error else ""))

    @staticmethod
    def _page_identity(page):
        return _compact_text((page.get("character") or {}).get("value"))

    def _verify_selected(self, page, expected):
        observed = self._page_identity(page)
        if not _name_compatible(expected, observed):
            raise RuntimeError(f"点击角色后身份未确认：目标={expected}，识别={observed or '未知'}")
        roster = [str(item.get("name") or "") for item in page.get("roster") or []]
        compatible_rows = [name for name in roster if _name_compatible(expected, name)]
        if len(compatible_rows) > 1:
            raise RuntimeError(f"角色名存在前缀歧义，未确认：目标={expected}")
        return observed

    def _recover_to_rescuer_list(self, region, hwnd, *, expected_name=None):
        """Recover one failed entry without guessing the current page.

        A failed row is local to that row.  We can continue when the current
        screen is provably the list, detail, or initial modal; any unknown
        screen remains a hard stop because sending a blind close/back input
        could attach the next image to the wrong role.
        """
        for _ in range(2):
            frame, rows = self._read_frame(hwnd)
            kind = _page_kind(rows)
            if kind == "rescuer_list":
                return frame, parse_rescuer(rows)
            if kind == "journey_initial":
                self._click_transition(hwnd, region, .889, .203,
                                       "quick_lookup_recover_close_initial",
                                       "journey_initial", "rescuer_detail")
                continue
            if kind == "rescuer_detail":
                frame, list_rows = self._click_transition(
                    hwnd, region, .043, .064, "quick_lookup_recover_back_detail",
                    "rescuer_detail", "rescuer_list")
                return frame, parse_rescuer(list_rows)
            raise RuntimeError("无法安全恢复角色列表，当前页面未确认")
        raise RuntimeError("恢复角色列表超时")

    def _report_progress(self, *, current=None, read_attempts=0, error=None):
        """Publish best-effort progress without coupling the scanner to WebUI."""
        if not self.progress:
            return
        try:
            total = (self.scan_total if self.scan_kind == "arcanum"
                     else len(self.discovered_roster) if self.inventory_complete else None)
            self.progress({"attempted": self.completed_count + len(self.failed_characters),
                           "succeeded": self.completed_count,
                           "failed": len(self.failed_characters), "total": total,
                           "inventory_complete": self.inventory_complete,
                           "success_rate": self.completed_count / total if total else None,
                           "phase": self.phase, "discovered": len(self.discovered_roster),
                           "retries": self.retry_count, "current": current,
                           "read_attempts": read_attempts, "stage": self.stage,
                           "last_retry_error": error})
        except Exception:
            # A UI update must not affect game input or verification.
            pass

    def _record_step(self, scan_id, event, character=None):
        self.stage = event
        self._report_progress(current=character)
        try:
            append_event(self.root, self.scan_kind or "rescuer", event,
                         scan_id=scan_id, task_id=self.task_id, stage=event,
                         character=character)
        except OSError:
            pass

    def _record_retry(self, scan_id, name, attempt, error):
        """Keep retry reasons in ignored runtime logs, bounded in each record."""
        try:
            record = {"scan_id": scan_id, "task_id": self.task_id,
                      "character": name, "attempt": attempt, "stage": self.stage,
                      "error": f"{type(error).__name__}: {error}"}
            if self.last_observation:
                rows = self.last_observation[1]
                record["observed_page"] = _page_kind(rows)
                markers = {"救援者", "查看详情", "基本", "旅程初始信息", "潜质",
                           "救援者介绍", "阿尔克那", "旅程效果", "旅程初始效果"}
                record["markers"] = [{"text": row["text"],
                                      "confidence": round(row["confidence"], 3)}
                                     for row in _rows(rows) if row["text"] in markers]
            append_event(self.root, self.scan_kind or "rescuer", "read_retry", **record)
        except OSError:
            pass

    def _retry_rescuer_entry(self, scan_id, region, hwnd, name, slot,
                             *, select_roster=True, max_attempts=3, row_id=None):
        """Retry a role, then fail it only after recovering a confirmed list."""
        attempts = 0
        recovery_failures = 0
        retry_errors = []
        while True:
            attempts += 1
            self._check_active(hwnd)
            self._report_progress(current=name, read_attempts=attempts)
            try:
                self.stage = "confirm_list"
                list_frame, list_rows = self._read_frame(hwnd)
                if _page_kind(list_rows) != "rescuer_list":
                    raise RuntimeError("当前页面不是救援者列表，等待恢复后重试")
                list_page = parse_rescuer(list_rows)
                if not name:
                    name = self._page_identity(list_page)
                    if not name:
                        raise RuntimeError("当前选中的角色名未确认，重试读取")
                if select_roster:
                    self.stage = "select_rescuer"
                    if row_id is not None:
                        matches = [item for item in self._locate_roster_view(list_page)
                                   if item["row_id"] == row_id]
                        if matches and matches[0].get("name"):
                            name = matches[0]["name"]
                    else:
                        matches = [item for item in list_page.get("roster") or []
                                   if _name_compatible(name, str(item.get("name") or ""))]
                    if len(matches) != 1:
                        raise RuntimeError(f"未能唯一定位角色行：{name}")
                    x, y = _roster_click_percent(matches[0], list_frame.shape[1],
                                                  list_frame.shape[0])
                    self._check_active(hwnd)
                    self.controller.click_at_percent(
                        x, y, fast=False, source="quick_lookup_rescuer")
                else:
                    # Supplement mode starts directly from the selected card;
                    # it never clicks, searches or scrolls the roster.
                    self._verify_selected(list_page, name)
                selected_frame, selected_rows = self._wait_page(
                    hwnd, "rescuer_list", expected_name=name)
                selected_page = parse_rescuer(selected_rows)
                selected = self._verify_selected(selected_page, name)
                if _full_rescuer_name(selected, selected_page.get("title")) in self.verified_identities:
                    raise RuntimeError("选中的是本轮已验证的同名版本，未将重复身份计为新成功")
                detail = self._read_rescuer_detail(
                    scan_id, region, hwnd, selected_frame, selected_page, slot,
                    expected_name=name)
                if not detail.get("journey_initial") or not detail.get("can_continue"):
                    raise RuntimeError("旅程初始信息未确认")
                self._verify_selected(detail["journey_initial"], name)
                # The list identity crop is part of the same evidence unit. A
                # failed write is retried with the character, rather than
                # publishing a record that cannot be audited later.
                screenshot = self._save_crop(scan_id, "rescuer_identity", selected_frame,
                                             _IDENTITY_BOXES["rescuer"], str(slot))
                return selected, selected_page, detail, screenshot, attempts, retry_errors
            except QuickLookupInterrupted:
                raise
            except Exception as error:
                self._check_active(hwnd)
                self.retry_count += 1
                retry_errors.append(f"{type(error).__name__}: {error}")
                retry_errors = retry_errors[-20:]
                self._record_retry(scan_id, name, attempts, error)
                self._report_progress(current=name, read_attempts=attempts,
                                      error=retry_errors[-1])
                try:
                    self._recover_to_rescuer_list(
                        region, hwnd, expected_name=name)
                    recovery_failures = 0
                    if attempts >= max_attempts:
                        raise QuickLookupEntryFailed(
                            f"{name or '当前角色'}连续 {attempts} 次读取失败，已安全返回列表：{error}")
                except QuickLookupEntryFailed:
                    raise
                except QuickLookupInterrupted:
                    raise
                except Exception as recovery_error:
                    self._check_active(hwnd)
                    self._record_retry(scan_id, name, attempts, recovery_error)
                    recovery_failures += 1
                    if recovery_failures >= 3:
                        raise RuntimeError("连续三次无法确认恢复页面，扫描已停止；请查看分类日志") from recovery_error
                    retry_errors.append(
                        f"恢复页面: {type(recovery_error).__name__}: {recovery_error}")
                    retry_errors = retry_errors[-20:]
                self.sleep(min(1.5, 0.25 * attempts))

    def scan(self, kind, traverse_rescuer=True, max_rescuers=100):
        self.scan_kind = kind
        if kind not in {"rescuer", "arcanum"}:
            raise ValueError("扫描类型无效")
        region, hwnd = self.wait_for_foreground()
        region, hwnd, frame = self._capture()
        if frame.shape[:2] != (900, 1600):
            frame = cv2.resize(frame, (1600, 900), interpolation=cv2.INTER_AREA)
        items = self.ocr.recognize_detailed(frame)
        self.last_observation = (frame, items)
        if kind == "rescuer" and _page_kind(items) in {"rescuer_detail", "journey_initial"}:
            frame, page = self._recover_to_rescuer_list(region, hwnd)
            items = page["ocr"]
        if kind == "rescuer" and _page_kind(items) != "rescuer_list":
            raise ValueError("当前不是救援者列表页，请打开救援者界面后重试")
        scan_id = uuid.uuid4().hex
        if kind == "arcanum":
            if _arcanum_grid_tiles(items):
                return self._scan_arcanum_library(scan_id, region, hwnd)
            return self._scan_arcanum_page(scan_id, region, hwnd, frame, items)
        parsed = parse_rescuer(items)
        parsed.update({"scan_id": scan_id, "observed_at": datetime.now().astimezone().isoformat()})
        if kind == "rescuer" and traverse_rescuer:
            self._inventory_rescuers(scan_id, region, hwnd)
            if not self.inventory_complete:
                raise RuntimeError("列表清点不完整，未开始详情采集；请查看 inventory_page 日志")
            details, traversal = self._traverse_rescuers(
                scan_id, region, hwnd, frame, parsed, max_rescuers=max_rescuers)
            parsed["roster_details"] = details
            parsed["roster_scan"] = traversal
            if self.discovered_roster:
                parsed["roster"] = copy.deepcopy(self.discovered_roster)
        if kind == "rescuer":
            parsed["identity_image"] = self._save_crop(
                scan_id, "rescuer_identity", frame, _IDENTITY_BOXES["rescuer"])
        return sanitize_quick_lookup_scan(kind, parsed)

    def _wait_arcanum_page(self, hwnd, kind, timeout=8):
        self.stage = f"arcanum_{kind}"
        self._report_progress()
        deadline = self.clock() + timeout
        previous = None
        previous_view = None
        while self.clock() < deadline:
            frame, items = self._read_frame(hwnd)
            if kind == "grid":
                tiles = _arcanum_grid_tiles(items)
                if tiles and previous_view and _same_arcanum_view(previous_view, (frame, items)):
                    return frame, items
                previous_view = (frame, items) if tiles else None
                self.sleep(.2)
                continue
            elif kind == "panel":
                signature = "panel" if _arcanum_panel_confirmed(items) else None
            else:
                signature = (_name_value(parse_arcanum(items)["cards"][0].get("name"))
                             if _arcanum_detail_confirmed(items) else None)
            if signature and signature == previous:
                return frame, items
            previous = signature
            self.sleep(.2)
        raise RuntimeError(f"阿尔克纳页面确认超时：{kind}")

    def _read_arcanum_card(self, scan_id, frame, items, slot=None):
        if not _arcanum_detail_confirmed(items):
            raise ValueError("未确认阿尔克纳旅程效果详情")
        parsed = parse_arcanum(items)
        name = _name_value(parsed["cards"][0].get("name"))
        # Only the card-title band may supply a live scan name.
        if not name or not any(row["text"] == name and row["confidence"] >= .72
                               and 720 <= row["bounds"][0] <= 1250
                               and 275 <= row["bounds"][1] <= 330 for row in _rows(items)):
            raise ValueError("阿尔克纳卡名未在标题区域确认")
        identity = self._save_crop(scan_id, "arcanum_identity", frame,
                                   _IDENTITY_BOXES["arcanum"], str(slot) if slot else None)
        parsed["cards"][0]["image"] = identity
        parsed.update(scan_id=scan_id, observed_at=datetime.now().astimezone().isoformat())
        return parsed

    def _arcanum_effect_page(self, hwnd, region, frame, items):
        if _arcanum_detail_confirmed(items):
            return frame, items
        if not _arcanum_panel_confirmed(items):
            raise ValueError("当前不是阿尔克纳详情页，未发送输入")
        label = self._find_label(items, {"旅程效果"})
        if not label:
            raise ValueError("旅程效果标签未确认")
        self._click_row(hwnd, region, label, "quick_lookup_arcanum_effect",
                        (frame.shape[1], frame.shape[0]))
        return self._wait_arcanum_page(hwnd, "effects")

    def _close_arcanum(self, scan_id, hwnd, region):
        for attempt in range(3):
            _, rows = self._read_frame(hwnd)
            if _arcanum_grid_tiles(rows):
                return self._wait_arcanum_page(hwnd, "grid")
            if not (_arcanum_panel_confirmed(rows) or _arcanum_detail_confirmed(rows)):
                raise RuntimeError("阿尔克纳当前页面未确认，未猜测关闭位置")
            self._record_step(scan_id, "arcanum_closing")
            self._click_fixed(hwnd, region, .858, .254, "quick_lookup_close_arcanum")
            try:
                result = self._wait_arcanum_page(hwnd, "grid", timeout=4)
                self._record_step(scan_id, "arcanum_returned_to_grid")
                return result
            except QuickLookupInterrupted:
                raise
            except RuntimeError:
                if attempt == 2:
                    raise

    def _scan_arcanum_page(self, scan_id, region, hwnd, frame, items):
        self.phase = "arcanum_cards"
        if _arcanum_detail_confirmed(items) or _arcanum_panel_confirmed(items):
            frame, items = self._arcanum_effect_page(hwnd, region, frame, items)
            parsed = self._read_arcanum_card(scan_id, frame, items)
            self.completed_count = 1
            self.scan_total = 1
            self._record_step(scan_id, "arcanum_card_collected",
                              _name_value(parsed["cards"][0]["name"]))
            parsed["scan_summary"] = {"attempted": 1, "succeeded": 1,
                                      "failed": 0, "total": 1, "complete": True}
            return sanitize_quick_lookup_scan("arcanum", parsed)
        tiles = _arcanum_grid_tiles(items)
        if not tiles:
            raise ValueError("请打开阿尔克纳卡片列表或详情页，当前页面未确认")
        self.scan_total = len(tiles)
        patches = [_arcanum_art_patch(frame, tile) for tile in tiles]
        cards = []
        self._record_step(scan_id, "arcanum_grid_started")
        for index, patch in enumerate(patches, 1):
            self._check_active(hwnd)
            self._report_progress(current=f"卡片 {index}/{len(tiles)}")
            current_frame, current = self._wait_arcanum_page(hwnd, "grid")
            try:
                x, y = _locate_arcanum_art(current_frame, patch)
                self._click_fixed(hwnd, region, x, y,
                                  "quick_lookup_open_arcanum")
                card_frame, card_rows = self._wait_arcanum_page(hwnd, "panel")
                card_frame, card_rows = self._arcanum_effect_page(hwnd, region, card_frame, card_rows)
                parsed = self._read_arcanum_card(scan_id, card_frame, card_rows, index)
                self._record_step(scan_id, "arcanum_card_collected",
                                  _name_value(parsed["cards"][0]["name"]))
                self._close_arcanum(scan_id, hwnd, region)
                if self.on_arcanum_entry:
                    parsed["scan_summary"] = {"attempted": index,
                        "succeeded": self.completed_count + 1, "failed": len(self.failed_characters),
                        "total": len(tiles), "complete": False}
                    self.on_arcanum_entry(parsed)
                cards.extend(parsed["cards"])
                self.completed_count += 1
            except QuickLookupInterrupted:
                raise
            except (ValueError, RuntimeError) as error:
                self.failed_characters.append(f"卡片 {index}")
                self._record_retry(scan_id, f"卡片 {index}", 1, error)
                self._close_arcanum(scan_id, hwnd, region)
            self._report_progress()
        return sanitize_quick_lookup_scan("arcanum", {
            "cards": cards, "scan_id": scan_id,
            "observed_at": datetime.now().astimezone().isoformat(),
            "scan_summary": {"attempted": len(tiles), "succeeded": len(cards),
                             "failed": len(self.failed_characters), "total": len(tiles),
                             "complete": not self.failed_characters,
                             "stop_reason": "部分卡片读取失败，已返回列表" if self.failed_characters else None}})

    def _scroll_arcanum(self, scan_id, hwnd, clicks, *, x=.6):
        before = self._wait_arcanum_page(hwnd, "grid")
        self._check_active(hwnd)
        self.stage = "arcanum_scrolling"
        self._report_progress()
        self.controller.scroll_at_percent(x, .55, clicks=clicks,
                                          source="quick_lookup_arcanum_scroll")
        self.sleep(.3)
        after = self._wait_arcanum_page(hwnd, "grid")
        moved = not _same_arcanum_view(before, after)
        append_event(self.root, "arcanum", "arcanum_scroll", task_id=self.task_id,
                     scan_id=scan_id, direction="down" if clicks < 0 else "up",
                     moved=moved)
        return after, moved

    def _arcanum_to_top(self, scan_id, hwnd):
        unchanged = 0
        for _ in range(80):
            view, moved = self._scroll_arcanum(scan_id, hwnd, 4)
            unchanged = 0 if moved else unchanged + 1
            if unchanged >= 2:
                view, moved = self._scroll_arcanum(scan_id, hwnd, 6, x=.8)
                if not moved:
                    return view
                unchanged = 0
        raise RuntimeError("未确认阿尔克纳列表顶部，停止全库扫描")

    def _arcanum_inventory(self, scan_id, region, hwnd):
        self.controller.set_window_handle(hwnd)
        self.controller.set_game_region(region)
        self.phase = "arcanum_inventory"
        self._record_step(scan_id, "arcanum_inventory_started")
        view = self._arcanum_to_top(scan_id, hwnd)
        expected = _arcanum_owned_count(view[1])
        if expected is None:
            raise RuntimeError("未确认阿尔克纳列表总数，不能判定全库扫描完成")
        self.scan_total = expected
        patches = []
        stagnant = 0
        bottom = False
        previous = None
        for page in range(1, 121):
            if _arcanum_owned_count(view[1]) != expected:
                raise RuntimeError("扫描期间阿尔克纳数量变化，停止扫描")
            visible = _index_arcanum_tiles(*view, patches, discover=True)
            if previous is not None and not set(previous) & set(visible):
                raise RuntimeError("相邻卡片页没有重叠，未确认滚轮没有跳过卡片")
            append_event(self.root, "arcanum", "arcanum_inventory_page", task_id=self.task_id,
                         scan_id=scan_id, page=page, discovered=len(patches), total=expected)
            self._report_progress(current=f"清点 {len(patches)}/{expected}")
            if len(patches) > expected:
                raise RuntimeError("卡面去重数量超过列表总数，未确认全库")
            if stagnant >= 2:
                verified, moved = self._scroll_arcanum(scan_id, hwnd, -4, x=.8)
                if not moved:
                    bottom = True
                    break
                stagnant = 0
                view = verified
                previous = visible
                continue
            previous = visible
            view, moved = self._scroll_arcanum(scan_id, hwnd, -2)
            stagnant = 0 if moved else stagnant + 1
        if not bottom or len(patches) != expected:
            raise RuntimeError(f"全库清点未完成：识别 {len(patches)} / 总数 {expected}，未标记成功")
        self.inventory_complete = True
        self._record_step(scan_id, "arcanum_inventory_finished")
        return patches, self._arcanum_to_top(scan_id, hwnd)

    def _scan_arcanum_library(self, scan_id, region, hwnd):
        patches, view = self._arcanum_inventory(scan_id, region, hwnd)
        self.phase = "arcanum_library"
        attempted = set()
        cards = []
        last_error = None
        stagnant = 0
        bottom = False
        for page in range(1, 121):
            self._check_active(hwnd)
            if _arcanum_owned_count(view[1]) != self.scan_total:
                raise RuntimeError("扫描期间阿尔克纳数量变化")
            visible = _index_arcanum_tiles(*view, patches)
            for index in visible:
                if index in attempted:
                    continue
                label = f"卡片 {index + 1}/{self.scan_total}"
                self._report_progress(current=label)
                success = False
                for attempt in range(1, 3):
                    self._check_active(hwnd)
                    current_frame, current = self._wait_arcanum_page(hwnd, "grid")
                    try:
                        x, y = _locate_arcanum_art(current_frame, patches[index])
                        self._click_fixed(hwnd, region, x, y, "quick_lookup_open_arcanum")
                        card_frame, card_rows = self._wait_arcanum_page(hwnd, "panel")
                        card_frame, card_rows = self._arcanum_effect_page(
                            hwnd, region, card_frame, card_rows)
                        parsed = self._read_arcanum_card(scan_id, card_frame, card_rows, index + 1)
                        self._record_step(scan_id, "arcanum_card_collected",
                                          _name_value(parsed["cards"][0]["name"]))
                        view = self._close_arcanum(scan_id, hwnd, region)
                        if self.on_arcanum_entry:
                            parsed["scan_summary"] = {
                                "attempted": len(attempted) + 1, "succeeded": len(cards) + 1,
                                "failed": len(self.failed_characters), "total": self.scan_total,
                                "mode": "library", "complete": False}
                            self.on_arcanum_entry(parsed)
                        cards.extend(parsed["cards"])
                        self.completed_count += 1
                        success = True
                        break
                    except QuickLookupInterrupted:
                        raise
                    except (ValueError, RuntimeError) as error:
                        last_error = str(error)
                        self.retry_count += 1
                        self._record_retry(scan_id, label, attempt, error)
                        view = self._close_arcanum(scan_id, hwnd, region)
                attempted.add(index)
                if not success:
                    self.failed_characters.append(label)
                self._report_progress()
            view = self._wait_arcanum_page(hwnd, "grid")
            # Selecting a lower row can auto-scroll new cards into view.
            # Process those before issuing another wheel event.
            if any(index not in attempted for index in _index_arcanum_tiles(*view, patches)):
                stagnant = 0
                continue
            if stagnant >= 2:
                view, moved = self._scroll_arcanum(scan_id, hwnd, -4, x=.8)
                if not moved:
                    bottom = True
                    break
                stagnant = 0
                continue
            view, moved = self._scroll_arcanum(scan_id, hwnd, -2)
            stagnant = 0 if moved else stagnant + 1
        exhaustive = bottom and len(attempted) == self.scan_total
        complete = exhaustive and not self.failed_characters
        summary = {"mode": "library", "attempted": len(attempted), "succeeded": len(cards),
                   "failed": len(self.failed_characters), "total": self.scan_total,
                   "inventory_complete": True, "complete": complete,
                   "success_rate": len(cards) / self.scan_total,
                   "failed_characters": self.failed_characters,
                   "stop_reason": None if complete else
                       (last_error or "未尝试完整卡库，已保留成功记录")}
        append_event(self.root, "arcanum", "arcanum_library_summary",
                     task_id=self.task_id, scan_id=scan_id, **summary)
        return sanitize_quick_lookup_scan("arcanum", {
            "cards": cards, "scan_id": scan_id,
            "observed_at": datetime.now().astimezone().isoformat(),
            "scan_summary": summary})

    def scan_selected_rescuer(self):
        """Force-read the currently selected rescuer and return one detail row."""
        self.scan_kind = "rescuer"
        region, hwnd = self.wait_for_foreground()
        self.controller.set_window_handle(hwnd)
        self.controller.set_game_region(region)
        scan_id = uuid.uuid4().hex
        selected, selected_page, detail, screenshot, attempts, retry_errors = (
            self._retry_rescuer_entry(
                scan_id, region, hwnd, None, 1, select_roster=False))
        entry = {
            "slot": 1,
            "character": selected,
            "scan_status": "verified",
            "verified": True,
            "title": selected_page.get("title"),
            "identity_image": screenshot,
        }
        entry.update(detail)
        self.completed_count = 1
        self._report_progress()
        return sanitize_quick_lookup_scan("rescuer", {
            "page": "rescuer",
            "character": selected_page.get("character"),
            "title": selected_page.get("title"),
            "roster": selected_page.get("roster") or [],
            "scan_id": scan_id,
            "observed_at": datetime.now().astimezone().isoformat(),
            "identity_image": screenshot,
            "roster_details": [entry],
            "roster_scan": {"complete": False, "mode": "single",
                            "attempted": 1, "failed": 0,
                            "retries": max(0, attempts - 1),
                            "stop_reason": None, "max_rescuers": 1},
        })

    @staticmethod
    def _missing_roster_rows(page):
        """Check full visible row anchors, allowing extra partial edge names."""
        matched = {index for item in page.get("roster", [])
                   for index in item.get("level_evidence", item.get("evidence", []))}
        return [{"y": round(row["bounds"][1], 1)} for index, row in
                enumerate(_rows(page.get("ocr") or []))
                if index not in matched and _LEVEL.match(row["text"])
                and row["confidence"] >= .5
                and 1180 <= row["bounds"][0] <= 1530
                and 145 <= row["bounds"][1] <= 750]

    @staticmethod
    def _roster_page_complete(page):
        return bool(page.get("roster")) and not QuickLookupScanner._missing_roster_rows(page)

    def _locate_roster_view(self, page):
        visible = _physical_roster(page)
        matches = [index for index in range(len(self.discovered_roster) - len(visible) + 1)
                   if _same_roster_rows(self.discovered_roster[index:index + len(visible)], visible)]
        if len(matches) != 1:
            raise RuntimeError("当前列表行无法与完整清单唯一对齐")
        offset = matches[0]
        return [{**entry, "row_id": offset + i,
                 "name": entry.get("name") or self.discovered_roster[offset + i].get("name")}
                for i, entry in enumerate(visible)]

    def _seek_roster_top(self, hwnd, observe=None):
        previous = None
        unchanged = 0
        for _ in range(32):
            _, rows = self._wait_page(hwnd, "rescuer_list")
            page = parse_rescuer(rows)
            if observe:
                observe(page)
            signature = _physical_roster(page)
            if not signature:
                raise RuntimeError("无法确认列表顶部：角色行未识别")
            unchanged = unchanged + 1 if previous and _same_roster_rows(signature, previous) else 0
            if unchanged >= 2:
                self.controller.scroll_at_percent(.94, .54, clicks=8,
                                                  source="quick_lookup_confirm_top")
                _, verified_rows = self._wait_page(hwnd, "rescuer_list")
                verified = parse_rescuer(verified_rows)
                if observe:
                    observe(verified)
                if _same_roster_rows(_physical_roster(verified), signature):
                    return
                unchanged = 0
            previous = signature
            self.controller.scroll_at_percent(.87, .54, clicks=3,
                                              source="quick_lookup_seek_top")
        raise RuntimeError("未确认列表顶部，不计算完整列表通过率")

    def _inventory_rescuers(self, scan_id, region, hwnd):
        """Census the whole list first, independent of successful detail reads."""
        self.controller.set_window_handle(hwnd)
        self.controller.set_game_region(region)
        self.phase = "inventory"
        self._record_step(scan_id, "inventory_started")
        self._seek_roster_top(hwnd)
        known = []
        previous = None
        unchanged = 0
        moved = False
        for _ in range(64):
            _, rows = self._wait_page(hwnd, "rescuer_list")
            page = parse_rescuer(rows)
            visible = _physical_roster(page)
            if not visible:
                raise RuntimeError("列表清点未识别到角色")
            known = _join_roster_pages(known, visible)
            signature = visible
            same = bool(previous and _same_roster_rows(signature, previous))
            moved = moved or (previous is not None and not same)
            append_event(self.root, "rescuer", "inventory_page",
                         task_id=self.task_id, scan_id=scan_id,
                         characters=[entry["name"] for entry in signature],
                         missing_rows=self._missing_roster_rows(page), counted_rows=len(known))
            unchanged = unchanged + 1 if same else 0
            self.discovered_roster = [{**entry, "row_id": i} for i, entry in enumerate(known)]
            self._report_progress()
            if unchanged >= 2:
                self.controller.scroll_at_percent(.94, .54, clicks=-6,
                                                  source="quick_lookup_confirm_bottom")
                _, verified_rows = self._wait_page(hwnd, "rescuer_list")
                verified = parse_rescuer(verified_rows)
                if _same_roster_rows(_physical_roster(verified), signature):
                    if not moved:
                        raise RuntimeError("未观察到列表实际滚动，无法证明完整列表边界")
                    self.inventory_complete = True
                    break
                unchanged = 0
            previous = signature
            self.controller.scroll_at_percent(.87, .54, clicks=-3,
                                              source="quick_lookup_inventory_scroll")
        else:
            raise RuntimeError("未确认列表底部，不计算完整列表通过率")
        reverse_rows = []
        def observe_reverse(page):
            nonlocal reverse_rows
            rows = _physical_roster(page)
            reverse_rows = _join_roster_pages(rows, reverse_rows) if reverse_rows else rows
            append_event(self.root, "rescuer", "inventory_page",
                         task_id=self.task_id, scan_id=scan_id, direction="up",
                         characters=[row["name"] for row in rows],
                         missing_rows=self._missing_roster_rows(page))
        self._seek_roster_top(hwnd, observe=observe_reverse)
        self.inventory_complete = bool(self.inventory_complete
                                       and _same_roster_rows(reverse_rows, known))
        if self.inventory_complete:
            for entry, reverse in zip(known, reverse_rows):
                if not entry.get("name") and reverse.get("name"):
                    entry["name"] = reverse["name"]
            self.discovered_roster = [{**entry, "row_id": i} for i, entry in enumerate(known)]
        self._record_step(scan_id, "inventory_finished")
        append_event(self.root, "rescuer", "inventory_summary",
                     task_id=self.task_id, scan_id=scan_id, total=len(known),
                     inventory_complete=self.inventory_complete,
                     reverse_total=len(reverse_rows),
                     characters=[entry["name"] for entry in known])

    def _traverse_rescuers(self, scan_id, region, hwnd, initial_frame,
                           initial, *, max_rescuers=100):
        """Read every OCR-confirmed rescuer row, scrolling the right list.

        The caller opens the rescuer page.  Each visible row is entered through
        the detail and journey-initial-info panels before the list is scrolled
        to discover further rows.  Focus is never reclaimed and a foreground
        loss safely stops the pass with already collected entries intact.
        """
        self.controller.set_window_handle(hwnd)
        self.controller.set_game_region(region)
        self.phase = "collect"
        details = []
        seen = set()
        stagnant_scrolls = 0
        next_slot = 1
        complete = False
        stop_reason = None
        list_read_retries = 0
        ordinal_inventory = bool(self.discovered_roster
                                 and all("row_id" in row for row in self.discovered_roster))

        def remember_roster(rows):
            if ordinal_inventory:
                return
            positions = {str(item.get("name") or "").strip(): index
                         for index, item in enumerate(self.discovered_roster)
                         if str(item.get("name") or "").strip()}
            for item in rows or []:
                name = str(item.get("name") or "").strip()
                if not name:
                    continue
                if name in positions:
                    self.discovered_roster[positions[name]] = copy.deepcopy(item)
                else:
                    if self.inventory_complete:
                        self.inventory_complete = False
                    positions[name] = len(self.discovered_roster)
                    self.discovered_roster.append(copy.deepcopy(item))

        while len(seen) < max_rescuers and is_window_foreground(hwnd):
            try:
                current_frame, current_rows = self._wait_page(hwnd, "rescuer_list")
                current_page = parse_rescuer(current_rows)
                entries = (self._locate_roster_view(current_page) if ordinal_inventory
                           else list(current_page.get("roster") or []))
                if not entries:
                    raise RuntimeError("右侧角色列表未识别，继续等待后重试")
                remember_roster(entries)
                list_read_retries = 0
            except QuickLookupInterrupted:
                raise
            except Exception as error:
                list_read_retries += 1
                self._record_retry(scan_id, "<角色列表>", list_read_retries, error)
                if list_read_retries >= 20:
                    stop_reason = f"角色列表连续读取失败：{error}"
                    break
                self._check_active(hwnd)
                self.sleep(min(1.5, 0.25 * list_read_retries))
                continue
            for entry in entries:
                name = str(entry.get("name") or "").strip()
                key = entry["row_id"] if ordinal_inventory else name
                if key in seen or (not name and not ordinal_inventory):
                    continue
                try:
                    if not name:
                        name = f"未识别角色（列表第 {key + 1} 行）"
                        raise QuickLookupEntryFailed("该行名称未确认，计为失败，不发送猜测点击")
                    options = {"row_id": key} if ordinal_inventory else {}
                    selected, page, detail, screenshot, attempts, retry_errors = (
                        self._retry_rescuer_entry(
                            scan_id, region, hwnd, name, next_slot, **options))
                except QuickLookupInterrupted as error:
                    stop_reason = str(error)
                    break
                except QuickLookupEntryFailed as error:
                    seen.add(key)
                    self.failed_characters.append(name)
                    next_slot += 1
                    self._record_step(scan_id, "entry_failed", name)
                    append_event(self.root, "rescuer", "entry_failure_reason",
                                 task_id=self.task_id, scan_id=scan_id,
                                 character=name, error=str(error))
                    if len(seen) >= max_rescuers:
                        break
                    continue
                seen.add(key)
                self.verified_identities.add(_full_rescuer_name(selected, page.get("title")))
                entry_detail = {
                    "slot": next_slot,
                    "character": selected,
                    "scan_status": "verified",
                    "verified": True,
                    "title": page.get("title"),
                    "identity_image": screenshot,
                }
                entry_detail.update(detail)
                details.append(entry_detail)
                if self.on_rescuer_entry:
                    self.on_rescuer_entry(entry_detail, page, scan_id)
                next_slot += 1
                self.completed_count = len(details)
                self._report_progress()
                if len(seen) >= max_rescuers:
                    break
            if stop_reason:
                break
            if len(seen) >= max_rescuers or not is_window_foreground(hwnd):
                break
            signature_before = tuple(item.get("name") for item in entries)
            scroll_error = None
            scrolled_page = None
            for scroll_attempt in range(1, 4):
                try:
                    self._check_active(hwnd)
                    self.controller.scroll_at_percent(
                        0.87, 0.54, clicks=-5,
                        source="quick_lookup_rescuer_scroll")
                    _, scrolled_rows = self._wait_page(hwnd, "rescuer_list")
                    scrolled_page = parse_rescuer(scrolled_rows)
                    if not scrolled_page.get("roster"):
                        raise RuntimeError("滚动后右侧角色列表未识别")
                    scroll_error = None
                    break
                except QuickLookupInterrupted:
                    raise
                except Exception as error:
                    scroll_error = error
                    self._record_retry(scan_id, "<角色列表滚动>", scroll_attempt, error)
                    self._check_active(hwnd)
                    self.sleep(min(1.5, 0.25 * scroll_attempt))
            if scrolled_page is None:
                stop_reason = f"角色列表滚动失败：{scroll_error}"
                break
            signature_after = tuple(item.get("name") for item in scrolled_page.get("roster") or [])
            remember_roster(scrolled_page.get("roster") or [])
            stagnant_scrolls = stagnant_scrolls + 1 if signature_after == signature_before else 0
            if stagnant_scrolls >= 2:
                complete = True
                break
        if len(seen) >= max_rescuers:
            stop_reason = "达到扫描上限"
        elif not stop_reason and stagnant_scrolls >= 2:
            complete = True
        elif not stop_reason and not is_window_foreground(hwnd):
            stop_reason = "游戏失去前台，扫描安全停止"
        total = len(self.discovered_roster)
        if complete and len(seen) != total:
            complete = False
            stop_reason = "列表清点和实际尝试数量不一致"
        rate = len(details) / total if total and self.inventory_complete else None
        accepted = bool(complete and rate is not None and rate >= .90)
        if complete and not accepted:
            stop_reason = ("列表中存在未识别行，分母尚未完整确认" if rate is None
                           else f"本轮通过率 {rate:.1%}，未达到 90%")
        return details, {"complete": complete, "attempted": len(seen),
                         "failed": len(self.failed_characters), "succeeded": len(details),
                         "total": total, "inventory_complete": self.inventory_complete,
                         "success_rate": rate, "accepted": accepted,
                         "acceptance_threshold": .9,
                         "failed_characters": list(self.failed_characters),
                         "retries": self.retry_count, "stop_reason": stop_reason,
                         "max_rescuers": max_rescuers}

    @staticmethod
    def _find_label(rows, labels, *, prefer_bottom=True):
        wanted = set(labels)
        candidates = [row for row in _rows(rows)
                      if row["text"].replace(" ", "").replace("\n", "") in wanted
                      and row["confidence"] >= 0.62]
        if not candidates:
            return None
        return sorted(candidates, key=lambda row: (_center(row)[1], _center(row)[0]),
                      reverse=prefer_bottom)[0]

    def _click_row(self, hwnd, region, row, source, frame_size=None):
        self._check_active(hwnd)
        if frame_size is None:
            frame = self.capture.capture_game()
            frame_size = (frame.shape[1], frame.shape[0])
        width = max(1, int(frame_size[0]))
        height = max(1, int(frame_size[1]))
        x, y = _center(row)
        self.controller.set_window_handle(hwnd)
        self.controller.set_game_region(region)
        self.controller.click_at_percent(float(x) / width, float(y) / height,
                                         fast=False, source=source)

    def _click_fixed(self, hwnd, region, x, y, source):
        """Click a known control only after its page marker was OCR-confirmed."""
        self._check_active(hwnd)
        self.controller.set_window_handle(hwnd)
        self.controller.set_game_region(region)
        self.controller.click_at_percent(x, y, fast=False, source=source)

    def _click_transition(self, hwnd, region, x, y, source, origin, target,
                          *, expected_name=None):
        """Retry an ignored navigation click only while the origin is confirmed."""
        for attempt in range(1, 4):
            self._check_active(hwnd)
            self.sleep(.25)
            self._click_fixed(hwnd, region, x, y, source)
            try:
                return self._wait_page(hwnd, target, expected_name=expected_name, timeout=4.0)
            except QuickLookupInterrupted:
                raise
            except RuntimeError:
                frame, rows = self._read_frame(hwnd)
                page_kind = _page_kind(rows)
                if page_kind == target:
                    return self._wait_page(hwnd, target, expected_name=expected_name)
                if attempt == 3 or page_kind != origin:
                    raise
                if expected_name:
                    page = (parse_journey_initial(rows, expected_name)
                            if origin == "journey_initial" else parse_rescuer(rows))
                    self._verify_selected(page, expected_name)
                self.retry_count += 1
                self._report_progress(current=expected_name, error="导航点击未响应，核对原页面后重试")
                append_event(self.root, "rescuer", "navigation_retry",
                             task_id=self.task_id, stage=self.stage,
                             action=source, attempt=attempt, observed_page=origin)

    def _read_rescuer_detail(self, scan_id, region, hwnd, selected_frame, page, slot,
                             *, expected_name=None):
        """Traverse detail -> initial modal -> detail -> list with proof."""
        result = {"journey_initial": None, "journey_initial_portrait": None,
                  "uncertain": [], "can_continue": False}
        rows = page.get("ocr") or []
        detail_label = self._find_label(rows, {"查看详情"})
        if not detail_label:
            raise RuntimeError("查看详情按钮未确认")
        self._click_row(hwnd, region, detail_label, "quick_lookup_view_details",
                        (selected_frame.shape[1], selected_frame.shape[0]))
        frame, detail_rows = self._wait_page(hwnd, "rescuer_detail", expected_name=expected_name)
        detail_page = parse_rescuer(detail_rows)
        if expected_name:
            self._verify_selected(detail_page, expected_name)
        initial_label = self._find_label(detail_rows, {"旅程初始信息"})
        if not initial_label:
            raise RuntimeError("旅程初始信息按钮未确认")
        self._click_row(hwnd, region, initial_label, "quick_lookup_journey_initial",
                        (frame.shape[1], frame.shape[0]))
        initial_frame, initial_rows = self._wait_page(hwnd, "journey_initial", expected_name=expected_name)
        try:
            initial = parse_journey_initial(
                initial_rows, expected_name or (page.get("character") or {}).get("value"))
        except ValueError:
            raise RuntimeError("旅程初始信息角色未确认")
        self._verify_selected(initial, expected_name or self._page_identity(page))
        portrait_path = self._save_crop(scan_id, "journey_identity", initial_frame,
                                        _IDENTITY_BOXES["journey"], str(slot))
        initial["portrait_screenshot"] = portrait_path
        result["journey_initial"] = initial
        result["journey_initial_portrait"] = portrait_path
        result["uncertain"].extend(initial.get("uncertain") or [])
        self._record_step(scan_id, "initial_info_collected", expected_name)

        # The modal X is commonly missed by OCR; the verified modal kind is
        # the safety proof for this fixed hotspot.
        self._record_step(scan_id, "closing_initial", expected_name)
        detail_frame, detail_rows = self._click_transition(
            hwnd, region, .889, .203, "quick_lookup_close_initial",
            "journey_initial", "rescuer_detail", expected_name=expected_name)
        detail_page = parse_rescuer(detail_rows)
        if expected_name:
            self._verify_selected(detail_page, expected_name)

        # Full detail uses the top-left back arrow, not the modal close point.
        self._record_step(scan_id, "returning_to_list", expected_name)
        _, roster_rows = self._click_transition(
            hwnd, region, .043, .064, "quick_lookup_back_detail",
            "rescuer_detail", "rescuer_list", expected_name=expected_name)
        roster_page = parse_rescuer(roster_rows)
        if expected_name:
            self._verify_selected(roster_page, expected_name)
        self._record_step(scan_id, "returned_to_list", expected_name)
        result["can_continue"] = True
        return result
