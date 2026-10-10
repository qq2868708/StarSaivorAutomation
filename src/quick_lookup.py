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
from .controller import Controller
from .recognition import HybridOCR
from .runtime_control import RuntimeControl


_CJK = re.compile(r"^[\u3400-\u9fff·]{2,24}$")
_CJK_SHORT = re.compile(r"^[\u3400-\u9fff·]{2,10}$")
_NAME = re.compile(r"^[A-Za-z0-9\u3400-\u9fff·]{2,14}$")
_LEVEL = re.compile(r"^Lv\.?\s*(\d{1,3})(?:\s*/\s*(\d{1,3}))?", re.I)
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
                  "new_verified", "replaced", "total"}

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
                           "center": [cx, cy], **nearest_level})
    result["roster"] = sorted(roster, key=lambda item: item["center"][1])
    result["ocr"] = [{key: row[key] for key in ("text", "confidence", "box")} for row in rows]
    return result


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
    if not any(term in texts for term in ("阿尔克那", "支援卡", "旅程初始效果", "训练效果")):
        raise ValueError("当前画面不是阿尔克那/支援卡界面")

    level = _extract_level(rows)
    rarity = _extract_rarity(rows)
    effects = _extract_effects(rows)
    header_names = _extract_names(rows, x_min=735, x_max=1250, y_min=240, y_max=322, short=False)
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
            any(row["text"] == "旅程初始效果" and row["confidence"] >= .72 and
                735 <= row["bounds"][0] <= 1250 and 370 <= row["bounds"][1] <= 415
                for row in rows))


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
                "attempted": len(details),
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
                 should_stop=None, on_rescuer_entry=None):
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
        self.completed_count = 0
        self.retry_count = 0
        self.last_observation = None
        self.stage = "waiting_foreground"
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

    def _wait_page(self, hwnd, kind, *, expected_name=None, timeout=6.0, interval=.15):
        """Require page and identity together on two consecutive captures."""
        self.stage = kind
        deadline = self.clock() + max(.1, float(timeout))
        last_error = None
        accepted = 0
        while self.clock() < deadline:
            self._check_active(hwnd)
            frame, rows = self._read_frame(hwnd)
            try:
                if _page_kind(rows) == kind:
                    if expected_name:
                        page = (parse_journey_initial(rows, expected_name)
                                if kind == "journey_initial" else parse_rescuer(rows))
                        self._verify_selected(page, expected_name)
                    accepted += 1
                    if accepted >= 2:
                        self.transitions.append({"page": kind, "character": expected_name,
                                                 "confirmed": True})
                        return frame, rows
                else:
                    accepted = 0
            except (ValueError, RuntimeError) as error:
                last_error = error
                accepted = 0
            self.sleep(interval)
        raise RuntimeError(f"等待页面转场失败：{kind}" + (f"（{last_error}）" if last_error else ""))

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
                self._click_fixed(hwnd, region, .889, .203,
                                  "quick_lookup_recover_close_initial")
                self._wait_page(hwnd, "rescuer_detail")
                continue
            if kind == "rescuer_detail":
                self._click_fixed(hwnd, region, .043, .064,
                                  "quick_lookup_recover_back_detail")
                frame, list_rows = self._wait_page(hwnd, "rescuer_list")
                return frame, parse_rescuer(list_rows)
            raise RuntimeError("无法安全恢复角色列表，当前页面未确认")
        raise RuntimeError("恢复角色列表超时")

    def _report_progress(self, *, current=None, read_attempts=0, error=None):
        """Publish best-effort progress without coupling the scanner to WebUI."""
        if not self.progress:
            return
        try:
            self.progress({"attempted": self.completed_count, "failed": 0,
                           "retries": self.retry_count, "current": current,
                           "read_attempts": read_attempts, "stage": self.stage,
                           "last_retry_error": error})
        except Exception:
            # A UI update must not affect game input or verification.
            pass

    def _record_retry(self, scan_id, name, attempt, error):
        """Keep retry reasons in ignored runtime logs, bounded in each record."""
        try:
            path = self.root / "runtime" / "quick_lookup_events.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"scan_id": scan_id, "event": "read_retry",
                    "character": name, "attempt": attempt, "stage": self.stage,
                    "error": str(error),
                    "ts": datetime.now().astimezone().isoformat()}, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _retry_rescuer_entry(self, scan_id, region, hwnd, name, slot,
                             *, select_roster=True):
        """Read one roster entry until the complete evidence path succeeds.

        A row is never converted into a skipped/failed result merely because
        one click or OCR frame was wrong.  Every retry starts with a fresh list
        frame, confirms the target row again, and only then sends input.  The
        loop leaves immediately when the game loses the foreground; in every
        other case it waits, safely recovers a known page, and retries the same
        character.
        """
        attempts = 0
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
                except QuickLookupInterrupted:
                    raise
                except Exception as recovery_error:
                    self._check_active(hwnd)
                    retry_errors.append(
                        f"恢复页面: {type(recovery_error).__name__}: {recovery_error}")
                    retry_errors = retry_errors[-20:]
                self.sleep(min(1.5, 0.25 * attempts))

    def scan(self, kind, traverse_rescuer=True, max_rescuers=100):
        if kind not in {"rescuer", "arcanum"}:
            raise ValueError("扫描类型无效")
        region, hwnd = self.wait_for_foreground()
        region, hwnd, frame = self._capture()
        if frame.shape[:2] != (900, 1600):
            frame = cv2.resize(frame, (1600, 900), interpolation=cv2.INTER_AREA)
        items = self.ocr.recognize_detailed(frame)
        if kind == "rescuer" and _page_kind(items) != "rescuer_list":
            raise ValueError("当前不是救援者列表页，请打开救援者界面后重试")
        parser = parse_rescuer if kind == "rescuer" else parse_arcanum
        parsed = parser(items)
        scan_id = uuid.uuid4().hex
        parsed.update({"scan_id": scan_id, "observed_at": datetime.now().astimezone().isoformat()})
        if kind == "arcanum":
            if not _arcanum_detail_confirmed(items) or not _name_value(parsed["cards"][0].get("name")):
                raise ValueError("阿尔克那详情和卡名未确认，请打开旅程效果详情后重新读取")
            identity = self._save_crop(scan_id, "arcanum_identity", frame,
                                       _IDENTITY_BOXES["arcanum"])
            for card in parsed.get("cards") or []:
                card["image"] = identity
        if kind == "rescuer" and traverse_rescuer:
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

    def scan_selected_rescuer(self):
        """Force-read the currently selected rescuer and return one detail row."""
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
        details = []
        seen = set()
        stagnant_scrolls = 0
        next_slot = 1
        complete = False
        stop_reason = None
        list_read_retries = 0
        self.discovered_roster = []

        def remember_roster(rows):
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
                    positions[name] = len(self.discovered_roster)
                    self.discovered_roster.append(copy.deepcopy(item))

        while len(seen) < max_rescuers and is_window_foreground(hwnd):
            try:
                current_frame, current_rows = self._wait_page(hwnd, "rescuer_list")
                current_page = parse_rescuer(current_rows)
                entries = list(current_page.get("roster") or [])
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
                if not name or name in seen:
                    continue
                try:
                    selected, page, detail, screenshot, attempts, retry_errors = (
                        self._retry_rescuer_entry(
                            scan_id, region, hwnd, name, next_slot))
                except QuickLookupInterrupted as error:
                    stop_reason = str(error)
                    break
                seen.add(name)
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
                self.completed_count = len(seen)
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
        return details, {"complete": complete, "attempted": len(seen), "failed": 0,
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

        # The modal X is commonly missed by OCR; the verified modal kind is
        # the safety proof for this fixed hotspot.
        self._click_fixed(hwnd, region, 0.889, 0.203, "quick_lookup_close_initial")
        detail_frame, detail_rows = self._wait_page(hwnd, "rescuer_detail", expected_name=expected_name)
        detail_page = parse_rescuer(detail_rows)
        if expected_name:
            self._verify_selected(detail_page, expected_name)

        # Full detail uses the top-left back arrow, not the modal close point.
        self._click_fixed(hwnd, region, 0.043, 0.064, "quick_lookup_back_detail")
        _, roster_rows = self._wait_page(hwnd, "rescuer_list", expected_name=expected_name)
        roster_page = parse_rescuer(roster_rows)
        if expected_name:
            self._verify_selected(roster_page, expected_name)
        result["can_continue"] = True
        return result
