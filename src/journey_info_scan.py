"""Read the journey information panel without advancing the journey.

The scanner is deliberately separate from the running automation.  It is
invoked by an explicit WebUI action after a run has ended and may open the
single magnifying-glass panel when the terminal settlement page is visible.
It never selects rewards, skills, or any other post-run action.
"""
from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from typing import Any

import cv2

from .capture import ScreenCapture, is_window_foreground
from .controller import Controller
from .recognition import HybridOCR


_CJK = re.compile(r"^[\u3400-\u9fff·]{2,16}$")
_SLASH_NUMBER = re.compile(r"^(\d{1,5})\s*/\s*\d{1,5}$")
_NUMBER = re.compile(r"^\d{1,5}$")
_RANK = re.compile(r"^RANK\s*[:：]?\s*(\d{1,5})$", re.I)
_LEVEL = re.compile(r"^Lv\.?\s*(\d{1,3})$", re.I)


class JourneyInfoScanError(RuntimeError):
    """The current screen cannot be safely read as journey information."""


def _bounds(box):
    try:
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        if len(xs) != 4 or len(ys) != 4:
            return None
        return min(xs), min(ys), max(xs), max(ys)
    except (TypeError, ValueError):
        return None


def _row(text, confidence, box):
    bounds = _bounds(box)
    return {"text": str(text).strip(), "confidence": float(confidence),
            "box": box, "bounds": bounds}


def _rows(items):
    result = []
    for item in items or []:
        try:
            text, confidence, box = item[0], item[1], item[2]
            item = _row(text, confidence, box)
        except (IndexError, TypeError, ValueError):
            continue
        if item["text"] and item["bounds"]:
            result.append(item)
    return result


def _center(row):
    left, top, right, bottom = row["bounds"]
    return (left + right) / 2, (top + bottom) / 2


def _is_dialog(rows) -> bool:
    texts = " ".join(row["text"] for row in rows)
    return "旅程信息" in texts and "继承星辰档案" in texts


def _is_terminal_page(rows) -> bool:
    texts = " ".join(row["text"] for row in rows)
    if "旅程结束" in texts:
        return True
    has_rank = any(_RANK.fullmatch(row["text"]) for row in rows)
    slash_values = sum(bool(_SLASH_NUMBER.fullmatch(row["text"])) for row in rows)
    return has_rank and slash_values >= 3


def _nearest_value(rows, label, pattern, *, x_limit=None):
    """Return one unambiguous number next to a labelled value."""
    if label["confidence"] < 0.80:
        return None, None
    lx, ly = _center(label)
    candidates = []
    for index, candidate in enumerate(rows):
        if candidate is label or candidate["confidence"] < 0.80:
            continue
        if not pattern.fullmatch(candidate["text"]):
            continue
        left, top, right, bottom = candidate["bounds"]
        cx, cy = _center(candidate)
        if x_limit is not None and not (0 <= left <= x_limit):
            continue
        label_left, label_top, label_right, label_bottom = label["bounds"]
        # The panel uses both label/value columns.  Accept a value below the
        # label or immediately to its right, then keep the closest candidate.
        # The two stat columns are close together, so compare the left edge
        # rather than only the centre; otherwise the adjacent column becomes
        # a false second candidate for the same label.
        below = label_bottom - 6 <= top <= label_bottom + 42 and abs(left - label_left) <= 28
        if below:
            distance = abs(cy - ly) + abs(cx - lx) * 0.05
            candidates.append((distance, index, candidate))
    if len(candidates) != 1:
        # A value can be found twice by OCR (one row split into overlapping
        # boxes).  Only accept a clear nearest candidate.
        if not candidates:
            return None, None
        candidates.sort(key=lambda item: item[0])
        if len(candidates) > 1 and candidates[0][0] + 10 >= candidates[1][0]:
            return None, None
    _, index, candidate = min(candidates, key=lambda item: item[0])
    match = pattern.fullmatch(candidate["text"])
    return int(match.group(1) if match.lastindex else match.group(0)), index


def parse_journey_info(items) -> dict[str, Any]:
    """Parse only labelled, unambiguous values from a journey-info OCR pass."""
    rows = _rows(items)
    if not _is_dialog(rows):
        raise JourneyInfoScanError("当前画面不是“旅程信息”放大镜界面")

    values: dict[str, Any] = {"actual_setup": {}, "outcome": {}, "evidence": {}}
    evidence = values["evidence"]
    # Character name is the centred two-to-six-character label beneath SSR.
    names = []
    for index, item in enumerate(rows):
        left, top, right, bottom = item["bounds"]
        if 150 <= left <= 330 and 290 <= top <= 355 and item["confidence"] >= 0.80 and _CJK.fullmatch(item["text"]):
            names.append((index, item))
    if len(names) == 1:
        index, item = names[0]
        values["actual_setup"]["character"] = item["text"]
        evidence["character"] = [index]

    rank_rows = [(index, item) for index, item in enumerate(rows) if item["confidence"] >= 0.80 and _RANK.fullmatch(item["text"])]
    if len(rank_rows) == 1:
        index, item = rank_rows[0]
        values["outcome"]["rank"] = int(_RANK.fullmatch(item["text"]).group(1))
        evidence["rank"] = [index]

    stat_labels = ("力量", "体力", "韧性", "专注", "保护")
    for stat in stat_labels:
        labels = [
            (index, item) for index, item in enumerate(rows)
            if item["confidence"] >= 0.80 and item["text"] == stat
        ]
        if len(labels) != 1:
            continue
        label_index, label = labels[0]
        value, value_index = _nearest_value(rows, label, _SLASH_NUMBER, x_limit=430)
        if value is not None:
            values["outcome"].setdefault("final_stats", {})[stat] = value
            evidence["final_stats." + stat] = [label_index, value_index]

    potential_labels = [
        (index, item) for index, item in enumerate(rows)
        if item["confidence"] >= 0.80 and item["text"] == "潜质点数"
    ]
    if len(potential_labels) == 1:
        label_index, label = potential_labels[0]
        value, value_index = _nearest_value(rows, label, _NUMBER, x_limit=430)
        if value is not None:
            values["outcome"]["potential_points"] = value
            evidence["potential_points"] = [label_index, value_index]

    archive_indices = []
    for index, item in enumerate(rows):
        left, top, right, bottom = item["bounds"]
        if (450 <= left <= 1100 and 565 <= top <= 625 and item["confidence"] >= 0.80
                and _CJK.fullmatch(item["text"])
                and item["text"] not in {"旅程信息", "详细能力值", "潜质", "救援者介绍"}):
            archive_indices.append((left, index, item["text"]))
    archive_indices.sort()
    if archive_indices:
        values["actual_setup"]["journey_records"] = [
            {"name": text, "slot": slot + 1}
            for slot, (_, _, text) in enumerate(archive_indices[:20])
        ]
        evidence["journey_records"] = [index for _, index, _ in archive_indices[:20]]

    # Card artwork contains no reliable card name text.  Keep card levels and
    # rarity as scan metadata, but leave support_cards unknown instead of
    # fabricating names from an illustration.
    cards = []
    for item in rows:
        left, top, right, bottom = item["bounds"]
        match = _LEVEL.fullmatch(item["text"])
        if match and item["confidence"] >= 0.80 and 480 <= left <= 1250 and 370 <= top <= 450:
            cards.append({"level": int(match.group(1)), "box": item["box"]})
    values["support_card_slots"] = cards[:5]
    values["ocr"] = [{key: item[key] for key in ("text", "confidence", "box")} for item in rows]
    return values


class JourneyInfoScanner:
    def __init__(self, root, window_title="StarSavior", capture=None, ocr=None,
                 controller=None, sleep=time.sleep):
        self.root = Path(root)
        self.capture = capture or ScreenCapture(window_title)
        self.ocr = ocr or HybridOCR()
        self.controller = controller or Controller(auto_refocus=False)
        self.sleep = sleep

    def _read(self, frame):
        return self.ocr.recognize_detailed(frame)

    def _save(self, scan_id, label, frame):
        folder = self.root / "logs" / "screenshots"
        folder.mkdir(parents=True, exist_ok=True)
        name = f"journey_info_{scan_id}_{label}.png"
        path = folder / name
        if not cv2.imwrite(str(path), frame):
            raise JourneyInfoScanError("扫描截图保存失败")
        return (Path("logs") / "screenshots" / name).as_posix()

    def scan(self, open_magnifier=True):
        scan_id = uuid.uuid4().hex
        region = self.capture.find_window()
        hwnd = self.capture.get_hwnd()
        if not region or not hwnd:
            raise JourneyInfoScanError("找不到 StarSavior 游戏窗口")
        self.controller.set_window_handle(hwnd)
        self.controller.set_game_region(region)

        frame = self.capture.capture_game()
        items = self._read(frame)
        opened = False
        before_path = self._save(scan_id, "before", frame)
        if not _is_dialog(_rows(items)):
            if not open_magnifier or not _is_terminal_page(_rows(items)):
                raise JourneyInfoScanError("当前画面既不是放大镜页，也不是可安全打开的旅程结束页")
            if not is_window_foreground(hwnd):
                raise JourneyInfoScanError("当前仍是结算页；请先将 StarSavior 切到前台，扫描器才能安全点击放大镜")
            # The button is the magnifier in the fixed left stats panel.  The
            # terminal-page OCR check above prevents blind input on other UI.
            self.controller.click_at_percent(0.151, 0.315, fast=True, source="journey_info_scan")
            opened = True
            self.sleep(0.8)
            frame = self.capture.capture_game()
            items = self._read(frame)
        if not _is_dialog(_rows(items)):
            after_path = self._save(scan_id, "after", frame)
            raise JourneyInfoScanError(f"未能打开旅程信息面板（截图：{after_path}）")
        dialog_path = self._save(scan_id, "dialog", frame)
        parsed = parse_journey_info(items)
        parsed.update({"scan_id": scan_id, "opened_magnifier": opened,
                       "screenshot": dialog_path, "before_screenshot": before_path})
        return parsed

