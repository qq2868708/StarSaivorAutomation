"""Read-only observations for the daily automation workflow.

Daily pages and resource policies must be calibrated before adding inputs.
This module deliberately has no dependency on the foreground input controller.
"""
from datetime import datetime
import json
from pathlib import Path

import cv2

from .capture import ScreenCapture, is_window_foreground
from .light_ocr import LightOCR


def identify_daily_page(items):
    """Recognize only the home page verified in the running client."""
    texts = [str(text).replace(" ", "") for text, score, *_ in items
             if float(score) >= .8]
    navigation = [word for word in ("战斗", "管理", "总部", "公会", "商店", "观测")
                  if any(word == text for text in texts)]
    home_route = any("主流路线" in text for text in texts)
    page = "home" if home_route and len(navigation) >= 4 else "unknown"
    return {"page": page, "navigation": navigation,
            "home_route_visible": home_route, "input_ready": False}


def observe_daily(root, capture=None, ocr=None):
    """Save game-only pixels and OCR evidence without sending any input."""
    root = Path(root)
    destination = root / "runtime" / "daily"
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    report_path = destination / f"{stamp}.json"
    report = {"observed_at": datetime.now().astimezone().isoformat(),
              "mode": "observe_only", "inputs_sent": 0,
              "page": "unknown", "input_ready": False,
              "daily_completion": "unverified"}
    try:
        capture = capture or ScreenCapture("StarSavior")
        if not capture.find_window():
            raise RuntimeError("未找到 StarSavior 游戏窗口")
        hwnd = capture.get_hwnd()
        frame = capture.capture_game()
        screenshot = destination / f"{stamp}.png"
        if not cv2.imwrite(str(screenshot), frame):
            raise RuntimeError("保存日常识别截图失败")
        report.update(screenshot=str(screenshot.relative_to(root)).replace("\\", "/"),
                      foreground=is_window_foreground(hwnd),
                      capture_backend=capture.last_capture_backend,
                      size=[int(frame.shape[1]), int(frame.shape[0])])
        items = (ocr or LightOCR()).recognize(frame)
        report["ocr"] = [{"text": text, "confidence": float(score), "bbox": box}
                         for text, score, box in items]
        report.update(identify_daily_page(items))
        from .daily_arena import read_arena
        report["arena"] = read_arena(frame, items)
        if report["arena"]["page"] != "unknown":
            report["page"] = report["arena"]["page"]
        report["reason"] = ("已识别页面；当前入口仅观察，不执行日常"
                            if report["page"] != "unknown" else "页面尚未校准")
    except Exception as exc:
        report.update(error=f"{type(exc).__name__}: {exc}", reason="只读观察失败")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "latest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, report_path
