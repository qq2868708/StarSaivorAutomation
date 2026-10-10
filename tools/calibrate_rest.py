"""Calibrate rest-screen button positions from a foreground game screenshot.

Usage:
  python tools/calibrate_rest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "verification" / "calibrate_rest.png"


def capture_screen():
    print("截图中...")
    try:
        import pyautogui

        image = pyautogui.screenshot()
        print("pyautogui 截图成功")
    except Exception as error:
        print(f"pyautogui 失败: {error}")
        try:
            from PIL import ImageGrab

            image = ImageGrab.grab()
            print("PIL 截图成功")
        except Exception as fallback_error:
            print(f"PIL 失败: {fallback_error}")
            raise SystemExit(1)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def main() -> int:
    image = capture_screen()
    height, width = image.shape[:2]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT), image)
    print(f"截图已保存: verification/{OUTPUT.name} ({width}x{height})")

    print("\n初始化 OCR 引擎 (首次需下载模型)...")
    from src.light_ocr import LightOCR

    results = LightOCR().recognize(image)
    if not results:
        print("[错误] OCR 未识别到任何文字")
        return 1

    print(f"\n全屏 OCR 结果 ({len(results)} 条):")
    print(f"{'文本':<24} {'置信度':<8} {'X%':<8} {'Y%':<8}")
    print("-" * 56)

    found = []
    for text, confidence, bbox in results:
        if not text:
            continue

        points = np.array(bbox)
        center_x = float(np.mean(points[:, 0])) / width
        center_y = float(np.mean(points[:, 1])) / height
        clean = text.replace(" ", "").replace("\r", "").replace("\n", "").strip()

        if clean:
            marker = ""
            if any(keyword in clean for keyword in ["露宿", "住处", "冥想", "免费", "休息"]):
                marker = " <==="
            print(f"{clean:<24} {confidence:<8.2f} {center_x:<8.3f} {center_y:<8.3f}{marker}")

            if confidence > 0.3:
                for keyword in ["露宿", "住处", "冥想", "免费休息", "冥想室"]:
                    if keyword in clean:
                        found.append((clean, center_x, center_y, confidence))

    print(f"\n{'=' * 56}")
    print(f"找到 {len(found)} 个休息相关文字:")

    found.sort(key=lambda item: (item[2], -item[3]))
    for text, center_x, center_y, confidence in found:
        print(f"  '{text}' -> ({center_x:.3f}, {center_y:.3f}) conf={confidence:.2f}")

    if found:
        print("\n建议 REST_CLICK_POINTS (从上到下):")
        for index, (text, center_x, center_y, _confidence) in enumerate(found[:3]):
            print(f"  ({center_x:.2f}, {center_y:.2f}),   # [{index}] {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
