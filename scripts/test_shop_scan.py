"""
测试商店交易窗口物品扫描
依次点击物品2 → 3 → 1，OCR 名称/效果/告罄 三个区域
用法: 先打开游戏并进入交易界面，然后运行此脚本
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.controller import Controller
from src.capture import ScreenCapture
from src.recognition import HybridOCR


def _ocr_region(engine, screenshot, x, y, w, h):
    """读取区域内 OCR 文本"""
    return engine.recognize_region(screenshot, x, y, w, h)

# 坐标 (来自 ShopHandler, mouse_coords.py 校准)
ITEM_CLICK_POINTS = [
    (0.8490, 0.4102),   # 物品1
    (0.8521, 0.5287),   # 物品2
    (0.8547, 0.6333),   # 物品3
]

ITEM_NAME_REGION = (0.5151, 0.2444, 0.1958, 0.0973)
ITEM_EFFECT_REGION = (0.5109, 0.6278, 0.2026, 0.1361)
ITEM_SOLDOUT_REGION = (0.5542, 0.3556, 0.1192, 0.1296)

SOLD_OUT_KW = ["SOLD", "SOLDOUT", "售罄", "売切"]


def main():
    print("初始化...")
    ocr = HybridOCR()
    capture = ScreenCapture()
    controller = Controller(click_delay=0.3, action_interval=0.5)

    print("请确保游戏交易窗口已打开 (显示3个商品 + 详情面板)")
    input("按 Enter 开始扫描...")

    # 按顺序点击: 2, 3, 1
    scan_order = [2, 3, 1]

    for slot_num in scan_order:
        idx = slot_num - 1
        cx, cy = ITEM_CLICK_POINTS[idx]

        print(f"\n{'='*60}")
        print(f"  点击 物品{slot_num} | 坐标 ({cx:.4f}, {cy:.4f})")
        print(f"{'='*60}")

        controller.click_at_percent(cx, cy)
        time.sleep(0.35)

        screenshot = capture.capture_game()
        if screenshot is None:
            print("  [错误] 截图失败!")
            continue

        # 分别 OCR 三个区域
        soldout = _ocr_region(ocr, screenshot, *ITEM_SOLDOUT_REGION)
        name = _ocr_region(ocr, screenshot, *ITEM_NAME_REGION)
        effect = _ocr_region(ocr, screenshot, *ITEM_EFFECT_REGION)
        is_soldout = bool(soldout and any(kw in soldout.upper() for kw in SOLD_OUT_KW))

        print(f"  [告罄] {'!!! 已售罄 !!!' if is_soldout else '可购买'}")
        print(f"         原始OCR: '{soldout.strip()}'")
        print(f"  [名称] '{name.strip()}'")
        print(f"  [效果] '{effect.strip()}'")

    print(f"\n{'='*60}")
    print("  扫描完成 — 请检查上面输出是否正确")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
