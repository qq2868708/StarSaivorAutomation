"""截图工具: 截取游戏窗口保存为 PNG, 同时输出亮度统计."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from src.capture import ScreenCapture

cap = ScreenCapture()
screenshot = cap.capture_game()

if screenshot is None:
    print("截图失败: 未找到游戏窗口")
    sys.exit(1)

from datetime import datetime
import cv2
import numpy as np

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"screenshot_{ts}.png"
cv2.imwrite(filename, screenshot)
print(f"已保存: {filename}")

# 亮度统计
h, w = screenshot.shape[:2]

# 各训练行的图标区域 (5行, 右侧图标列)
# 图标大约在 x=0.60-0.80, 每行高度约 0.095
row_names = ["力量", "体力", "韧性", "专注", "保护"]
row_y_starts = [0.24, 0.34, 0.44, 0.54, 0.64]
row_h = 0.09

print("\n--- 各训练行图标区域亮度 ---")
for i, (name, y_start) in enumerate(zip(row_names, row_y_starts)):
    y1 = int(h * y_start)
    y2 = int(h * (y_start + row_h))
    x1 = int(w * 0.60)
    x2 = int(w * 0.80)
    roi = screenshot[y1:y2, x1:x2]
    if roi.size > 0:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        max_brightness = np.max(gray)
        # 明亮像素: >200 的像素占比
        bright_pixels = np.sum(gray > 200) / gray.size * 100
        print(f"  [{i+1}] {name}: 平均={mean_brightness:.0f} 最大={max_brightness:.0f} 亮像素(>200)={bright_pixels:.1f}%")
    else:
        print(f"  [{i+1}] {name}: 区域无效")

# 整体亮度
gray_full = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
print(f"\n整体: 平均亮度={np.mean(gray_full):.0f} 亮像素(>200)={np.sum(gray_full > 200) / gray_full.size * 100:.1f}%")
print(f"尺寸: {w}x{h}")
