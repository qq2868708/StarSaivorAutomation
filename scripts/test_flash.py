"""闪光训练检测: 使用更新后的双重检测逻辑 (浅黄HSV + 饱和度相对差异)."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import cv2
import numpy as np
from src.capture import ScreenCapture

cap = ScreenCapture()
screenshot = cap.capture_game()

if screenshot is None:
    print("截图失败: 未找到游戏窗口")
    sys.exit(1)

h, w = screenshot.shape[:2]
print(f"截图: {w}x{h}")

ROWS = [("力量", 0.86, 0.28), ("体力", 0.86, 0.38), ("韧性", 0.86, 0.48),
        ("集中", 0.86, 0.58), ("保护", 0.86, 0.68)]

rois_hsv = []
s_means = []
v_means = []

for name, rx, ry in ROWS:
    bx = max(0, int(w * (rx - 0.04)))
    by = max(0, int(h * (ry - 0.02)))
    bw = max(1, int(w * 0.08))
    bh = max(1, int(h * 0.04))
    roi = screenshot[by:by + bh, bx:bx + bw]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    rois_hsv.append(hsv)
    s_means.append(float(np.mean(hsv[:, :, 1])))
    v_means.append(float(np.mean(hsv[:, :, 2])))

# 方法1: 浅黄像素
print("\n=== 方法1: 浅黄像素 [H:15-40, S>=50, V>=150] ===")
lower = np.array([15, 50, 150])
upper = np.array([40, 255, 255])
result1 = [False] * 5
for i, (name, _, _) in enumerate(ROWS):
    mask = cv2.inRange(rois_hsv[i], lower, upper)
    ratio = np.count_nonzero(mask) / mask.size
    result1[i] = ratio > 0.05
    marker = "[FLASH]" if result1[i] else "       "
    print(f"  [{i+1}] {name}: ratio={ratio:.3f} {marker}")

# 方法2: 饱和度相对差异
print("\n=== 方法2: 饱和度相对差异 ===")
mean_s = sum(s_means) / len(s_means)
max_s = max(s_means)
print(f"  S均值={mean_s:.0f} 最大S={max_s:.0f} 差值={max_s-mean_s:.0f}")
result2 = [False] * 5
if max_s > mean_s + 10 and max_s > 60:
    for i in range(5):
        if s_means[i] >= max_s:
            result2[i] = True
for i, (name, _, _) in enumerate(ROWS):
    marker = "[FLASH]" if result2[i] else "       "
    print(f"  [{i+1}] {name}: S={s_means[i]:.0f} V={v_means[i]:.0f} {marker}")

# 汇总
print("\n=== 结果 ===")
final = [r1 or r2 for r1, r2 in zip(result1, result2)]
flash_rows = [ROWS[i][0] for i, f in enumerate(final) if f]
source = []
for i, f in enumerate(final):
    if f:
        parts = []
        if result1[i]: parts.append("浅黄")
        if result2[i]: parts.append("饱和度")
        source.append(f"{ROWS[i][0]}({'+'.join(parts)})")
print(f"  闪光训练: {', '.join(source) if source else '无'}")
