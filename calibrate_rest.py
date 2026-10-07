"""
休息界面按钮位置校准工具

用法:
  1. 打开游戏, 进入休息界面 (确保能看到露宿/住处/冥想室三个选项)
  2. 确认游戏窗口在前台
  3. 运行: python calibrate_rest.py
  4. 脚本会截图 + OCR, 打印检测到的按钮位置
"""
import sys
import cv2
import numpy as np

print("截图中...")
try:
    import pyautogui
    img_pil = pyautogui.screenshot()
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    print("pyautogui 截图成功")
except Exception as e:
    print(f"pyautogui 失败: {e}")
    try:
        from PIL import ImageGrab
        img_pil = ImageGrab.grab()
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        print("PIL 截图成功")
    except Exception as e2:
        print(f"PIL 失败: {e2}")
        sys.exit(1)

h, w = img.shape[:2]
cv2.imwrite("calibrate_rest.png", img)
print(f"截图已保存: calibrate_rest.png ({w}x{h})")

print("\n初始化 OCR 引擎 (首次需下载模型)...")
from paddleocr import PaddleOCR
ocr = PaddleOCR(lang="ch", show_log=False)
results = ocr.ocr(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cls=False)

if results is None or results[0] is None:
    print("[错误] OCR 未识别到任何文字")
    sys.exit(1)

print(f"\n全屏 OCR 结果 ({len(results[0])} 条):")
print(f"{'文本':<24} {'置信度':<8} {'X%':<8} {'Y%':<8}")
print("-" * 56)

found = []
for line in results[0]:
    bbox = line[0]
    text = line[1][0]
    conf = line[1][1]
    if not text:
        continue

    pts = np.array(bbox)
    cx_pct = float(np.mean(pts[:, 0])) / w
    cy_pct = float(np.mean(pts[:, 1])) / h
    clean = text.replace(" ", "").replace("\r", "").replace("\n", "").strip()

    if clean:
        marker = ""
        if any(kw in clean for kw in ["露宿", "住处", "冥想", "免费", "休息"]):
            marker = " <==="
        print(f"{clean:<24} {conf:<8.2f} {cx_pct:<8.3f} {cy_pct:<8.3f}{marker}")

        if conf > 0.3:
            for kw in ["露宿", "住处", "冥想", "免费休息", "冥想室"]:
                if kw in clean:
                    found.append((clean, cx_pct, cy_pct, conf))

print(f"\n{'='*56}")
print(f"找到 {len(found)} 个休息相关文字:")

# 按 Y 坐标排序
found.sort(key=lambda x: (x[2], -x[3]))

for text, cx, cy, conf in found:
    print(f"  '{text}' → ({cx:.3f}, {cy:.3f}) conf={conf:.2f}")

if found:
    print(f"\n建议 REST_CLICK_POINTS (从上到下):")
    for i, (text, cx, cy, conf) in enumerate(found[:3]):
        print(f"  ({cx:.2f}, {cy:.2f}),   # [{i}] {text}")
