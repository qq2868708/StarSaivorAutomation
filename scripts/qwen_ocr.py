"""Send OCR upgrade discussion to Qwen"""
import requests
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

prompt = """用户要求将 Starsavior 的图像识别升级为 ONNX Runtime + DirectML（AMD 7800 XT GPU 加速）。

当前架构：
- WinOCR (Windows 内置, 即时, ~0.8s) — handler can_handle 热路径
- PaddleOCR (CPU, 懒加载, 2-3分钟首次加载) — 详细识别/校准

目标：
- 用 ONNX Runtime + DirectML 替换 PaddleOCR CPU 推理
- 保持 WinOCR 作为快速回退
- 寻找对中文文字识别最准确的 DirectML 兼容模型

候选模型方案：
A. PaddleOCR PP-OCRv4/v5 → ONNX 导出 (官方支持, 中文最佳)
B. TrOCR (Microsoft, Transformer-based, 英文为主)
C. Surya (多语言, Transformer-based)
D. EasyOCR → ONNX (PyTorch 模型导出)

Claude Code 的建议：方案A — PP-OCRv4 ONNX 导出
- PaddleOCR 有官方 ONNX 导出工具 (paddle2onnx)
- PP-OCRv4 中文识别准确率业界最高
- ONNX Runtime 1.18+ 支持 DirectML provider, AMD 7800 XT 完全兼容
- 检测模型 (DB) + 识别模型 (CRNN) 分别导出为两个 ONNX 模型
- 推理速度预计 < 0.3s (vs 当前 PaddleOCR CPU ~1-2s)

请简短评价这个方案并给出补充建议（200字以内）。"""

resp = requests.post(
    "http://localhost:1234/v1/chat/completions",
    json={
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1024,
        "stream": False,
    },
    timeout=120,
)
content = resp.json()["choices"][0]["message"]["content"]
print(content)

out = PROJECT / "ai_suggestions" / "qwen_ocr_onnx.md"
out.write_text("# Qwen 对 ONNX Runtime + DirectML OCR 方案的评价\n\n" + content, encoding="utf-8")
print(f"\n[保存] {out}")
