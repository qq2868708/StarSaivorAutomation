"""Send GPU info to Qwen for updated recommendations"""
import requests
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

prompt = """用户回复了之前的讨论，关键信息：

**GPU: AMD Radeon 7800 XT (RDNA 3, 16GB VRAM)**

这是重要的技术约束更新。请重新评估之前的方案：

1. ROCm 官方只支持 Linux，不支持 Windows 原生。7800 XT 在 Windows 下的 GPU 加速方案：
   - ONNX Runtime + DirectML (推荐，7800 XT 完全支持)
   - PaddlePaddle 在 Windows 下不支持 AMD GPU，只能用 CPU
   - 训练模型需要在 Linux 环境（WSL2 或双系统）

2. 请基于这个硬件约束，修正你之前的 PaddleOCR + 自定义模型方案。

3. 同时，用户之前提到的训练评分公式（第7点：除理事长/记者外全部移植），请确认你的实现方向。

请用中文简短回答（300字以内），聚焦于：Windows + 7800 XT 下实际可行的方案是什么？"""

resp = requests.post(
    "http://localhost:1234/v1/chat/completions",
    json={
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 2048,
        "stream": False,
    },
    timeout=120,
)
content = resp.json()["choices"][0]["message"]["content"]
print(content)

out = PROJECT / "ai_suggestions" / "qwen_gpu_response.md"
out.write_text("# Qwen 对 GPU (7800 XT) 的回应\n\n" + content, encoding="utf-8")
print(f"\n[保存] {out}")
