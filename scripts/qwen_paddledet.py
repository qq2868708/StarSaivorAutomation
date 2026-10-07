"""与 Qwen 讨论 PaddleDetection 自定义模型训练方案"""
import requests
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

prompt = """## 背景
我们在做一个 Starsavior (赛马娘类游戏) 训练自动化机器人。目前的 OCR 使用 RapidOCR + PP-OCRv4 ONNX + DirectML (AMD 7800 XT 16GB) 进行文本识别。

当前痛点：
- PP-OCRv4 的检测模型对游戏内的中文文本（繁体/简体混合、特殊字体、小字号）检测不够准确
- 需要用游戏实际截图训练自定义文本检测模型来替换 PP-OCRv4 的检测部分

## 约束条件
- GPU: AMD 7800 XT (16GB)
- 操作系统: Windows 11
- ROCm 官方不支持 Windows，训练必须在 Linux 环境
- 推理可以用 ONNX Runtime + DirectML (已实现)
- 当前已安装: rapidocr-onnxruntime 1.4.4, onnxruntime-directml

## 需要你和 Claude Code 协作完成

### 1. 训练环境搭建
- WSL2 + Ubuntu 22.04 或单独的双系统？
- ROCm 驱动在 WSL2 中是否可行？
- 如果 WSL2+ROCm 不可行，有哪些替代方案（云 GPU、Docker、等）？
- 安装 PaddleDetection 的精确步骤

### 2. 数据集采集策略
- 机器人运行时会自动截图，可以采集训练中遇到的事件、奖励、训练属性等界面
- 需要标注文本区域的 bounding box
- 推荐什么标注工具？LabelImg? Label Studio? PPOCRLabel?
- 数据集格式应该用 PaddleOCR 的检测格式还是 ICDAR2015？

### 3. 训练流程和参数
- 基于 PP-OCRv4 的检测模型 fine-tune 还是从头训练？
- batch_size、lr、epoch 建议值？
- 训练数据需要多少张？

### 4. 模型导出和集成
- 训练完成后导出 ONNX
- 用 onnxruntime-directml 在 Windows 推理
- 替换现有 DirectMLOCR 的检测模型部分

请给出具体可实施的方案，用中文回答，300字以内每个小点。"""

resp = requests.post(
    "http://localhost:1234/v1/chat/completions",
    json={
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 4096,
        "stream": False,
    },
    timeout=180,
)
content = resp.json()["choices"][0]["message"]["content"]
print(content)

out = PROJECT / "ai_suggestions" / "qwen_paddledet.md"
out.write_text("# Qwen 对 PaddleDetection 训练方案的讨论\n\n" + content, encoding="utf-8")
print(f"\n[保存] {out}")
