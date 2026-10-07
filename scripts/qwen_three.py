"""发送三个任务给 Qwen 讨论"""
import requests
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

prompt = """用户提出三个新任务，请与 Claude Code 协作讨论实现方案：

## 任务1：属性目标配置 + 目标接近降权

用户要求在训练开始前输入每个属性的期望值（力量/体力/韧性/专注/保护），然后作为训练目标。在评分函数中，当某属性当前值接近目标时降低该训练权重。

Claude Code 建议方案：
- config.yaml 新增 training.target_stats 配置（5个属性的目标值）
- trainer.py 启动时读取并提示用户确认/修改
- rule_engine.py 的 _score_training_rows() 加入目标降权逻辑：gap = target - current; if gap <= 10: score *= 0.5; elif gap <= 30: score *= 0.8
- 当前属性值从 _ocr_all_stats() 获取

## 任务2：事件处理系统完善

基于之前 plan 中的设计，实现：
- profiles/events/{profile}.json 中事件 keyword/alias 子串匹配（已部分完成）
- _match_known_event(): OCR选项文字 → 匹配已知事件 → 返回推荐选项
- _auto_learn_event(): 未匹配的事件自动添加到 JSON（status: "pending"）
- 角色覆盖：同一事件不同角色可选不同选项

## 任务3：自定义 PaddleDetection 模型训练

用户想训练自定义文本检测模型。当前环境：Windows + AMD 7800 XT (16GB)。

约束：
- 训练必须在 Linux 环境（ROCm 官方不支持 Windows）
- 推理可用 ONNX Runtime + DirectML（已实现）
- 数据集需要：训练中可能遇到的事件、奖励、训练属性等截图 + 文本区域标注

需要你和 Claude Code 协作完成：
- 数据集采集策略（自动截图 + 标注工具）
- 训练环境搭建（WSL2 或云 GPU）
- 训练流程和参数
- 模型导出 ONNX 集成到现有 DirectMLOCR

请对三个任务给出具体可实施方案，用中文回答，300字以内每个任务。"""

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

out = PROJECT / "ai_suggestions" / "qwen_three_tasks.md"
out.write_text("# Qwen 对三个新任务的讨论\n\n" + content, encoding="utf-8")
print(f"\n[保存] {out}")
