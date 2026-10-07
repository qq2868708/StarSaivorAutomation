"""发送讨论内容给 Qwen，进行协作讨论"""
import requests, json, sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

analysis = (PROJECT / "ai_suggestions/umaplay_analysis.md").read_text(encoding="utf-8")
prompt_base = (PROJECT / "ai_suggestions/discussion_prompt.txt").read_text(encoding="utf-8")

placeholder = ("[见 umaplay_analysis.md — 核心内容: YOLO+PaddleOCR+逻辑回归三层架构, "
               "训练评分系统(支援卡人数×1.5/彩虹训练×1.3/体力风险×0.3/属性目标权重), "
               "事件处理系统(关键词匹配+角色覆盖)]")

full_prompt = prompt_base.replace(placeholder,
    "=== 以下是完整分析报告 ===\n" + analysis)

print(f"Prompt length: {len(full_prompt)} chars, sending to Qwen...")

resp = requests.post(
    "http://localhost:1234/v1/chat/completions",
    json={
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": 0.7,
        "max_tokens": 8192,
        "stream": False,
    },
    timeout=300,
)
result = resp.json()
content = result["choices"][0]["message"]["content"]

out = PROJECT / "ai_suggestions/qwen_discussion_round1.md"
out.write_text(
    f"# Qwen + Claude Code 协作讨论 第1轮\n\n"
    f"## 讨论主题: Umaplay 训练评分移植 + PaddleOCR/ROCm 方案\n\n"
    f"---\n\n{content}",
    encoding="utf-8"
)

print(f"[Qwen] 返回 {len(content)} 字符")
print(f"[保存] {out}")
print("="*60)
print(content)
