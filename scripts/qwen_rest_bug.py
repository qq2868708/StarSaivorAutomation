"""将日志问题发送给 Qwen 分析"""
import requests
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

prompt = """## 问题
Starsavior 训练机器人在第2轮做出了错误的休息决策。以下是日志和代码分析：

### 日志
```
[状态] 体力=73% | 失败率=27% | 心情=Good | 金钱=$89
[属性] 力量=495 | 体力=N/A | 韧性=79 | 专注=177 | 保护=10
  [1] 力量训练: icons=1, failRate=27%
  [2] 体力训练: icons=1, failRate=27%
  [3] 韧性训练: icons=2, failRate=27%
  [4] 集中训练: icons=1, failRate=27%
  [5] 保护训练: icons=1, failRate=0%
[规则引擎] 决策: matched=rest_normal_fail, action=rest
```

### 根因分析
- speed.json 规则: `rest_normal_fail`: any_fail_rate >= 20 → rest
- rule_engine.py: `any_fail_rate` 取的是优先属性行（攻击方向=力量）的失败率，即 27%
- 27% >= 20% → 触发休息
- 但保护训练失败率=0%，是安全可训练的选项，被完全忽略

### 代码位置
rule_engine.py `_builtin_probe` 第 500 行:
```python
if priority_fail_rate > context.legacy_fail_rate_threshold:
    return ...REST...
```

### 可能的修复方案
1. 修改 _builtin_probe: 休息决策前先检查是否存在任意行失败率 < 阈值，有则不休息改为评分选最优
2. 新增 min_fail_rate 字段: 所有行失败率的最小值，规则引擎用 min_fail_rate >= 20 而不是 any_fail_rate >= 20
3. 提高 rest_normal_fail 阈值为 >= 30: 简单但不治本

用中文回答，选哪个方案最好以及为什么，100字以内。"""

resp = requests.post(
    "http://localhost:1234/v1/chat/completions",
    json={
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1024,
        "stream": False,
    },
    timeout=180,
)
content = resp.json()["choices"][0]["message"]["content"]
print(content)

out = PROJECT / "ai_suggestions" / "qwen_rest_bug.md"
out.write_text("# Qwen 对休息决策 Bug 的分析\n\n" + content, encoding="utf-8")
print(f"\n[保存] {out}")
