# SS自动化跑马 - 日志系统说明

## 日志存放位置

```
Starsavior/
  logs/
    run_20260526_143025.json    # 第1把跑马
    run_20260526_152110.json    # 第2把跑马
    run_20260526_164533.json    # 第3把跑马
    ...
```

- 每把跑马生成 **一个** JSON 文件
- 文件名 = `run_` + 启动时间 (`年月日_时分秒`)
- 跑马结束后自动保存，`logs/` 目录不存在时自动创建

## 多开跑马

如果要同时开多个游戏窗口跑马：

1. 复制整个 `Starsavior` 文件夹到不同目录
2. 每个副本用不同的 `config.yaml`（改 `game.window_title` 区分窗口）
3. 各自独立启动

每个副本的日志都在各自目录下的 `logs/` 里。跑完后把所有 `logs/` 里的 JSON 文件汇总到一个地方即可。

汇总后目录结构示例：

```
collected_logs/
  attack/                        # 攻击向跑马
    run_20260526_140001.json
    run_20260526_150530.json
    run_20260526_161200.json
  survival/                      # 生存向跑马
    run_20260526_141500.json
    run_20260526_152800.json
  speed/                         # 速度向跑马
    run_20260526_143000.json
    run_20260526_154500.json
    run_20260526_170200.json
```

按构建方向分目录有助于后续分策略分析。

## 日志文件结构

每个 JSON 文件包含三部分：

### 1. 元数据

```json
{
  "run_id": "20260526_143025",
  "started_at": "2026-05-26T14:30:25",
  "duration_seconds": 1847.3,
  "profile": "speed",
  "build_direction": "attack",
  "exit_reason": "journey_end"
}
```

| 字段 | 含义 |
|------|------|
| `run_id` | 本次跑马唯一ID = 启动时间 |
| `duration_seconds` | 跑马总耗时（秒）|
| `profile` | 规则配置名（speed/attack/survival/default/traditional）|
| `build_direction` | 构建方向（attack/survival）|
| `exit_reason` | 退出原因（journey_end正常结束 / max_rounds达上限 / keyboard_interrupt手动中断）|

### 2. 摘要统计

```json
{
  "summary": {
    "total_turns": 42,
    "total_actions": 126,
    "train_count": 30,
    "rest_count": 8,
    "shop_visits": 3,
    "shop_items_bought": 5,
    "events_encountered": 7,
    "arcanum_clicks": 6
  }
}
```

### 3. 每回合详细记录

```json
{
  "turns": [
    {
      "round": 5,
      "turn": 5,
      "timestamp": "2026-05-26T14:32:10",
      "state_before": {
        "stamina": 72.0,
        "mood": "Good",
        "money": 85
      },
      "actions": [
        {
          "type": "decision",
          "ts": "2026-05-26T14:32:10.123",
          "mode": "rule_engine",
          "profile": "speed",
          "matched_rule": "strength_icons_3",
          "action": "train_strength",
          "target_row": 0,
          "metrics": {
            "icon_counts": [3, 1, 2, 0, 1],
            "fail_rates": [18, 20, 22, 25, 28],
            "gain_values": [5, 3, 2, 1, 1],
            "strength_stat": 320,
            "stamina_stat": 280
          }
        },
        {
          "type": "train",
          "ts": "2026-05-26T14:32:12.456",
          "row_index": 0,
          "attribute": "力量训练",
          "metrics": {
            "rule_id": "strength_icons_3",
            "builtin_default": false
          }
        }
      ]
    }
  ]
}
```

### 关键动作类型

| type | 含义 | 关注字段 |
|------|------|---------|
| `decision` | 决策过程 | `matched_rule`, `action`, `icon_counts[5]`, `fail_rates[5]`, `gain_values[5]` |
| `train` | 执行训练 | `attribute`, `row_index` |
| `rest` | 执行休息 | `option`, `cost`, `stamina_before`, `reason` |
| `shop` | 商店交易 | `scanned[3]` 物品列表, `bought[]` 实际购买, 当日 `state` |
| `event` | 事件选择 | `event_id`, `event_name`, `option_selected`, `auto_learned` |
| `arcanum` | 支援卡点击 | `card_index`, `priority_attr` |
| `card_select` | 奖励卡片选择 | `card_index`, `reason`, `card_texts[3]` |
| `handler` | 画面处理器调度 | `name` 处理器名, `elapsed_ms` 耗时 |

## 后续 AI 接入

日志系统为后续接入 AI 模型（本地 Qwen3-8B / DeepSeek-R1-8B）做了以下准备：

1. **训练数据积累** — 每条决策都记录了当前的 `icon_counts[5]`、`fail_rates[5]`、`gain_values[5]`、属性面板值，这些都是 AI 模型的特征输入
2. **决策-结果闭环** — 回合 `N` 的 `decision` + 回合 `N+1` 的 `state_before` 构成因果对：做完决策后，体力/心情/金钱如何变化
3. **商店选择偏好** — `shop.scanned` vs `shop.bought` 展示了扫描到的物品和实际购买决策的差异，可用于训练购买偏好模型
4. **事件决策标注** — `auto_learned: true` 表示该事件尚未有人工标注，累积后可批量 review 并确认推荐选项

建议每跑完 50-100 把后，汇总全部 JSON 到一个目录，届时接入 AI 进行离线分析。
