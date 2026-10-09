# SS自动化跑马 - 日志系统说明

## 日志存放位置

```
Starsavior/
  logs/
    run_20260526_143025_abcd1234.json   # 单次脚本运行片段
    journeys/
      <journey_id>.json                # 同一旅程的配置、运行片段与最终结果
    ...
```

- 每次启动生成一个 `run_*.json` 文件；重启恢复的运行通过共同的 `journey_id` 关联。
- `logs/journeys/<journey_id>.json` 是旅程汇总。两类文件均为本地生成数据，忽略版本控制。
- 决策与输入前后、调度结束、安全暂停和退出时保存；旧日志保留原格式，不补造角色或结果。

## 复盘记录（运行日志 schema_version=2）

汇总记录包括：

- `target_history`：目标角色、玩法、装备技能、训练方向和目标版本；属于计划，不作为已选角色或已装备技能的证明。
- `actual_setup` / `setup_history`：实际角色、旅程记录、支援卡。每项带值、时间、来源和证据；预输入/结算补录标为 `user_declared`，后续自动选卡处理器可在核对成功后通过 `logger.record_setup(..., "screen_confirmed", evidence)` 写入。卡片可附 ID、稀有度、等级、突破、借用状态和槽位。
- `segments`：同一旅程各次运行的文件、退出原因及父运行。checkpoint 带旅程编号，因此恢复不会被当成一次新的完整跑马。
- `outcome`：旅程是否结束、结束画面、五维最终属性、综合等级、潜质点数、最终评鉴结果、已装备技能及目标是否达成；`execution_mode` 会明确记录为 `normal` 或 `exploration`。
- `exploration`：未知事件的分支观测次数与效果证据。数字效果可用于后续方向评分；只有文字/转场效果的事件在有限样本后也会收敛，不会因无法 OCR 数字而卡住旅程。
- `data_quality`：缺失字段、是否从开局完整记录、旅程中是否更改过目标。中途接入会保留缺少前段的标记。

`null` 表示未核对；`[]` 表示明确没有携带/装备。得到后续材料不代表已经合成或装备目标技能，目标达成只根据明确的已装备技能记录计算。到达旅程结束页也不自动等于评鉴胜利。

单次运行文件另有 `strategy` / `strategy_history`、`observations`、`inputs`：保存配置与实际规则快照及内容摘要、页面 OCR（置信度与坐标）、截图、逐次观测变化、输入的 prepared/sent/uncertain 阶段及来源。动作关联观测编号和目标版本；观测间变化只说明该时间区间发生了变化，不能当成某一次操作的确定收益。既有 `state_before` 来自决策读取器，可能使用缓存或兜底值；观测中无法明确读出的值留空。

观测中的 `game_date` / `game_turn` 只读取画面明确显示的月份旬数、X/45；`round` 是脚本调度次数，不能当作游戏回合。训练指标保存 `known_*_mask`，每位对应一行，未识别行的零值不能当成实际零收益。runtime 事件也带旅程编号。

最终属性只能来自真正的结束画面或人工核对，不能拿最近一次主菜单属性充当结果。OCR 只接受清晰、唯一的属性标签和数值，不把“力量+10”、建议等级或技能效果计入最终属性。未读出的结果保留截图和原文，在运行页「旅程复盘记录」中于旅程结束后补录；原始观测和补录历史都保留。

本地接口：`GET /api/journey-records` 查看最近汇总；`PATCH /api/journey-records/{journey_id}` 补录实际携带配置与结果。本轮进程退出前拒绝补录最终结果。运行页预检查中的「实际携带配置」为选填资料，不影响策略选择。自动选角、选择旅程记录和支援卡的流程将在后续阶段接入这些字段。

旅程结束后，WebUI 的「读取当前旅程信息」会读取左侧放大镜的「旅程信息」页；已打开的页面可以通过后台窗口截图读取。若当前仍是结算页，扫描器只会在确认结算版式且 StarSavior 处于前台时点击一次放大镜；不会恢复自动化，也不会处理奖励或潜质。角色、等级、五项属性、潜质点数和可读的继承旅程记录会以 `screen_confirmed` 写入并保留 OCR 与截图证据。支援卡页面只有立绘、稀有度和等级而没有稳定的卡名文字时，支援卡名称保持 `null`，继续由人工补充，避免按图猜名。重复扫描不会覆盖已由人工核对的字段。

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
2. **状态变化记录** — 关联决策、输入与后续观测；复盘需要考虑区间内其他操作、随机收益及识别缺失，不直接把相邻回合解释成因果对
3. **商店选择偏好** — `shop.scanned` vs `shop.bought` 展示了扫描到的物品和实际购买决策的差异，可用于训练购买偏好模型
4. **事件决策标注** — `auto_learned: true` 表示该事件尚未有人工标注，累积后可批量 review 并确认推荐选项

建议每跑完 50-100 把后，汇总全部 JSON 到一个目录，届时接入 AI 进行离线分析。
