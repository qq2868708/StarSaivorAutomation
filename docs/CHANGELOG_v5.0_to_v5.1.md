# StarSavior v5.1 更新日志

> 自 `1ac4fc5` (v5.0: 完整训练自动化机器人) 以来所有代码优化

**变更统计**: 13 个文件, +5850 行, -791 行, 新增 `profiles/skills/` `scripts/` 等目录

---

## 一、核心引擎

### 1. 规则引擎 → Umaplay 训练评分 (`src/rule_engine.py`)

v5.0 的默认决策基于"图标计数优先"选择训练行。v5.1 重写为 **Umaplay 评分模型**:

```
评分 = 基础增益 × 图标加成 × 闪光加成 × 体力惩罚 × 目标降权
```

- **闪光训练检测**: 新增 `is_flash` 字段, 闪光训练的增益值 ×1.5 倍率
- **图标加成**: 0图标=×0.5, 1-2图标=×1.0, 3图标=×1.3, 4+图标=×1.6
- **体力惩罚**: 训练后体力<0 时按比例扣分
- **目标降权**: 接近属性目标时降低该训练优先级 (避免溢出)
- **失败率阈值**: 可在配置中独立设置各场景的阈值

### 2. OCR 引擎 → 混合 OCR (`src/recognition.py`)

新增 `HybridOCR` 混合引擎:

| 方法 | 引擎 | 用途 | 速度 |
|------|------|------|------|
| `recognize()` | WinOCR | handler 检测/状态读取/训练指标 | 即时 |
| `recognize_detailed()` | DirectML OCR (PP-OCRv4) | 全屏校准/高精度识别 | ~0.5s |

---

## 二、检测优化 (`src/recognition.py`)

### 3. 事件/交易/训练三方误判防御

v5.0 存在的三个误判问题全部修复:

| 问题 | 根因 | 修复 |
|------|------|------|
| 天气事件被误判为训练 | `is_training_screen` 将事件选项"继续训练"当作训练关键词 | 新增 `_is_event_context` 排除检查, 要求 ≥2 个训练关键词 |
| 合宿事件被误判为交易 | `_is_trade_text` 的 `\d{1,4}` 匹配事件选项编号"1""2""3" | 改为 `\d{2,4}` (至少2位数字), 并在 `is_trade_screen` 开头加事件排除 |
| `is_event_marker_screen` 不检测合宿 | 仅检查"旅程事件""阿尔克那事件"等 | 新增"合宿""固定事件""连续事件"关键词 + 分支文本回退检查 |

### 4. 闪光训练双重检测 (`src/trainer.py`)

v5.0 无此功能。v5.1 实现:

- **方法1 HSV**: H:12-45, S≥35, V≥130 的暖黄色像素占比 >3%
- **方法2 相对饱和度**: 各行 S 均值与全局均值比较, 显著偏高即为闪光
- 两方法独立运行, 结果取并集 (最多可检测5行全闪光)

### 5. 屏幕截图超时保护 (`src/capture.py`)

`PrintWindow` 在目标窗口无响应时会永久阻塞。v5.1 将截图放入子线程 + `CAPTURE_TIMEOUT=5s` 超时, 防止卡死。

---

## 三、训练逻辑 (`src/trainer.py`)

### 6. 合宿策略系统

新增完整的合宿 (30-38回合) 策略:

- **预合宿蓄力** (25-29回合): 低增益训练 → 休息蓄力
- **合宿激进训练**: 加成属性增益>45→无视失败率继续训练; ≤45→按正常阈值休息
- **合宿休息决策**: 心情Best且金钱充裕→冥想室($60); 心情非Best且有钱→住处($30); 否则→露宿(免费)
- **位置感知**: 弗洛拉(力量加成) vs 卡莱德(体力加成)
- 37回合(合宿最后一回合)预留金钱用于交易

### 7. 保护机制

新增三重保护 (`config.yaml:safety`):

| 机制 | 默认值 | 行为 |
|------|--------|------|
| 运行超时 | 45分钟 | 超时自动暂停, 触发 AI 分析 |
| 卡死检测 | 连续5轮 Unknown | 截图保存现场, 触发分析 |
| 轮超时 | 60秒 | 单轮超过此时间视为卡死 |

### 8. 扫描模式

新增三种扫描模式 (`config.yaml:training.scan_mode`):

- `"fast"`: 只读1行 (最快)
- `"normal"`: 读3行 (平衡, 默认)
- `"focus"`: 逐行读全部5行 (最精确, 最慢)

### 9. 属性目标系统

可为各属性设目标值 (`config.yaml:training.target_stats`), 接近目标时自动降低该训练的评分权重, 避免属性溢出。

---

## 四、处理器 (`src/handlers.py`)

### 10. 事件处理增强

- **热键事件**: 特定事件使用键盘数字键选择 (非鼠标点击), 如弗洛拉沙滩排球
- **事件配置匹配**: 从 `profiles/events/{profile}.json` 加载已知事件, 自动选择推荐选项
- **自动学习**: 遇到未知事件自动添加到配置, 标记 `status: "pending"` 待人工确认
- **重复事件防护**: OCR 相似度 ≥70% 跳过, 防止卡死

### 11. 商店物品配置匹配

从 `profiles/shop/{profile}.json` 加载已知物品, 优先匹配已知物品再回退到正则分类。未知物品自动学习添加到配置。

### 12. 旅程结束技能选择 (`JourneyEndHandler`)

全新实现, 见下方"技能选择系统"。

### 13. 休息增强

- **合宿休息**: 三级选择 (冥想室/住处/露宿)
- **休息验证**: `_wait_for_screen` 返回 bool, 超时后 ESC 兜底而非无限循环

### 14. 键盘输入 (`src/controller.py`)

从 PyAutoGUI 迁移到纯 SendInput:

- 新增 `send_number_key(1-4)` — 事件热键选择
- 新增 `send_space()` — 跳过战斗
- `send_escape()` 重写 — SendInput 键盘事件
- 完整的 `KEYBDINPUT` 结构体 + `INPUT_UNION`

---

## 五、技能选择系统

### 15. 技能配置 (`profiles/skills/`)

新建目录, 按构建方向配置技能优先级:

| 文件 | 方向 | 技能数 |
|------|------|--------|
| `attack.json` | 攻击 | 10 (星光轨迹→攻击技巧) |
| `speed.json` | 速度 | 14 |
| `survival.json` | 生存 | 12 |
| `default.json` | 通用 | 12 |

每个技能包含: 名称、优先级、OCR 关键词列表、是否为多级技能。

### 16. 攻击向技能优先级 (用户指定)

| 优先级 | 技能名 | 特性 |
|--------|--------|------|
| 10 | 星光轨迹-XXX号 | XXX为变量, 前缀匹配 |
| 9 | 血之回声 | |
| 8 | 锐利成长 | |
| 7 | 锐利得势 | |
| 6 | 洞察星痕 | |
| 5 | 速度星痕 | |
| 4 | 速度天赋 | |
| 3 | 暴击天赋 | |
| 2 | 洞察感知 | 感知类, 最多5级 |
| 1 | 攻击技巧 | 技巧类, 最多5级 |

---

## 六、日志与耗时分析

### 17. 回合计时 (`src/logger.py` + `src/trainer.py`)

每轮训练记录耗时, 终端打印 `[计时] 本轮耗时 X.Xs`。

日志 JSON 新增 `timing` 段:

```json
{
  "timing": {
    "total_seconds": 885.3,
    "handlers": { "事件选择": { "count": 5, "avg_ms": 450, ... } },
    "per_turn": { "avg_seconds": 12.3, "max_seconds": 45.2 },
    "slowest_turns": [ { "round": 12, "duration_seconds": 45.2 } ],
    "breakdown": { "handler_pct": 65, "other_time_seconds": 310 }
  }
}
```

### 18. 耗时分析工具 (`scripts/analyze_timing.py`)

```
python scripts/analyze_timing.py              # 自动分析最新日志
python scripts/analyze_timing.py logs/xxx.json # 指定日志
```

输出: Handler 耗时分布表、最慢回合 Top 5、瓶颈诊断、优化建议。

---

## 七、配置变更 (`config.yaml`)

新增约 40 行配置:

- `training.scan_mode` / `target_stats`
- `safety.*` (3个保护参数)
- `arcanum.icon_priority_limit`
- `camp.*` (8个合宿策略参数)
- `journey_end.auto_learn_skills` (false→true)

---

## 八、事件数据库 (`profiles/events/speed.json`)

较 v5.0 的少量事件, 新增 55+ 事件条目, 包括:

- 弗洛拉合宿事件 (沙滩排球等)
- 固定事件 (节日里的不速之客、突袭等)
- 支援卡事件 / 救援者事件
- 热键选择事件 (selection_method: "hotkey")
- 自动学习条目 (status: "pending")

---

## 九、自动化流水线 (`scripts/pipeline.py`)

全自动训练-分析-改进闭环:

```
跑马 → Qwen3-8B 分析日志 → Claude Code 自动修改代码 → 重跑验证
```

三种模式: 单次分析 / 监听模式 / 全自动循环模式。

---

## 十、新增脚本

| 文件 | 用途 |
|------|------|
| `scripts/pipeline.py` | 全自动闭环流水线 |
| `scripts/analyze_timing.py` | 运行后耗时分析 |
| `scripts/test_flash.py` | 闪光训练检测测试 |
| `scripts/test_shop_scan.py` | 商店扫描测试 |
| `scripts/screenshot.py` | 独立截图工具 |
| `scripts/classify_items.py` | 物品分类器测试 |
| `calibrate_state.py` | 属性区域校准 |
| `capture_screen.py` | 窗口捕获调试 |
| `mouse_coords.py` | 坐标检测工具 |
