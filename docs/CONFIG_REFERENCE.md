# SS自动化跑马配置说明

本文档覆盖项目的所有可配置项：`config.yaml` 主配置、训练规则 Profile、事件决策 Profile、商店物品 Profile。

---

## 一、config.yaml — 主配置文件

### 1.1 游戏设置 (`game`)

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `window_title` | string | `"StarSavior"` | 游戏窗口标题，脚本通过此关键字查找窗口（部分匹配即可） |
| `windowed_mode` | bool | `true` | 是否窗口模式。`true` 时截图使用 `PrintWindow`（支持遮挡），窗口不能最小化 |

---

### 1.2 属性定义 (`attributes`)

游戏内五种基础属性，脚本通过枚举值 1-5 来引用：

| 枚举值 | 属性名 | 训练行 | 说明 |
|--------|--------|--------|------|
| `1` | 力量 | 第1行 | 影响攻击力、暴击相关 |
| `2` | 生命 | 第2行 | 影响生命值、防御 |
| `3` | 韧性 | 第3行 | 影响耐力/敏捷相关 |
| `4` | 专注 | 第4行 | 影响命中、暴击率等 |
| `5` | 保护 | 第5行 | 影响防御、抗性 |

一般不修改此项，仅当游戏更新改变属性名称时需调整。

---

### 1.3 训练策略 (`training`)

#### 核心决策模式

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `rule_profile` | string | `"speed"` | **规则引擎 Profile 文件名**（不含 `.json` 扩展名）。设为空字符串 `""` 则回退到传统权重模式 |

**规则引擎模式（推荐）**：加载 `profiles/training/<rule_profile>.json`，按规则列表从上到下匹配，第一个命中的规则决定行动。可选值：

| Profile | 构建方向 | 失败率阈值 | 特点 |
|---------|----------|------------|------|
| `speed` | 攻击向 | 20% | 速度优先，图标阈值最低 |
| `attack` | 攻击向 | 25% | 攻击向，失败率容忍度稍高（25%） |
| `survival` | 生存向 | 20% | 生存向，优先韧性→体力→力量→保护→集中 |
| `default` | 攻击向 | 20% | 默认策略，与 speed 类似 |

#### 传统权重模式（仅 `rule_profile: ""` 时生效）

`priority` 是传统模式下最核心的配置：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `priority` | int (1-5) | `1` | **优先训练哪个属性**。填 `1`=力量, `2`=生命, `3`=韧性, `4`=专注, `5`=保护 |

传统模式工作原理：对每个训练行计算加权评分，评分 = `priority_bonus`（优先属性加分）+ `gain_score`（增益分）+ `failure_penalty`（失败率惩罚），选最高分行训练；若所有行失败率都超过阈值则休息。

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `failure_threshold` | int (0-100) | `20` | **普通失败率阈值 (%)**。某行失败率 >= 此值 且 无闪光训练时，该行被视为"差"，推荐休息 |
| `flash_fail_threshold` | int (0-100) | `80` | **闪光训练失败率阈值 (%)**。图标数 >= `flash_icon_threshold` 时，失败率容忍度提高到此值 |
| `flash_icon_threshold` | int | `3` | **闪光训练图标数门槛**。训练行检测到的支援卡图标 >= 此值时视为"闪光训练"（多人头高收益） |
| `min_gain_threshold` | int | `3` | 优先属性增益低于此值时认为"不够好"，降低该行评分 |
| `adventure_gain_threshold` | int | `100` | **冒险阈值**：单次训练增益 >= 此值时，即使失败率很高也值得训练。设为 `0` 禁用冒险逻辑 |

#### 传统模式 AI 决策权重 (`training.weights`)

仅在传统模式下生效，控制评分公式各部分的比重：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `priority_bonus` | int | `40` | 优先属性额外加分权重。值越大，越倾向于选优先属性的训练行 |
| `gain_score` | int | `35` | 属性增益分值权重。值越大，越看重单次训练能拿到的属性值 |
| `failure_penalty` | int | `25` | 失败率惩罚权重。值越大，越倾向于选失败率低的行 |

#### 高级选项

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `per_row_fail_rate` | bool | `true` | **每行独立失败率 OCR**。`true`：对每行展开详情后单独 OCR 读取失败率（更精确）。`false`：用体力推导统一失败率（简单但粗糙）。推荐开启 |
| `icon_counting` | bool | `true` | **HSV 图标计数**。`true`：用 HSV 颜色检测统计每行支援卡圆形图标数量来评估训练质量。`false`：仅用 OCR 增益值评估 |
| `advantage_multiplier` | float | `1.35` | **优势训练倍率**。检测到训练行右侧蓝色大拇指标记时，评分乘以该值；设为 `1` 可关闭倍率。失败率和休息规则仍优先 |
| `fail_rate_regions` | list | 5 个坐标区域 | 每行失败率 OCR 的百分比区域 `[x, y, w, h]`，分别为力量/体力/韧性/集中/保护训练行。修改后需用 `test_fail_rate_pos.py` 重新校准 |

---

### 1.4 休息策略 (`rest`)

当训练被判定为"不好"时，触发休息决策。三种休息方式按规则选择。

#### 触发条件

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `trigger_failure_threshold` | int (0-100) | `20` | 失败率超过此值 且 没有好的训练属性时触发休息 |
| `stamina_continue_threshold` | int (0-100) | `80` | 体力 >= 此值 且 金钱不够付费休息时 → 不休息，继续训练（避免浪费免费休息） |
| `first_year_rounds` | int | `12` | 前 N 轮视为"第一年"。第一年特殊规则：没有好训练 + 心情非 Best → 强制用付费休息（提高心情以加速成长）。设为 `0` 禁用第一年特殊策略 |

#### 三种休息方式

| 选项 | 名称 | 花费 | 体力恢复 | 额外效果 |
|------|------|------|----------|----------|
| 1 | 免费休息 | $0 | 30% | 无 |
| 2 | 付费休息 | $30 | 30%（50%概率50%） | 提升心情 |
| 3 | 高级休息 | $60 | 60% | 随机训练加成 |

每种选项的具体参数：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `options.1.name` | string | `"免费休息"` | 游戏内第一个休息选项的名称（OCR 匹配用） |
| `options.1.cost` | int | `0` | 花费金钱 |
| `options.1.stamina_recover` | int | `30` | 恢复体力百分比 |
| `options.2.name` | string | `"付费休息($30)"` | 第二个休息选项名称 |
| `options.2.cost` | int | `30` | 花费金钱 |
| `options.2.stamina_recover` | int | `30` | 基础恢复 |
| `options.2.stamina_bonus_chance` | int | `50` | 额外恢复到 50% 的概率 |
| `options.2.mood_boost` | bool | `true` | 是否提升心情 |
| `options.3.name` | string | `"高级休息($60)"` | 第三个休息选项名称 |
| `options.3.cost` | int | `60` | 花费金钱 |
| `options.3.stamina_recover` | int | `60` | 基础恢复体力百分比（50%概率恢复到80%） |
| `options.3.stamina_bonus_chance` | int | `50` | 额外恢复到80%的概率 |
| `options.3.training_bonus` | bool | `true` | 是否获得随机训练加成 |

#### 选择优先级规则

选择顺序（从上到下，第一个满足的生效）：

1. **第一年特殊规则**（优先级最高）：回合数 <= `first_year_rounds` + 心情非 Best → 强制选选项2（付费休息）。如果没钱则选选项1（免费休息）
2. **规则1**：心情 = Best 且 体力 > `best_mood_free_stamina` → 选项1（免费休息）— 体力充沛，省钱
3. **规则2**：心情 = Best 且 体力 <= `best_mood_free_stamina` → 选项3（高级休息$60）— 体力不足，花钱恢复+训练加成
4. **规则3**：心情非 Best → 选项2（付费休息$30）— 花钱提升心情
5. **规则4**（兜底）→ 选项1（免费休息）

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `selection.best_mood_free_stamina` | int | `40` | 关键阈值。心情Best时：体力>此值→免费休息，体力<=此值→高级休息 |

---

### 1.5 心情系统 (`mood`)

心情影响训练增益倍率，心情越好训练获得的属性越高。

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `levels.0` | string | `"Worst"` | 心情最差 |
| `levels.1` | string | `"Bad"` | 心情差 |
| `levels.2` | string | `"Normal"` | 心情一般 |
| `levels.3` | string | `"Good"` | 心情好 |
| `levels.4` | string | `"Best"` | 心情最佳 |

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `gain_multiplier.Worst` | float | `0.6` | 心情 Worst 时训练增益只有正常的 60% |
| `gain_multiplier.Bad` | float | `0.8` | 80% |
| `gain_multiplier.Normal` | float | `1.0` | 100%（基准） |
| `gain_multiplier.Good` | float | `1.2` | 120% |
| `gain_multiplier.Best` | float | `1.5` | 150% |

一般不需要修改此项，仅当游戏更新心情名称或倍率时调整。

---

### 1.6 操作设置 (`operation`)

控制脚本的操作节奏和最大运行轮次。

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `click_delay` | float | `0.15` | 每次点击后等待时间（秒）。降低可提速，但过小可能导致界面来不及响应 |
| `action_interval` | float | `0.3` | 每次操作之间的间隔（秒）。降低可提速 |
| `max_rounds` | int | `100` | **最大训练轮次**。设为 `0` 表示无限循环直到旅程结束。一轮 = 一次训练/休息/事件的完整操作 |
| `hotkey_start` | string | `"f5"` | 开始训练的热键（备用） |
| `hotkey_confirm` | string | `"enter"` | 确认训练的热键（备用） |

---

### 1.7 卡片选择 (`card_select`)

训练后偶尔出现奖励卡片选择，此配置控制选择策略。

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `speed_first` | bool | `true` | **速度最优先**。出现速度属性的卡片时优先选择，忽略下面的优先级列表 |

**攻击向卡片优先级**（`attack_priority`）：越靠前越优先，通过 OCR 关键字匹配：
```yaml
attack_priority:
  - "暴击率"      # 最高优先级
  - "暴击伤害"
  - "攻击力"      # 最低优先级
```

**生存向卡片优先级**（`survival_priority`）：
```yaml
survival_priority:
  - "生命力"
  - "防御力"
```

根据训练 Profile 的构建方向自动选用对应的优先级列表。

---

### 1.8 阿尔克那支援卡 (`arcanum`)

第一年优先点击支援卡以触发"闪光训练"（多支援卡同时出现的训练回合）。

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enabled` | bool | `true` | 是否启用支援卡优先点击。`true`：前 N 回合主动点击支援卡提升羁绊。`false`：不主动点支援卡，但训练决策仍会优先选择支援卡图标多的训练行（通过规则引擎的图标计数规则或 `train_most_icons` 动作实现） |
| `turn_limit` | int | `10` | **生效回合数**。前 N 回合优先点满羁绊，超过后恢复正常训练决策 |

#### 支援卡检测区域（一般不需要改）

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `card_row_y` | float | `0.60` | 卡片行顶部 Y 坐标（百分比） |
| `card_row_h` | float | `0.24` | 卡片行高度 |
| `card_area_x` | float | `0.18` | 卡片区域左边界 X |
| `card_area_w` | float | `0.65` | 卡片区域总宽度 |
| `card_count` | int | `5` | 支援卡数量 |

#### 点击与检测参数

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `bond_yellow_ratio` | float | `0.35` | 羁绊进度条中黄色像素占比阈值。>= 此值认为羁绊已满 |
| `post_click_wait` | float | `1.0` | 点击支援卡后等待时间（秒），等动画播放 |
| `click_y_offset` | float | `0.45` | 卡片点击位置（相对于卡片高度的偏移） |

#### 回合检测（一般不需要改）

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `turn_menu_x` | float | `0.065` | 菜单按钮 X 坐标 |
| `turn_menu_y` | float | `0.065` | 菜单按钮 Y 坐标 |
| `turn_ocr_region` | list | `[0.30, 0.69, 0.12, 0.08]` | 回合数 OCR 区域 `[x, y, w, h]` |

---

### 1.9 地区移动/月台 (`move_platform`)

跑马过程中出现的"地区移动"特殊事件，需要选择月台。

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `option_index` | int (1-2) | `1` | 选择的月台序号。attack/speed/default 用 1，survival 用 2 |

---

### 1.10 讨伐委托 (`commission`)

自动执行讨伐委托。

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enabled` | bool | `true` | 是否启用自动讨伐委托 |
| `rank_threshold` | int | `5` | **RANK 差阈值**。当前 RANK >= (建议等级 - 阈值) 时执行。值越大越激进（更低的等级也去打） |

---

### 1.11 商店 (`shop`)

控制自动购买行为。

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enabled` | bool | `true` | 是否启用自动商店购买。`false` 时遇到商店直接跳过 |

#### 物品类型优先级 (`shop.item_priority`)

越靠前的类型越优先购买。脚本通过 OCR 识别物品效果文字，用正则表达式分类：

| 类型 | 说明 | 匹配规则示例 |
|------|------|-------------|
| `mood_recovery` | 心情恢复 | 文字含"心情恢复" |
| `stamina_recovery` | 耐力恢复 | 文字含"耐力 N 恢复" |
| `potential_points` | 潜质点数增加 | 文字含"潜质点数 N 增加" |
| `vitality_potion` | 活力药水 | 文字含"活力药水"，训练失败时可继续训练 |
| `attribute_boost` | 属性增加 | 文字含"属性名 N 增加" |
| `training_exp` | 训练经验增加 | 文字含"XXX训练经验增加" |
| `status_buff` | 状态获得 | 文字含"XXX状态获得" |
| `random_effect` | 随机效果 | 文字含"随机效果"，点击查看加成属性 |
| `consumable` | 通用消耗品 | 兜底：文字含"名称 数量"格式的普通物品 |

#### 购买过滤规则（自动生效）

- 心情 = Best 时 → 跳过 `mood_recovery` 类物品（不需要恢复心情）
- 体力已满时 → 跳过 `stamina_recovery` 类物品
- 金币不足时 → 跳过售价超过当前金币的物品
- 物品已售罄（SOLDOUT）→ 跳过

**属性匹配规则**（`attribute_boost`、`training_exp`、`random_effect`）：
- 仅购买加成属性**匹配当前优先训练属性**的物品。例如优先力量训练时，只买力量相关的属性增加/训练经验/随机效果
- **`training_exp` 特殊限制**：专注和保护的训练经验**永不购买**（无论当前优先什么属性）。只有力量/体力/韧性的训练经验在匹配优先属性时才会购买
- **`random_effect`**：点击物品后 OCR 详情面板，解析其中出现的属性名（力量/体力/韧性/专注/保护/生命/命中），若任一匹配当前优先属性则购买

#### 交易界面坐标（已校准，一般不需要改）

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `trade_button_x` | float | `0.89` | 交易按钮 X（百分比） |
| `trade_button_y` | float | `0.56` | 交易按钮 Y |
| `appraisal_button_x` | float | `0.88` | 评鉴战按钮 X |
| `appraisal_button_y` | float | `0.47` | 评鉴战按钮 Y |

物品 OCR 区域和点击位置见配置文件中的 `item_regions`、`item_click_points`、`buy_confirm_x/y` 等。

---

### 1.12 旅程结束 (`journey_end`)

45 回合结束后的旅程结算画面。

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `auto_learn_skills` | bool | `false` | **自动学习技能**。`true`：自动进入技能界面选技能（功能未完善）。`false`：停留在旅程结束画面，脚本终止 |
| `confidence` | float | `0.8` | 模板匹配置信度阈值（0-1），越高越严格 |
| `ocr_lang` | string | `"ch"` | OCR 识别语言，`"ch"` = 中文 |

---

## 二、训练规则 Profile（`profiles/training/<name>.json`）

规则引擎模式下，脚本按 JSON 文件中的规则列表从上到下依次评估，**第一个命中的规则立即生效**。每个 Profile 文件自行定义规则列表。

### 规则结构

```json
{
  "id": "规则唯一ID",
  "field": "匹配字段",
  "operator": ">= | > | < | <=",
  "value": 阈值数值,
  "action": "命中后执行的动作",
  "enabled": true
}
```

`enabled: false` 可临时禁用某条规则。不带 `field`/`operator`/`value` 的规则（仅 `action`）作为兜底，始终匹配。

### 可用字段 (`field`)

| 字段名 | 说明 | 数据来源 |
|--------|------|----------|
| `any_fail_rate` | **所有训练行的最小失败率** (%) | 每行 OCR |
| `strength_fail_rate` | 力量训练失败率 | 每行 OCR |
| `stamina_fail_rate` | 体力训练失败率 | 每行 OCR |
| `agility_fail_rate` | 韧性训练失败率 | 每行 OCR |
| `focus_fail_rate` | 集中训练失败率 | 每行 OCR |
| `guard_fail_rate` | 保护训练失败率 | 每行 OCR |
| `any_gain` | **所有训练行中的最大增益值** (+N) | 每行 OCR |
| `strength_gain` | 力量训练增益值 | 每行 OCR |
| `stamina_gain` | 体力训练增益值 | 每行 OCR |
| `agility_gain` | 韧性训练增益值 | 每行 OCR |
| `focus_gain` | 集中训练增益值 | 每行 OCR |
| `guard_gain` | 保护训练增益值 | 每行 OCR |
| `strength_icons` | 力量训练支援卡图标数 | HSV 检测 |
| `stamina_icons` | 体力训练支援卡图标数 | HSV 检测 |
| `agility_icons` | 韧性训练支援卡图标数 | HSV 检测 |
| `focus_icons` | 集中训练支援卡图标数 | HSV 检测 |
| `guard_icons` | 保护训练支援卡图标数 | HSV 检测 |
| `strength_stat` | 当前力量属性值 | 属性面板 OCR |
| `stamina_stat` | 当前体力属性值 | 属性面板 OCR |

### 可用动作 (`action`)

| 动作名 | 实际行为 |
|--------|----------|
| `rest` | 进入休息流程 |
| `train_strength` | 点击第 1 行（力量训练） |
| `train_stamina` | 点击第 2 行（体力训练） |
| `train_agility` | 点击第 3 行（韧性训练） |
| `train_focus` | 点击第 4 行（集中训练） |
| `train_guard` | 点击第 5 行（保护训练） |
| `train_best_gain` | 自动选择增益值最高的行训练 |
| `train_most_icons` | 自动选择支援卡图标数最多的行训练（并列时按构建方向决定：攻击向→力/体/韧，生存向→韧/体/力） |

### 典型规则设计模式

规则列表通常包含以下几层（按顺序）：

1. **极端保底** — 失败率极高时强制休息（如 `>=100`）
2. **冒险层** — 总增益巨大时无视失败率直接训练（如 `>=100`）
3. **普通阈值** — 失败率到达阈值时休息（如 `>=20`）
4. **闪光训练** — 某行图标数达门槛时优先训（如 `>=3`）
5. **兜底** — 始终匹配的默认动作

### 四个预设 Profile 的差异

| 对比项 | `speed` | `attack` | `survival` | `default` |
|--------|---------|----------|------------|-----------|
| 构建方向 | 攻击 | 攻击 | 生存 | 攻击 |
| 普通休息阈值 | 20% | **25%** | 20% | 20% |
| 图标门槛 | >=3 | >=3 | >=3 | >=3 |
| 冒险阈值 | >=100 | >=100 | >=100 | >=100 |
| 图标规则顺序 | 力量→体力→韧性→集中→保护 | 力量→韧性→体力→集中→保护 | **韧性→体力→力量→保护→集中** | 力量→韧性→体力→集中→保护 |
| 兜底动作 | 训练力量 | 训练力量 | 训练韧性 | 训练力量 |

关键区别：
- **attack** 的失败率容忍度最高（25%），图标规则中力量排在体力前面
- **survival** 图标规则顺序完全不同：韧性排在最前面，兜底训练韧性
- **speed** 与 **default** 类似，但 `speed` 使用较为激进的速度优先策略

### 自定义规则

可以直接编辑 JSON 文件添加/修改规则。规则按 `rules` 数组中的顺序从上到下评估。例如：

```json
{
  "id": "my_custom_rule",
  "field": "strength_gain",
  "operator": ">=",
  "value": 50,
  "action": "train_strength",
  "enabled": true
}
```

这条规则的意思是：如果力量训练增益 >= 50，就训练力量（在评估到这条规则时生效）。

---

## 三、事件决策 Profile（`profiles/events/<name>.json`）

事件 Profile 与训练 Profile 同名，通过 `config.yaml` 中的 `training.rule_profile` 自动关联。

### 事件记录结构

```json
{
  "id": "事件唯一ID",
  "event_name": "事件中文名称",
  "category": "触发时期（如：三月中旬/随机事件）",
  "status": "confirmed | pending | conditional",
  "recommended_option": 1,
  "options": [
    {
      "index": 1,
      "keyword": "选项1的关键词（用于OCR匹配）",
      "alias": ["别名1", "别名2"]
    }
  ],
  "note": "选此选项的原因说明",
  "option_y_overrides": [0.0, 0.0, 0.0]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 唯一标识符，自动生成或手动指定 |
| `event_name` | string | 事件中文名称（如"闲暇时间"） |
| `category` | string | 分类标签，说明事件出现的时期 |
| `status` | string | `"confirmed"` = 已验证可自动选择；`"pending"` = 首次遇到待人工确认；`"conditional"` = 需根据游戏状态判断（如攻击/生存向不同） |
| `recommended_option` | int | **推荐选择第几个选项**（1-based）。设为 `null` 表示尚未确定，脚本将选选项1 |
| `options` | array | 事件的所有选项列表 |
| `options[].index` | int | 选项序号（1-based） |
| `options[].keyword` | string | 选项的关键词，用于 OCR 文本模糊匹配（子串匹配） |
| `options[].alias` | array | 别名字符串列表，提高匹配成功率 |
| `note` | string | 备注说明，仅供人工参考，不影响脚本行为 |
| `option_y_overrides` | array | 可选。某些事件的选项 Y 坐标与默认不同，用此字段覆盖 |

### 匹配与自学习机制

1. **已知事件匹配**：OCR 识别屏幕上的选项文字 → 对每个已知事件的 `options` 做子串匹配 → 要求至少 50% 的选项匹配 → 使用该事件的 `recommended_option`
2. **自学习**：当所有已知事件都不匹配时 → 自动创建新条目，`status: "pending"`，`recommended_option: null` → 脚本选选项 1 作为兜底
3. **后续编辑**：用户可打开 JSON 文件，手动设置 `recommended_option` 和 `status`

### Profile 选择

事件的 Profile 文件名需要与训练 Profile 一致（如 `speed` 对 `speed.json`）。如果没有对应的事件文件，脚本会创建空文件并自学习。

---

## 四、商店物品 Profile（`profiles/shop/<name>.json`）

商店物品数据库，与训练 Profile 同名自动关联。

### 物品结构

```json
{
  "id": 1,
  "name": "普通牛奶",
  "type": "consumable",
  "keywords": ["普通牛奶"],
  "attr_name": null,
  "value": null,
  "priority": null
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | 自增 ID |
| `name` | string | 物品名称 |
| `type` | string | 物品类型，与 `item_priority` 列表中的类型对应 |
| `keywords` | array | 用于 OCR 匹配的关键词列表 |
| `attr_name` | string/null | 如果是 `attribute_boost` 类型，记录提升的是哪个属性 |
| `value` | int/null | 提升的数值 |
| `priority` | int/null | 手动优先级覆盖。`null` 使用默认的 `item_priority` 排序 |

### 自学习机制

遇到未知物品时：OCR 识别 → 正则分类（见 1.11 节）→ 自动添加到 JSON 文件。已识别的物品下次直接通过 `keywords` 匹配，无需重新正则分类。

---

## 五、配置生效方式

| 文件 | 生效时机 |
|------|----------|
| `config.yaml` | 脚本启动时读取 |
| `profiles/training/*.json` | 每次进入训练决策时懒加载 |
| `profiles/events/*.json` | EventHandler 初始化时加载，触发事件时实时匹配 |
| `profiles/shop/*.json` | ShopHandler 初始化时加载，扫描到新物品时追加保存 |

所有 JSON Profile 文件的修改**即时生效**，无需重启脚本。`config.yaml` 需重启脚本才能生效。
