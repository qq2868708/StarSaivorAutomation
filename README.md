# StarSavior 训练自动化

StarSavior 训练自动化是一个 Windows 桌面自动化工具，基于截图、OCR 识别与 Win32 输入模拟，围绕《Star Savior》（星之救援者）的训练流程提供自动化辅助。

## 免责声明

**本项目仅供学习交流使用，完全开源，严禁用于任何商业行为。**

StarSavior 训练自动化是非官方桌面自动化工具，与《Star Savior》的开发商、发行商、平台方或相关权利方无任何关联，也未获得其认可、赞助或授权。

使用本项目可能违反游戏服务条款、平台规则、活动规则或其他协议。使用者需自行确认并承担使用、修改、运行本软件产生的全部风险，包括但不限于：

- 账号处罚、封禁
- 数据损失、误操作
- 游戏不稳定或自动化行为带来的其他后果

本项目不修改游戏文件，不向游戏进程注入代码，不绕过技术保护措施。代码按现状（AS-IS）提供，不承诺安全性、可用性、正确性或适用于特定用途。

**如果您使用、修改或分发本项目，即表示您已阅读并同意上述声明。**

## 项目定位

- 面向真实游戏窗口的桌面自动化，非注入式修改
- 以页面 Handler 为中心拆分复杂流程，避免单文件状态机失控
- 以 `profiles/` 下的 JSON 配置文件驱动事件、商店、训练策略
- 支持传统权重模式与规则引擎模式双模式切换
- 后台窗口截图（PrintWindow），窗口被遮挡也能运行
- SendInput 硬件级点击模拟，不占用实际鼠标光标

## 当前能力

- 训练主流程自动化：事件选择、商店交易、卡片选择、训练决策、休息决策
- 规则引擎模式：懒加载扫描 → 多规则匹配 → 自动训练决策
- 传统权重模式：属性优先级 + 失败率阈值 + 增益评分
- 事件自动匹配：OCR 识别 → 已知事件库模糊匹配 → 自动选择推荐选项
- 商店自动购买：物品分类识别 → 条件过滤 → 优先级评分 → 多物品购买
- 旅程结束检测与自动终止
- 配置文件驱动，无需修改代码即可切换策略

## 技术栈

- Python 3.11+
- OpenCV（图像处理、模板匹配）
- PaddleOCR（中文 OCR 识别）
- NumPy（数值计算）
- PyYAML（配置文件解析）
- Win32 API（PrintWindow 截图 / SendInput 点击 / PostMessage）

## 仓库结构

```text
.
├── main.py                  # 主入口
├── ui_app.py                # 本地检查台：截图、决策、日志、事件与权重编辑
├── config.yaml              # 全局配置文件
├── src/
│   ├── trainer.py           # 训练主循环、Handler 链调度
│   ├── handlers.py          # 全部 Handler 处理器
│   ├── recognition.py       # 屏幕检测、OCR、游戏状态读取
│   ├── controller.py        # 窗口截图与鼠标键盘模拟
│   ├── decision.py          # 决策引擎（传统权重模式）
│   ├── rule_engine.py       # 规则引擎（JSON 规则驱动模式）
│   ├── icon_counter.py      # 训练图标计数
│   ├── fail_rate_ocr.py     # 每行失败率 OCR
│   ├── capture.py           # 窗口捕获底层
│   └── arcanum.py           # 支援卡检测
├── profiles/
│   ├── training/            # 训练规则配置文件
│   ├── events/              # 事件决策配置（按 profile 分文件）
│   └── shop/                # 商店物品配置（自动学习）
├── templates/               # 校准截图与模板
├── docs/                    # 辅助文档
└── SleepRunner/             # 参考项目（C# 原版）
```

## 环境要求

- Windows 10/11 x64
- Python 3.11+
- 可交互桌面上的《Star Savior》游戏窗口

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置文件

编辑 `config.yaml`，关键配置项：

```yaml
training:
  rule_profile: "speed"       # 规则引擎 profile: attack/survival/default/speed
  priority: 1                 # 传统模式优先属性: 1=力量 2=生命 3=韧性 4=专注 5=保护

shop:
  enabled: true               # 是否启用自动商店

journey_end:
  auto_learn_skills: false    # 是否自动学习技能（暂未完善）
```

### 3. 运行

```bash
python main.py
```

按 Enter 开始训练，Ctrl+C 中断。

### 4. 打开检查台

```bash
python ui_app.py
```

检查台不会连接游戏窗口，也不会发送输入。它读取 `logs/` 中的运行日志、
本地 PNG 截图、`profiles/events/document_verified.json` 和 `config.yaml`，
按“运行 / 事件库 / 配置”分 Tab 展示运行摘要、历史决策、日志、事件分支和
量化收益，并允许结构化修改事件分支和训练/事件决策权重。脚本会把产生决策的
游戏画面保存到 `logs/screenshots/`；在“运行”Tab 点击历史决策即可回看对应截图。
旧日志没有截图关联时会明确显示为不可用。修改后分别点击“保存事件”或“保存权重”；
下一次脚本启动时读取这些文件。

## Handler 优先级链

| 优先级 | Handler | 职责 |
|--------|---------|------|
| P0 | SkipHandler | 跳过战斗/过场动画 |
| P1 | BattleDefeatHandler | 战斗失败处理 |
| P2 | BattleLeaveHandler | 战斗结算退出 |
| P2 | CommissionHandler | 讨伐委托 |
| P2 | MainMenuHandler | 主菜单路由 |
| P3 | MovePlatformHandler | 地区移动/月台选择 |
| P4 | CampSelectHandler | 营地选择 |
| P5 | ShopHandler | 商店交易 |
| P6 | EventHandler | 事件选择 |
| P6 | TrainingHandler | 训练决策 |
| P10 | CardSelectHandler | 奖励卡片选择 |
| P18 | RestHandler | 休息决策 |
| P19 | JourneyEndHandler | 旅程结束 |
| P99 | UnknownHandler | 未知界面兜底 |

## 配置与运行时文件

- `config.yaml` — 全局配置（训练策略、商店、休息、属性定义等）
- `profiles/training/<profile>.json` — 规则引擎训练策略
- `profiles/events/<profile>.json` — 事件决策配置（首次遇到的未知事件自动添加）
- `profiles/shop/<profile>.json` — 商店物品数据库（首次遇到的未知物品自动添加）

## 文档索引

- [免责声明](#免责声明)
- [SleepRunner 参考项目](./SleepRunner/README.md)

## 当前状态

项目持续迭代中。目前已经完成 Handler 流水线架构移植、事件/商店配置系统、训练规则引擎、旅程结束处理等核心功能。后续计划：

- 完善旅程结束技能自动选择
- 提升 OCR 识别稳定性
- 增加运行日志与统计

目前需注意：

- Windows only
- 依赖真实窗口截图和 OCR，受分辨率、DPI、界面样式影响
- 需要游戏窗口处于可见状态（PrintWindow 支持遮挡，但窗口不能最小化）

## 开源许可

本项目代码完全开源，仅供学习交流使用。禁止用于任何商业用途或违反游戏服务条款的行为。

参考项目 [SleepRunner](./SleepRunner/) 的授权及免责声明见其各自目录。

## 更新日志

### v5.1 (2026-05-27)

- **规则引擎**: 改用 Umaplay 训练评分模型（基础增益 × 图标加成 × 闪光加成 × 体力惩罚 × 目标降权）
- **OCR 引擎**: 新增 HybridOCR 混合引擎（WinOCR 即时识别 + DirectML OCR 高精度识别）
- **误判修复**: 修复事件/交易/训练三方误判（天气事件、合宿事件、沙滩排球不再被错误识别）
- **闪光训练检测**: HSV + 相对饱和度双重检测，支持最多5行全闪光
- **合宿策略**: 预合宿蓄力、激进训练、三级休息选择（冥想室/住处/露宿）
- **技能选择**: 旅程结束时按优先级自动学习技能（攻击/速度/生存/通用四方向配置）
- **保护机制**: 运行超时、卡死检测、轮超时三重保护
- **耗时分析**: 每轮计时 + 日志 timing 段 + `scripts/analyze_timing.py` 分析工具
- **事件数据库**: 新增 55+ 事件条目，支持热键选择
- **输入模拟**: 从 PyAutoGUI 迁移到纯 SendInput
- **自动闭环**: `scripts/pipeline.py` 全自动训练-分析-改进流水线
