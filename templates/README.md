# 模板图片和区域坐标设置指南

## 第一步：运行校准模式
```bash
python main.py --calibrate
```
截取游戏画面保存到 `templates/_calibration_full.png`。

## 第二步：在截图上获取坐标
用画图软件打开 `_calibration_full.png`，找到以下区域的像素坐标：

### 状态栏区域（屏幕顶部）
- **体力值**: 显示百分比数字的位置（如 80%）
- **心情**: 显示英文心情词的位置（Worst/Bad/Normal/Good/Best）
- **金钱**: 显示金钱数字的位置（如 $120）

### 训练选项区域
- 5 个属性训练按钮

### 休息区域（屏幕下方）
- **休息入口按钮**: 点击后显示 3 种休息方式
- **3 个休息选项按钮**:
  - 免费休息 (恢复 30%)
  - 付费休息 $30 (恢复 30%, 概率 50%, 提升心情)
  - 高级休息 $60 (恢复 60%, 随机训练加成)

## 第三步：修改 `templates/regions.json`
将坐标填入 regions.json，参考文件中的示例格式。

## 第四步：验证
```bash
python main.py --debug    # 截图+OCR 验证
python main.py --simulate # AI决策模拟器（无需游戏）

## 第五步：开始训练
```bash
python main.py
```

## 可选：截取按钮模板图片
使用 Win+Shift+S 截取关键按钮保存到本目录，提高识别精度：
- `train_button.png` - 训练入口按钮
- `rest_button.png` - 休息入口按钮
- `confirm.png` - 确认按钮
