"""
StarSavior 训练自动化机器人 - 主入口

用法:
  python main.py                 # 正常启动
  python main.py --calibrate     # 校准模式（设置模板图片和区域坐标）
  python main.py --debug         # 调试模式：截图+OCR，不执行操作
  python main.py --auto --develop # 开发调试：未知界面持续观察，不因 Unknown 终止
  python main.py --auto --resume  # 从 runtime/checkpoint.json 重新识别当前界面
  python main.py --simulate      # 决策模拟器：测试AI决策逻辑（无需游戏）
"""

import sys
import argparse
import os
import yaml
from pathlib import Path


def load_config():
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("[错误] 找不到 config.yaml")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def simulate(config):
    """决策模拟器 - 手动输入状态和属性来测试AI决策"""
    from src.decision import DecisionEngine, TrainingOption, GameState

    engine = DecisionEngine(config)
    attr_names = list(engine.attributes.values())

    print("\n" + "=" * 60)
    print("  AI 决策模拟器")
    print("  输入游戏状态和训练选项，查看AI如何决策")
    print("  输入 'q' 退出")
    print("=" * 60)

    while True:
        print(f"\n{'─'*60}")
        try:
            stamina = input("  体力值 (0-100, 默认100): ").strip()
            if stamina.lower() == 'q':
                break
            stamina = float(stamina) if stamina else 100

            mood = input("  心情 (Worst/Bad/Normal/Good/Best, 默认Normal): ").strip()
            if mood.lower() == 'q':
                break
            mood = mood if mood else "Normal"

            money = input("  金钱 (默认100): ").strip()
            if money.lower() == 'q':
                break
            money = float(money) if money else 100

            round_num = input("  当前轮次 (默认1): ").strip()
            if round_num.lower() == 'q':
                break
            round_num = int(round_num) if round_num else 1

            priority = input(f"  优先属性ID 1-5 ({attr_names}, 默认{engine.priority}): ").strip()
            if priority.lower() == 'q':
                break
            if priority:
                engine.priority = int(priority)

        except (ValueError, EOFError):
            print("  输入无效，请重试")
            continue

        state = GameState(stamina=stamina, mood=mood, money=money, round_num=round_num)
        print(f"\n  状态: {state}")

        # 读取训练选项增益
        print(f"  输入各属性增益 (回车跳过):")
        gains = []
        for i, name in enumerate(attr_names, 1):
            val = input(f"    {name}: ").strip()
            if val.lower() == 'q':
                return
            gains.append(float(val) if val else 0)

        options = [
            TrainingOption(i + 1, name, gains[i], state.failure_rate)
            for i, name in enumerate(attr_names)
        ]

        # AI 决策
        action, target = engine.make_decision(options, state, debug=True)

        if action == "train":
            print(f"\n  >>> AI决策: 训练 [{target.attr_name}] "
                  f"(增益={target.gain}, 失败率={target.failure_rate}%)")
        else:
            print(f"\n  >>> AI决策: 休息 [{target.name}] "
                  f"(花费${target.cost}, 恢复{target.stamina_recover}%体力)")


def main():
    # Config and profiles belong to this script, regardless of shortcut/start directory.
    os.chdir(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser(description="StarSavior 训练自动化（本地修复版）")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="检查环境和 OCR，不发送游戏输入")
    modes.add_argument("--debug", action="store_true", help="只截图和识别，不点击")
    modes.add_argument("--simulate", action="store_true", help="离线决策模拟")
    modes.add_argument("--calibrate", action="store_true", help="校准工具")
    modes.add_argument("--auto", action="store_true", help="直接开始训练")
    parser.add_argument("--develop", action="store_true",
                        help="开发调试模式：未知界面不触发卡死终止；正式运行不要使用")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="默认读取 checkpoint 并重新识别当前界面；--no-resume 忽略历史状态")
    parser.add_argument("--require-game", action="store_true", help="检查时要求找到游戏窗口")
    args = parser.parse_args()
    if args.require_game and not args.check:
        parser.error("--require-game 需要与 --check 一起使用")
    config = load_config()
    from src.trainer import Trainer

    if args.check:
        from src.self_check import run_startup_check
        return run_startup_check(Trainer(config, development_mode=args.develop), require_game=args.require_game)
    if args.calibrate:
        trainer = Trainer(config, development_mode=args.develop)
        trainer.calibrate()
    elif args.debug:
        trainer = Trainer(config, development_mode=args.develop)
        print("调试模式: 截取游戏画面 + OCR 全屏识别 + 屏幕检测")
        screenshot = trainer.capture.capture_game()
        trainer.capture.save_capture(screenshot, "_debug")
        print("截图已保存到 templates/_debug.png")

        h, w = screenshot.shape[:2]
        print(f"\n截图尺寸: {w}x{h}")

        # 屏幕检测
        screen_type = trainer.ui.detect_screen(screenshot)
        print(f"屏幕检测: {screen_type}")

        # 分支文字
        branch = trainer.ui.detector.read_branch_text(screenshot)
        print(f"分支区域文字: '{branch[:120] if branch else '(空)'}'")

        # 读取状态
        state = trainer.ui.read_game_state(screenshot)
        print(f"识别状态: 体力={state['stamina']:.0f}%, "
              f"心情={state['mood']}, 金钱=${state['money']}")

        # 全屏 OCR
        print(f"\n全屏 OCR 结果 (前40条, 含相对坐标):")
        import numpy as np
        texts = trainer.ui.ocr.recognize_detailed(screenshot)
        for i, (text, conf, bbox) in enumerate(texts[:40]):
            if bbox is not None:
                pts = np.array(bbox)
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                xp = cx / w
                yp = cy / h
                print(f"  [{conf:.2f}] ({xp:.3f},{yp:.3f}) {text}")
            else:
                print(f"  [{conf:.2f}] (--,--) {text}")
    elif args.simulate:
        simulate(config)
    elif args.auto:
        trainer = Trainer(config, development_mode=args.develop)
        trainer.run(auto_start=True, resume=args.resume)
    else:
        trainer = Trainer(config, development_mode=args.develop)
        trainer.run(resume=args.resume)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已停止。")
    except Exception as error:
        print(f"\n[启动失败] {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
