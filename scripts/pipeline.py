#!/usr/bin/env python
"""
全自动训练-分析流水线

流程:
  1. 启动 Starsavior 训练 (手动操作, 或自动触发)
  2. 训练结束 → 自动生成日志到 logs/
  3. 读取最新日志 → 发送给 Qwen3-8B 分析
  4. Qwen 返回策略建议 → 保存到 ai_suggestions/latest.md
  5. 在终端打印建议, 等待 Claude Code 应用修改
  6. 回到步骤 1 开始下一轮

用法:
  python scripts/pipeline.py                    # 单次: 分析最新日志
  python scripts/pipeline.py --watch            # 监听模式: 检测到新日志自动分析
  python scripts/pipeline.py --loop             # 循环模式: 等待 Claude Code 修改后继续
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 确保使用 venv 中的 Python (避免系统 Python 缺少 cv2 等依赖)
_VENV_PYTHON = Path(__file__).resolve().parent.parent / "venv" / "Scripts" / "python.exe"
_PYTHON_EXE = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable

from src.ai_analyzer import AIAnalyzer


# ============================================================
# 文件监视
# ============================================================

def _list_logs(log_dir: Path) -> list[Path]:
    """列出所有日志文件, 按时间排序"""
    if not log_dir.is_dir():
        return []
    return sorted(log_dir.glob("run_*.json"), key=lambda p: p.stat().st_mtime)


def watch_mode(analyzer: AIAnalyzer, log_dir: str = "logs"):
    """监听模式: 持续检测新日志, 出现后自动分析"""
    log_path = Path(log_dir)
    known = set(p.name for p in _list_logs(log_path))
    print(f"[监听] 当前已有 {len(known)} 个日志文件")
    print(f"[监听] 等待新日志... (Ctrl+C 退出)")
    print()

    try:
        while True:
            current = set(p.name for p in _list_logs(log_path))
            new_files = current - known

            if new_files:
                newest = max(new_files, key=lambda n: (log_path / n).stat().st_mtime)
                log_file = log_path / newest
                print(f"\n[监听] 检测到新日志: {log_file.name}")
                print(f"[监听] 等待 3 秒以确保文件写入完成...")
                time.sleep(3)

                analyzer.analyze(log_file)
                known = current
                print(f"\n[监听] 继续等待新日志... (Ctrl+C 退出)")
                print()

            time.sleep(2)

    except KeyboardInterrupt:
        print("\n[监听] 已退出")


# ============================================================
# 循环模式 (自动: 跑马 → 分析 → Claude Code 修代码 → 重跑)
# ============================================================

def _read_done_marker() -> Optional[str]:
    """读取 ai_suggestions/done_marker 确认 Claude Code 已完成修改"""
    marker = Path("ai_suggestions/done_marker.txt")
    if marker.is_file():
        content = marker.read_text().strip()
        marker.unlink()
        return content
    return None


def _write_trigger_marker():
    """写入触发标记让 Claude Code 知道需要应用建议"""
    trigger = Path("ai_suggestions/trigger.txt")
    trigger.parent.mkdir(exist_ok=True)
    trigger.write_text(datetime.now().isoformat())


def _wait_for_claude_fix(timeout_minutes: int = 30) -> bool:
    """等待 Claude Code 读取建议并修改代码 (通过 done_marker 确认)"""
    print(f"\n[循环] 等待 Claude Code 应用修改... (最多 {timeout_minutes} 分钟)")
    print(f"[循环]   标记文件: ai_suggestions/done_marker.txt")
    print(f"[循环]   请对 Claude Code 说: '读取 ai_suggestions/latest.md 并应用建议, 然后创建 ai_suggestions/done_marker.txt'")
    print()

    deadline = time.time() + timeout_minutes * 60
    while time.time() < deadline:
        marker = _read_done_marker()
        if marker:
            print(f"[循环] Claude Code 已完成修改 (备注: {marker})")
            return True
        time.sleep(3)

    print(f"[循环] 等待超时 ({timeout_minutes} 分钟), 继续下一轮...")
    return False


def auto_loop_mode(analyzer: AIAnalyzer, max_iterations: int = 100):
    """全自动闭环模式: 自动跑马 → Qwen分析 → Claude Code修改 → 自动重跑"""
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)
    previous = set(p.name for p in _list_logs(log_path))
    iteration = 0
    project_root = Path(__file__).resolve().parent.parent

    # 读取保护参数用于启动显示
    import yaml
    config_path = project_root / "config.yaml"
    safety_display = {"max_runtime_minutes": 45, "stuck_threshold": 12}
    if config_path.exists():
        try:
            cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
            safety_display.update(cfg.get("safety", {}))
        except Exception:
            pass

    print("=" * 60)
    print("  StarSavior 全自动训练-分析-改进 流水线")
    print("=" * 60)
    print("工作流:")
    print("  1. 自动启动 Starsavior (python main.py --auto)")
    print(f"  2. 保护: 超时{safety_display['max_runtime_minutes']}分钟 / 卡死{safety_display['stuck_threshold']}轮 → 自动暂停")
    print("  3. Qwen3-8B 分析日志 → 修改建议")
    print("  4. Claude Code 读取建议 → 自动修改代码")
    print("  5. 自动重跑验证效果")
    print(f"  最多迭代: {max_iterations} 次")
    print()

    try:
        while iteration < max_iterations:
            iteration += 1
            print(f"\n{'='*60}")
            print(f"  迭代 #{iteration} / {max_iterations}")
            print(f"{'='*60}")

            # Step 1: 自动启动跑马
            print(f"\n[循环] 启动 Starsavior (第 {iteration} 轮)...")
            print(f"  命令: python main.py --auto")
            print(f"  等待中 (最长 60 分钟)...")
            print()

            bot_start = time.time()
            try:
                subprocess.run(
                    [_PYTHON_EXE, "main.py", "--auto"],
                    cwd=str(project_root),
                    timeout=3600,  # 硬超时 1 小时兜底
                )
            except subprocess.TimeoutExpired:
                print("[循环] Bot 硬超时 (1小时), 强制终止")
            bot_elapsed = time.time() - bot_start
            print(f"\n[循环] Bot 已退出 (运行 {bot_elapsed / 60:.1f} 分钟)")

            # Step 2: 等日志写完, 找最新的
            time.sleep(1)
            current = set(p.name for p in _list_logs(log_path))
            new_files = current - previous
            if not new_files:
                print("[循环] 未找到新日志, 可能是游戏未启动或窗口问题")
                print("[循环] 等待 30 秒后重试...")
                time.sleep(30)
                continue

            newest = max(new_files, key=lambda n: (log_path / n).stat().st_mtime)
            log_file = log_path / newest
            print(f"[循环] 新日志: {log_file.name}")

            # 检查退出原因
            exit_reason = "unknown"
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    log_data = json.load(f)
                exit_reason = log_data.get("exit_reason", "unknown")
                print(f"[循环] 退出原因: {exit_reason}")
                if exit_reason in ("timeout", "stuck"):
                    print(f"[循环] ⚠ 保护机制触发! 需要 AI 诊断...")
            except Exception as e:
                print(f"[循环] 读取日志失败: {e}")

            previous = set(p.name for p in _list_logs(log_path))

            # Step 3: Qwen 分析
            result = analyzer.analyze(log_file)
            if not result:
                print("[循环] 分析失败, 等待 30 秒后继续...")
                time.sleep(30)
                continue

            print(f"\n[循环] Qwen 建议摘要:")
            print("-" * 40)
            print(result[:800])
            if len(result) > 800:
                print(f"\n... (完整报告 {len(result)} 字符)")

            # Step 4: 触发 Claude Code
            _write_trigger_marker()
            print(f"\n[循环] 等待 Claude Code 自动应用修改...")

            fixed = _wait_for_claude_fix(timeout_minutes=10)

            if fixed:
                print(f"[循环] ✓ 代码已修改, 自动开始下一轮验证...")
            else:
                print(f"[循环] ⚠ 超时未检测到修改, 继续下一轮...")

    except KeyboardInterrupt:
        print(f"\n[循环] 共执行 {iteration} 次迭代, 已退出")


def loop_mode(analyzer: AIAnalyzer, log_dir: str = "logs"):
    """简化循环模式: 仅检测新日志并分析 (不自动重跑)"""
    auto_loop_mode(analyzer)


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Starsavior AI 训练-分析流水线")
    parser.add_argument(
        "log_file", nargs="?",
        help="要分析的日志文件路径 (不指定则自动找最新的)",
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="监听模式: 持续检测新日志并自动分析",
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="循环模式: 跑马 → 分析 → Claude修改 → 重跑 (全自动闭环)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=100,
        help="--loop 模式的最大迭代次数 (默认100)",
    )
    args = parser.parse_args()

    analyzer = AIAnalyzer()

    if args.watch:
        watch_mode(analyzer)
    elif args.loop:
        auto_loop_mode(analyzer, max_iterations=args.max_iterations)
    elif args.log_file:
        result = analyzer.analyze(args.log_file)
        if result:
            print("\n" + "=" * 60)
            print("Qwen3-8B 分析结果:")
            print("=" * 60)
            print(result)
        else:
            sys.exit(1)
    else:
        log_file = AIAnalyzer.find_latest_log()
        if log_file is None:
            print("[错误] 没有找到日志文件, 请先运行 Starsavior 训练。")
            sys.exit(1)
        print(f"[信息] 自动选择最新日志: {log_file.name}\n")
        result = analyzer.analyze(log_file)
        if result:
            print("\n" + "=" * 60)
            print("Qwen3-8B 分析结果:")
            print("=" * 60)
            print(result)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
