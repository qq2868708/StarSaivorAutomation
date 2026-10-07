#!/usr/bin/env python
"""
运行后耗时分析工具。

读取日志 JSON 中的 timing 数据, 输出:
  - 总耗时概览
  - 各 Handler 耗时分布 (平均/最大/总耗时)
  - 最慢的 N 个回合 (含 handler 链路)
  - 瓶颈诊断和建议

用法:
  python scripts/analyze_timing.py logs/run_20260527_025301.json
  python scripts/analyze_timing.py                    # 自动找最新日志
"""

import json
import sys
from pathlib import Path
from typing import Optional


def find_latest_log(log_dir: str = "logs") -> Optional[Path]:
    log_path = Path(log_dir)
    if not log_path.is_dir():
        return None
    logs = sorted(log_path.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def analyze(log_path: str) -> dict:
    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    timing = data.get("timing", {})
    turns = data.get("turns", [])
    summary = data.get("summary", {})

    print("=" * 60)
    print("  StarSavior 运行耗时分析")
    print("=" * 60)
    print(f"日志: {Path(log_path).name}")
    print(f"退出原因: {data.get('exit_reason', '?')}")
    print(f"总回合: {summary.get('total_turns', 0)}, "
          f"总动作: {summary.get('total_actions', 0)}")
    print(f"训练: {summary.get('train_count', 0)}, "
          f"休息: {summary.get('rest_count', 0)}, "
          f"商店: {summary.get('shop_visits', 0)}, "
          f"事件: {summary.get('events_encountered', 0)}")
    print()

    # 1. 总耗时
    total = timing.get("total_seconds", data.get("duration_seconds", 0))
    print(f"[总耗时] {total:.1f}s ({total / 60:.1f}min)")
    print()

    # 2. Handler 耗时分布
    handlers = timing.get("handlers", {})
    breakdown = timing.get("breakdown", {})
    handler_pct = breakdown.get("handler_pct", 0)
    if handlers:
        print("--- Handler 耗时分布 ---")
        print(f"{'Handler':<18} {'次数':>5} {'总耗时':>8} {'平均':>8} {'最大':>8}")
        print("-" * 50)
        sorted_handlers = sorted(handlers.items(),
                                 key=lambda x: x[1]["total_seconds"], reverse=True)
        for name, stats in sorted_handlers:
            print(f"{name:<18} {stats['count']:>5} {stats['total_seconds']:>7.1f}s "
                  f"{stats['avg_ms']:>7.0f}ms {stats['max_ms']:>7.0f}ms")
        print()

        # 瓶颈分析
        print(f"[耗时构成] Handler: {handler_pct}%, 其他(截图/OCR/等待): {100 - handler_pct}%")
        print()

        # 找出耗时占比最高的 handler
        if sorted_handlers:
            top = sorted_handlers[0]
            if top[1]["total_seconds"] > total * 0.2:
                print(f"[瓶颈] {top[0]} 占 {top[1]['total_seconds'] / total * 100:.0f}% 总耗时")
                if top[1]["avg_ms"] > 5000:
                    print(f"  -> 平均每次 {top[1]['avg_ms']:.0f}ms, 检查是否有等待过长或重复操作")
                if top[1]["max_ms"] > 15000:
                    print(f"  -> 最慢一次 {top[1]['max_ms']:.0f}ms, 可能有卡顿")
                print()

    # 3. 最慢回合
    slowest = timing.get("slowest_turns", [])
    if slowest:
        print("--- 最慢回合 (Top 5) ---")
        for t in slowest:
            r = t["round"]
            dur = t["duration_seconds"]
            # 找对应回合的 handler chain
            turn_data = None
            for turn in turns:
                if turn.get("round") == r:
                    turn_data = turn
                    break
            chain_desc = ""
            if turn_data:
                handlers_in_turn = [a["name"] for a in turn_data.get("actions", [])
                                    if a.get("type") == "handler"]
                if handlers_in_turn:
                    chain_desc = " → ".join(handlers_in_turn[-5:])  # 最后5个
            print(f"  第{r}轮: {dur:.1f}s  {chain_desc}")
        print()

    # 4. 建议
    print("--- 优化建议 ---")
    suggestions = []

    # 检查是否有超过30秒的轮次
    for t in slowest:
        if t["duration_seconds"] > 30:
            suggestions.append(f"第{t['round']}轮耗时{t['duration_seconds']:.0f}s, 检查是否卡死在某个handler")

    # 检查 handler 超时
    sorted_handlers = sorted(handlers.items(),
                             key=lambda x: x[1]["total_seconds"], reverse=True) if handlers else []
    for name, stats in sorted_handlers:
        if stats["max_ms"] > 20000:
            suggestions.append(f"{name} 最慢{stats['max_ms']:.0f}ms, 建议检查对应handler的超时/重试逻辑")
        if stats["avg_ms"] > 3000 and stats["count"] > 3:
            suggestions.append(f"{name} 平均{stats['avg_ms']:.0f}ms(共{stats['count']}次), "
                               f"考虑减少等待时间或优化OCR区域")

    # 检查回合均耗时
    per_turn = timing.get("per_turn", {})
    if per_turn.get("avg_seconds", 0) > 10:
        suggestions.append(f"平均每轮{per_turn['avg_seconds']:.1f}s, 偏慢. "
                           f"检查adaptive_wait或screenshot等待时间")

    # 检查等待占比
    if handler_pct < 40:
        suggestions.append(f"Handler仅占{handler_pct}%, {100 - handler_pct}%时间在截图/等待, "
                           f"考虑降低未命中等待间隔")

    if suggestions:
        for i, s in enumerate(suggestions, 1):
            print(f"  {i}. {s}")
    else:
        print("  未发现明显瓶颈, 耗时正常.")

    print()
    return data


if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        latest = find_latest_log()
        if latest is None:
            print("[错误] 没有找到日志文件")
            sys.exit(1)
        log_file = str(latest)
        print(f"[信息] 自动选择最新日志: {Path(log_file).name}\n")

    analyze(log_file)
