"""StarSavior automation inspector/editor.

The UI is intentionally read-only with respect to the game window.  It reads
saved screenshots, run logs, event JSON and config.yaml, and only writes the
configuration files after the user presses an explicit save button.
"""
from __future__ import annotations

import json
import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - the runtime requirements include PyYAML
    yaml = None

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover - the runtime requirements include Pillow
    Image = None
    ImageTk = None

try:
    from src.event_effects import quantify_effect_text
except Exception:  # pragma: no cover - allows opening the UI during partial installs
    def quantify_effect_text(text: str) -> dict[str, Any]:
        return {"stats": {}, "unknown_effects": [text] if text else []}


ROOT = Path(__file__).resolve().parent
EVENTS_PATH = ROOT / "profiles" / "events" / "document_verified.json"
CONFIG_PATH = ROOT / "config.yaml"
LOG_DIR = ROOT / "logs"

TRAINING_TARGETS = {
    "力量训练": (0.86, 0.28),
    "体力训练": (0.86, 0.38),
    "韧性训练": (0.86, 0.48),
    "集中训练": (0.86, 0.58),
    "保护训练": (0.86, 0.68),
}


def _compact(value: Any, limit: int = 64) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _verified_label(event: dict[str, Any]) -> str:
    if event.get("document_verified"):
        return "文档核验"
    return "本地规则" if event.get("source_type") else "未核验"


def _option_summary(option: dict[str, Any]) -> str:
    quantified = option.get("quantified") or quantify_effect_text(option.get("effect_text", ""))
    parts: list[str] = []
    for name, value in (quantified.get("stats") or {}).items():
        parts.append(f"{name}{value:+g}")
    for key, label in (("stamina_delta", "耐力"), ("potential_points", "潜质"),
                      ("mood_delta", "心情"), ("currency_delta", "硬币")):
        value = quantified.get(key, 0)
        if value:
            parts.append(f"{label}{value:+g}")
    for name in quantified.get("potential_names") or []:
        parts.append(f"潜能:{name}")
    for name in quantified.get("relic_names") or []:
        parts.append(f"遗物:{name}")
    if not parts:
        parts.extend(quantified.get("unknown_effects") or [])
    return _compact("；".join(parts) or "未量化")


class DataStore:
    def __init__(self) -> None:
        self.events_data: dict[str, Any] = {"events": []}
        self.config: dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        if EVENTS_PATH.is_file():
            self.events_data = json.loads(EVENTS_PATH.read_text(encoding="utf-8"))
        else:
            self.events_data = {"events": []}
        if yaml is not None and CONFIG_PATH.is_file():
            self.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        else:
            self.config = {}

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.events_data.setdefault("events", [])

    def save_events(self) -> None:
        self.events_data["event_count"] = len(self.events)
        EVENTS_PATH.write_text(
            json.dumps(self.events_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def save_config(self) -> None:
        if yaml is None:
            raise RuntimeError("PyYAML 未安装，无法保存 config.yaml")
        CONFIG_PATH.write_text(
            yaml.safe_dump(self.config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def logs(self) -> list[tuple[Path, dict[str, Any]]]:
        result: list[tuple[Path, dict[str, Any]]] = []
        for path in sorted(LOG_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                result.append((path, value))
        return result

    def screenshot_candidates(self) -> list[Path]:
        roots = [ROOT, ROOT / "templates", ROOT / "logs", ROOT.parent / "verification"]
        paths: list[Path] = []
        for base in roots:
            if not base.exists():
                continue
            pattern = "*.png"
            iterator = base.glob(pattern) if base in (ROOT, ROOT / "templates") else base.rglob(pattern)
            paths.extend(p for p in iterator if p.is_file())
        return sorted(set(paths), key=lambda p: p.stat().st_mtime, reverse=True)

    @staticmethod
    def resolve_screenshot(reference: Any) -> Path | None:
        """Resolve a logger path while accepting old absolute paths."""
        if not reference:
            return None
        path = Path(str(reference))
        candidates = [path] if path.is_absolute() else [ROOT / path, ROOT / "logs" / path.name]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None


class InspectorUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("StarSavior 自动化检查台")
        self.geometry("1500x920")
        self.minsize(1100, 700)
        self.store = DataStore()
        self.event_refs: dict[str, dict[str, Any]] = {}
        self.decision_refs: dict[str, dict[str, Any]] = {}
        self.selected_event: dict[str, Any] | None = None
        self.selected_option: dict[str, Any] | None = None
        self.image_ref = None
        self.weight_vars: dict[str, tk.StringVar] = {}
        self.event_filter_var = tk.StringVar()
        self.run_summary_vars = {key: tk.StringVar(value="—") for key in
                                 ("run", "actions", "training", "events")}
        self.event_name_var = tk.StringVar()
        self.event_recommended_var = tk.StringVar()
        self.option_index_var = tk.StringVar()
        self.option_keyword_var = tk.StringVar()
        self.option_alias_var = tk.StringVar()
        self.option_effect_var = tk.StringVar()
        self.status_var = tk.StringVar(value="就绪")
        self._configure_style()
        self._build()
        self.refresh_all()

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.configure(background="#f3f5f9")
        style.configure("App.TFrame", background="#f3f5f9")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("Header.TLabel", background="#f3f5f9", foreground="#172033",
                        font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("Subtitle.TLabel", background="#f3f5f9", foreground="#64748b",
                        font=("Microsoft YaHei UI", 10))
        style.configure("MetricTitle.TLabel", background="#ffffff", foreground="#64748b",
                        font=("Microsoft YaHei UI", 9))
        style.configure("MetricValue.TLabel", background="#ffffff", foreground="#172033",
                        font=("Microsoft YaHei UI", 15, "bold"))
        style.configure("Treeview", background="#ffffff", fieldbackground="#ffffff",
                        foreground="#243247", rowheight=30, font=("Microsoft YaHei UI", 10))
        style.configure("Treeview.Heading", background="#e9edf5", foreground="#334155",
                        font=("Microsoft YaHei UI", 10, "bold"), padding=(6, 6))
        style.map("Treeview", background=[("selected", "#dbeafe")],
                  foreground=[("selected", "#172033")])
        style.configure("TButton", padding=(10, 6), font=("Microsoft YaHei UI", 10))
        style.configure("Accent.TButton", background="#4f46e5", foreground="#ffffff",
                        padding=(12, 7), font=("Microsoft YaHei UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#4338ca")])
        style.configure("TNotebook", background="#f3f5f9", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 9), font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("TLabelframe", background="#ffffff", foreground="#334155")
        style.configure("TLabelframe.Label", background="#ffffff", foreground="#334155",
                        font=("Microsoft YaHei UI", 10, "bold"))

    def _build(self) -> None:
        header = ttk.Frame(self, padding=(20, 16), style="App.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="StarSavior 自动化检查台", style="Header.TLabel").pack(side="left")
        ttk.Label(header, text="查看运行记录、截图和本地决策规则", style="Subtitle.TLabel").pack(side="left", padx=18)
        ttk.Button(header, text="刷新数据", command=self.refresh_all).pack(side="right", padx=4)
        ttk.Button(header, text="加载截图…", command=self.choose_screenshot).pack(side="right", padx=4)
        ttk.Button(header, text="保存全部", command=self.save_all, style="Accent.TButton").pack(side="right", padx=4)

        ttk.Label(self, textvariable=self.status_var, background="#e8edf5", foreground="#475569",
                  anchor="w", padding=(14, 6)).pack(side="bottom", fill="x")
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=14, pady=(0, 12))

        self.run_tab = ttk.Frame(self.notebook, padding=12, style="App.TFrame")
        self.events_tab = ttk.Frame(self.notebook, padding=12, style="App.TFrame")
        self.config_tab = ttk.Frame(self.notebook, padding=12, style="App.TFrame")
        self.notebook.add(self.run_tab, text="运行")
        self.notebook.add(self.events_tab, text="事件库")
        self.notebook.add(self.config_tab, text="配置")
        self._build_run_tab()
        self._build_events_tab(self.events_tab)
        self._build_weights_tab(self.config_tab)

    def _build_run_tab(self) -> None:
        self.run_tab.columnconfigure(0, weight=1)
        self.run_tab.rowconfigure(1, weight=1)
        self.run_tab.rowconfigure(2, weight=1)

        metrics = ttk.Frame(self.run_tab, style="App.TFrame")
        metrics.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for column, (key, title) in enumerate((("run", "最近运行"), ("actions", "记录动作"),
                                                ("training", "训练次数"), ("events", "事件次数"))):
            card = ttk.Frame(metrics, padding=(14, 10), style="Card.TFrame")
            card.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 8, 0))
            metrics.columnconfigure(column, weight=1)
            ttk.Label(card, text=title, style="MetricTitle.TLabel").pack(anchor="w")
            ttk.Label(card, textvariable=self.run_summary_vars[key], style="MetricValue.TLabel").pack(anchor="w", pady=(2, 0))

        split = ttk.PanedWindow(self.run_tab, orient="horizontal")
        split.grid(row=1, column=0, sticky="nsew")
        decisions = ttk.LabelFrame(split, text="历史决策（点击记录查看对应截图）", padding=8)
        screenshot = ttk.LabelFrame(split, text="决策时截图", padding=8)
        split.add(decisions, weight=5)
        split.add(screenshot, weight=6)

        columns = ("time", "type", "decision", "target", "shot")
        self.decision_tree = ttk.Treeview(decisions, columns=columns, show="headings", height=8)
        for col, title, width in (("time", "时间", 82), ("type", "类型", 78),
                                  ("decision", "决策", 145), ("target", "点击目标", 145),
                                  ("shot", "截图", 58)):
            self.decision_tree.heading(col, text=title)
            self.decision_tree.column(col, width=width, anchor="w")
        self.decision_tree.pack(side="left", fill="both", expand=True)
        decision_scroll = ttk.Scrollbar(decisions, orient="vertical", command=self.decision_tree.yview)
        decision_scroll.pack(side="right", fill="y")
        self.decision_tree.configure(yscrollcommand=decision_scroll.set)
        self.decision_tree.bind("<<TreeviewSelect>>", self.on_decision_selected)

        self.shot_label = ttk.Label(screenshot, text="选择一条历史决策查看截图", anchor="center")
        self.shot_label.pack(fill="both", expand=True)
        self.shot_path_var = tk.StringVar(value="")
        ttk.Label(screenshot, textvariable=self.shot_path_var, wraplength=680,
                  foreground="#64748b").pack(fill="x", pady=(8, 0))

        log_box = ttk.LabelFrame(self.run_tab, text="运行日志（与本次运行同一页）", padding=8)
        log_box.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        self.log_text = tk.Text(log_box, height=9, wrap="word", state="disabled")
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(log_box, orient="vertical", command=self.log_text.yview)
        scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scroll.set)

    def _build_events_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        parent.rowconfigure(2, weight=2)
        toolbar = ttk.Frame(parent, style="App.TFrame")
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(toolbar, text="事件 / 分支 / 量化收益", font=("Microsoft YaHei UI", 13, "bold")).pack(side="left")
        ttk.Label(toolbar, text="搜索").pack(side="left", padx=(24, 6))
        search = ttk.Entry(toolbar, textvariable=self.event_filter_var, width=28)
        search.pack(side="left")
        self.event_filter_var.trace_add("write", lambda *_: self.populate_events())
        ttk.Button(toolbar, text="清除", command=lambda: self.event_filter_var.set("")).pack(side="left", padx=5)

        top = ttk.LabelFrame(parent, text="事件索引", padding=8)
        top.grid(row=1, column=0, sticky="nsew")
        top.columnconfigure(0, weight=1)
        top.rowconfigure(0, weight=1)
        event_columns = ("name", "category", "source", "verified", "recommended", "options")
        self.event_tree = ttk.Treeview(top, columns=event_columns, show="headings", height=14)
        specs = (("name", "事件", 190), ("category", "类别", 155), ("source", "来源", 90),
                 ("verified", "状态", 80), ("recommended", "推荐", 60), ("options", "分支数", 60))
        for col, title, width in specs:
            self.event_tree.heading(col, text=title)
            self.event_tree.column(col, width=width, anchor="w")
        self.event_tree.grid(row=0, column=0, sticky="nsew")
        event_scroll = ttk.Scrollbar(top, orient="vertical", command=self.event_tree.yview)
        event_scroll.grid(row=0, column=1, sticky="ns")
        self.event_tree.configure(yscrollcommand=event_scroll.set)
        self.event_tree.bind("<<TreeviewSelect>>", self.on_event_selected)

        editor = ttk.LabelFrame(parent, text="结构化编辑", padding=10)
        editor.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        editor.columnconfigure(0, weight=1)
        editor.rowconfigure(1, weight=1)
        meta = ttk.Frame(editor)
        meta.pack(fill="x")
        ttk.Label(meta, text="事件名").grid(row=0, column=0, sticky="w")
        ttk.Entry(meta, textvariable=self.event_name_var, width=32).grid(row=0, column=1, padx=6, sticky="ew")
        ttk.Label(meta, text="推荐分支").grid(row=0, column=2, sticky="w")
        ttk.Entry(meta, textvariable=self.event_recommended_var, width=8).grid(row=0, column=3, padx=6)
        ttk.Button(meta, text="保存事件", command=self.save_event).grid(row=0, column=4, padx=6)
        meta.columnconfigure(1, weight=1)

        branch_columns = ("index", "keyword", "effect", "quantified")
        self.option_tree = ttk.Treeview(editor, columns=branch_columns, show="headings", height=8)
        for col, title, width in (("index", "序号", 45), ("keyword", "分支文本/关键词", 220),
                                  ("effect", "收益原文", 280), ("quantified", "量化收益", 220)):
            self.option_tree.heading(col, text=title)
            self.option_tree.column(col, width=width, anchor="w")
        self.option_tree.pack(fill="both", expand=True, pady=(8, 6))
        self.option_tree.bind("<<TreeviewSelect>>", self.on_option_selected)

        option_form = ttk.Frame(editor)
        option_form.pack(fill="x")
        labels = (("序号", self.option_index_var), ("关键词", self.option_keyword_var),
                  ("别名（逗号分隔）", self.option_alias_var), ("收益原文", self.option_effect_var))
        for row, (label, var) in enumerate(labels):
            ttk.Label(option_form, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(option_form, textvariable=var).grid(row=row, column=1, sticky="ew", pady=2)
        option_form.columnconfigure(1, weight=1)
        buttons = ttk.Frame(editor)
        buttons.pack(fill="x", pady=(6, 0))
        ttk.Button(buttons, text="更新分支", command=self.update_option).pack(side="left")
        ttk.Button(buttons, text="新增分支", command=self.add_option).pack(side="left", padx=6)
        ttk.Button(buttons, text="删除分支", command=self.delete_option).pack(side="left")

    def _build_weights_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="编辑后点击“保存权重”写入 config.yaml；数值仅影响后续脚本运行。",
                  wraplength=700).pack(anchor="w", pady=(0, 12))
        self.weights_frame = ttk.Frame(parent)
        self.weights_frame.pack(fill="both", expand=True)
        ttk.Button(parent, text="保存权重", command=self.save_weights).pack(anchor="w", pady=10)

    def refresh_all(self) -> None:
        self.store.reload()
        self.populate_events()
        self.populate_weights()
        self.populate_logs()
        candidates = self.store.screenshot_candidates()
        if candidates and not self.decision_tree.selection():
            self.show_screenshot(candidates[0])
        self.status_var.set(f"已加载 {len(self.store.events)} 个事件；日志 {len(self.store.logs())} 个；点击运行记录可回看截图")

    def populate_events(self) -> None:
        for item in self.event_tree.get_children():
            self.event_tree.delete(item)
        self.event_refs.clear()
        query = self.event_filter_var.get().strip().lower()
        visible_index = 0
        for event in self.store.events:
            searchable = " ".join(str(event.get(key, "")) for key in
                                    ("event_name", "id", "category", "source_sheet", "source_type"))
            if query and query not in searchable.lower():
                continue
            iid = str(visible_index)
            visible_index += 1
            self.event_refs[iid] = event
            self.event_tree.insert("", "end", iid=iid, values=(
                _compact(event.get("event_name") or event.get("id"), 28),
                _compact(event.get("category"), 22),
                _compact(event.get("source_sheet") or event.get("source_type") or "—", 16),
                _verified_label(event),
                event.get("recommended_option", "—"),
                len(event.get("options") or []),
            ))

    def on_event_selected(self, _event=None) -> None:
        selection = self.event_tree.selection()
        if not selection:
            return
        self.selected_event = self.event_refs.get(selection[0])
        if not self.selected_event:
            return
        self.event_name_var.set(self.selected_event.get("event_name", ""))
        self.event_recommended_var.set(str(self.selected_event.get("recommended_option", "")))
        self.populate_options()

    def populate_options(self) -> None:
        for item in self.option_tree.get_children():
            self.option_tree.delete(item)
        if not self.selected_event:
            return
        for row, option in enumerate(self.selected_event.get("options") or []):
            iid = str(row)
            self.option_tree.insert("", "end", iid=iid, values=(
                option.get("index", row + 1), option.get("keyword", ""),
                _compact(option.get("effect_text", ""), 40), _option_summary(option),
            ))

    def on_option_selected(self, _event=None) -> None:
        selection = self.option_tree.selection()
        if not selection or not self.selected_event:
            return
        try:
            option = self.selected_event["options"][int(selection[0])]
        except (KeyError, IndexError, ValueError):
            return
        self.selected_option = option
        self.option_index_var.set(str(option.get("index", "")))
        self.option_keyword_var.set(option.get("keyword", ""))
        self.option_alias_var.set(",".join(option.get("alias") or []))
        self.option_effect_var.set(option.get("effect_text", ""))

    def save_event(self) -> None:
        if not self.selected_event:
            messagebox.showinfo("提示", "请先选择一个事件")
            return
        try:
            self.selected_event["event_name"] = self.event_name_var.get().strip()
            self.selected_event["recommended_option"] = int(self.event_recommended_var.get())
        except ValueError:
            messagebox.showerror("保存失败", "推荐分支必须是整数")
            return
        self.store.save_events()
        self.populate_events()
        self.status_var.set("事件已保存")

    def update_option(self) -> None:
        if not self.selected_event or not self.selected_option:
            messagebox.showinfo("提示", "请先选择一个分支")
            return
        try:
            self.selected_option["index"] = int(self.option_index_var.get())
        except ValueError:
            messagebox.showerror("保存失败", "分支序号必须是整数")
            return
        self.selected_option["keyword"] = self.option_keyword_var.get().strip()
        self.selected_option["alias"] = [x.strip() for x in self.option_alias_var.get().split(",") if x.strip()]
        self.selected_option["effect_text"] = self.option_effect_var.get().strip()
        self.store.save_events()
        self.populate_options()
        self.status_var.set("分支已更新；量化收益将在刷新后重算")

    def add_option(self) -> None:
        if not self.selected_event:
            messagebox.showinfo("提示", "请先选择一个事件")
            return
        options = self.selected_event.setdefault("options", [])
        next_index = max((int(x.get("index", 0)) for x in options), default=0) + 1
        options.append({"index": next_index, "keyword": "新分支", "alias": [], "effect_text": ""})
        self.store.save_events()
        self.populate_options()
        self.status_var.set("已新增分支")

    def delete_option(self) -> None:
        if not self.selected_event:
            return
        selection = self.option_tree.selection()
        if not selection:
            return
        del self.selected_event["options"][int(selection[0])]
        self.store.save_events()
        self.populate_options()
        self.status_var.set("已删除分支")

    def _weight_specs(self) -> list[tuple[str, str, Any]]:
        training = self.store.config.setdefault("training", {})
        weights = training.setdefault("weights", {})
        events = self.store.config.setdefault("events", {})
        direction_weights = events.setdefault("decision_weights", {})
        defaults = {
            "attack": {"力量": 5.0, "体力": 4.0, "韧性": 4.0, "专注": 3.0, "保护": 2.0},
            "survival": {"韧性": 5.0, "体力": 4.0, "保护": 4.0, "力量": 3.0, "专注": 2.0},
        }
        specs: list[tuple[str, str, Any]] = []
        for key, label in (("priority_bonus", "训练：优先属性加分"),
                           ("gain_score", "训练：增益权重"),
                           ("failure_penalty", "训练：失败率惩罚")):
            specs.append((f"training.weights.{key}", label, weights.get(key, 0)))
        for key, label in (("failure_threshold", "训练：失败率阈值"),
                           ("adventure_gain_threshold", "训练：冒险增益阈值"),
                           ("advantage_multiplier", "训练：优势训练倍率")):
            specs.append((f"training.{key}", label, training.get(key, 0)))
        for direction, values in defaults.items():
            current = direction_weights.setdefault(direction, {})
            for stat, default in values.items():
                specs.append((f"events.decision_weights.{direction}.{stat}",
                              f"事件：{direction}向 {stat}", current.get(stat, default)))
        return specs

    def populate_weights(self) -> None:
        for child in self.weights_frame.winfo_children():
            child.destroy()
        self.weight_vars.clear()
        for row, (path, label, value) in enumerate(self._weight_specs()):
            ttk.Label(self.weights_frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
            var = tk.StringVar(value=str(value))
            self.weight_vars[path] = var
            ttk.Entry(self.weights_frame, textvariable=var, width=16).grid(row=row, column=1, sticky="w", padx=12, pady=4)
        self.weights_frame.columnconfigure(2, weight=1)

    @staticmethod
    def _set_path(root: dict[str, Any], path: str, value: Any) -> None:
        parts = path.split(".")
        node = root
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def save_weights(self) -> None:
        try:
            for path, var in self.weight_vars.items():
                raw = var.get().strip()
                value: int | float = float(raw) if "." in raw else int(raw)
                self._set_path(self.store.config, path, value)
            self.store.save_config()
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("保存失败", str(exc))
            return
        self.status_var.set("决策权重已保存到 config.yaml")

    def populate_logs(self) -> None:
        for item in self.decision_tree.get_children():
            self.decision_tree.delete(item)
        self.decision_refs.clear()
        logs = self.store.logs()
        latest_text = ""
        for path, data in logs[:3]:
            latest_text += f"\n===== {path.name} =====\n"
            latest_text += json.dumps(data, ensure_ascii=False, indent=2)
            latest_text += "\n"
        latest = logs[0][1] if logs else {}
        summary = latest.get("summary") or {}
        self.run_summary_vars["run"].set(logs[0][0].stem.replace("run_", "") if logs else "暂无记录")
        self.run_summary_vars["actions"].set(str(summary.get("total_actions", "—")))
        self.run_summary_vars["training"].set(str(summary.get("train_count", "—")))
        self.run_summary_vars["events"].set(str(summary.get("events_encountered", "—")))

        for _, data in logs[:3]:
            for action in data.get("actions", [])[-20:]:
                self._insert_action(action)
            for turn in data.get("turns", []):
                for action in turn.get("actions", []):
                    if action.get("type") in {"decision", "train", "rest", "event", "arcanum", "card_select", "shop"}:
                        self._insert_action(action)
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("1.0", latest_text or "暂无日志；运行脚本后点击刷新数据。")
        self.log_text.configure(state="disabled")

    def _insert_action(self, action: dict[str, Any]) -> None:
        action_type = action.get("type") or action.get("name") or "action"
        decision = action.get("attribute") or action.get("action") or action.get("event_name") or action.get("name") or "—"
        target = self._target_text(action)
        timestamp = str(action.get("ts") or action.get("timestamp") or "")[-8:]
        iid = str(len(self.decision_refs))
        self.decision_refs[iid] = action
        shot = "有" if self.store.resolve_screenshot(action.get("screenshot")) else "—"
        self.decision_tree.insert("", "end", iid=iid,
                                  values=(timestamp, action_type, _compact(decision, 22), target, shot))

    def on_decision_selected(self, _event=None) -> None:
        selection = self.decision_tree.selection()
        if not selection:
            return
        action = self.decision_refs.get(selection[0])
        if not action:
            return
        path = self.store.resolve_screenshot(action.get("screenshot"))
        if path is None:
            self.shot_label.configure(image="", text="这条历史记录没有保存对应截图\n（旧日志或截图已被清理）")
            self.shot_path_var.set("")
            self.status_var.set("该记录没有可用截图")
            return
        self.show_screenshot(path)
        self.status_var.set(f"已显示决策截图：{path.name}")

    @staticmethod
    def _target_text(action: dict[str, Any]) -> str:
        if "target_x" in action and "target_y" in action:
            return f"客户区 ({float(action['target_x']):.3f}, {float(action['target_y']):.3f})"
        row = action.get("row_index", action.get("target_row"))
        if row is not None:
            names = list(TRAINING_TARGETS.items())
            try:
                name, (x, y) = names[int(row)]
                return f"{name} ({x:.3f}, {y:.3f})"
            except (IndexError, ValueError):
                pass
        if action.get("option_selected") is not None:
            option = int(action["option_selected"])
            count = int(action.get("num_options") or 3)
            centers = {2: [0.65, 0.74], 3: [0.57, 0.65, 0.73], 4: [0.52, 0.59, 0.66, 0.73]}
            y = (centers.get(count) or centers[3])[max(0, min(option - 1, len((centers.get(count) or centers[3])) - 1))]
            return f"事件选项 ({0.800:.3f}, {y:.3f})"
        return "—"

    def choose_screenshot(self) -> None:
        path = filedialog.askopenfilename(
            title="选择截图", initialdir=str(ROOT), filetypes=[("PNG 图片", "*.png"), ("所有文件", "*.*")]
        )
        if path:
            self.show_screenshot(Path(path))

    def show_screenshot(self, path: Path) -> None:
        if Image is None or ImageTk is None:
            self.shot_label.configure(text=f"Pillow 未安装\n{path}")
            return
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail((680, 420))
            self.image_ref = ImageTk.PhotoImage(image)
            self.shot_label.configure(image=self.image_ref, text="")
            self.shot_path_var.set(str(path))
        except (OSError, ValueError) as exc:
            self.shot_label.configure(image="", text=f"无法读取截图: {exc}")

    def save_all(self) -> None:
        if self.selected_event:
            self.save_event()
        self.save_weights()
        self.status_var.set("事件与权重均已保存")


def main() -> None:
    app = InspectorUI()
    app.mainloop()


if __name__ == "__main__":
    main()
