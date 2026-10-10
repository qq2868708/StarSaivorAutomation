# 环境与启动

## 固定环境

2026-10-10 在 Windows 11 x64 上恢复并实测：

| 项目 | 版本 |
| --- | --- |
| Python | 3.12.14 |
| RapidOCR / ONNX Runtime | 1.4.4 / 1.23.2 |
| NumPy / OpenCV | 1.26.4 / 4.10.0.84 |
| Pillow / PyYAML | 11.3.0 / 6.0.2 |
| FastAPI / Uvicorn | 0.142.2 / 0.54.0 |

完整依赖采用 `requirements-lock.txt`。`requirements.txt` 是运行依赖入口，
`requirements-legacy.txt` 仅留档，不安装旧 PaddleOCR。

当前项目需要 64 位 Python 3.11/3.12。不能为了兼容 Python 3.13 把
RapidOCR 降到 1.2.3：该版本的文本分数计算额外加入近零值，两个字的分数上限
约为 0.667，无法通过原有 0.72 页面识别门槛。1.4.4 使用正常的字符平均分。

## 双击启动

`启动面板.bat` 调用 `tools/bootstrap_ui.py`：

1. 优先复用 `.venv`，不存在时使用 Python 3.12/3.11 建立独立环境。
2. 检查锁定依赖，仅缺失、版本不符或传入 `--repair` 时运行安装。
3. 执行 `pip check`，导入 WebUI，并在内存合成“力量”图片，验证真正的 OCR
   推理及短文本分数；不是只做 `import`。这一步不截图游戏、不发送输入。
4. 检查通过后启动或复用面板，默认打开浏览器。失败保留窗口。

命令行选项：

```powershell
.\启动面板.bat --check-only
.\启动面板.bat --repair --check-only
.\启动面板.bat --no-browser
```

若 Python 未注册到 `py`，首次建环境时可通过本地环境变量
`STARSAVIOR_PYTHON` 指定解释器；路径不写入仓库配置。
虚拟环境仍依赖创建它的基础解释器，因此不能移走或删除基础 Python。
启动器不自动安装系统 Python、不删除旧 `.venv`，也不修改正在运行面板的依赖。

## 本地证据

- `runtime/bootstrap.log`：当前初始化日志，保留一份 `bootstrap.previous.log`。
  日志对用户名目录、项目路径、URL 与常见凭据字段脱敏。
- `runtime/environment.json`：面板启动时的版本快照。
- `runtime/environment-before-rollback.json`：本次回退前的快照。
- `runtime/venv-backup-py313/`：旧环境备份，未删除；恢复时需放回原 `.venv` 路径。
- `runtime/live_lookup_validation.json`：本次实机验收结果。
- `runtime/quick_lookup_rescuer.jsonl`：生产扫描接口的阶段日志。

这些文件均被 Git 忽略，不要将虚拟环境、完整游戏截图或私人配置提交到仓库。

## 验收范围

通过生产 WebUI 的“补充当前角色”接口，实机分别采集“克拉丽莎”和“埃米莉”，
两轮均零重试、状态 `completed`。均核验了：

`initial_info_collected → closing_initial → returning_to_list → returned_to_list → finished`

完成后再次截图识别，确认回到救援者列表且选中角色正确。没有发送训练输入。

上述两角色验证不是全量列表验收：测试中另发现一个生僻名字在不同区域被识别成
“亚瑟拉”/“亚瑟荘”。该差异仍会触发身份保护，不使用宽松替换或错配数据来掩盖它。
阿尔克那全量遍历也不在本次返回流程验收范围内。

随后完成了整份 44 行列表的生产接口实机验证：43 成功、1 失败，通过率 97.73%，
达到 90% 门槛，最终返回列表。完整口径和证据见 [完整列表验收](full-roster-validation.md)。
