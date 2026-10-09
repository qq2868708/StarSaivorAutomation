# 脚本运行、恢复与 AI 接管协议

自动识别、策略选择、输入、重试、恢复由 Python 脚本执行。运行时不调用模型服务，也不需要 Codex 保持运行。AI 只在脚本已经退出当前操作流程并安全暂停后，作为可选的故障处理端接入。

## 状态

`starting → recovering → running` 为正常启动；`running → paused_manual` 为人工暂停。
识别持续失败、输入失焦、异常或超时进入 `paused_safe`。继续会进入 `recovering`，由脚本重新绑定游戏窗口、截取当前画面、清理过期的 Handler 缓存并重新调度。不会继续执行暂停前的旧点击序列。

没有 checkpoint 也可以从当前已支持的旅程页面启动。有 checkpoint 时，它只提供上次动作和业务状态作为参考：实际游戏画面优先；页面变化后不恢复旧回合猜测或待执行输入。回合重新读取游戏中的 X/45。商店阶段缓存仅在配置和画面都相同时复用。

`journey_end` 为整局旅程结束，交还人工处理潜质和奖励；`stopped` 保留可恢复记录。重复运行实例由操作系统文件锁阻止，即使命令行和 WebUI 同时启动。

## 人工控制

仅监听 `127.0.0.1:8765`。

| API | 用途 |
| --- | --- |
| `GET /api/runtime` | 进程存活、运行状态、最近 checkpoint |
| `POST /api/runtime/start` | 启动脚本，默认恢复；已有实例时返回该实例 |
| `POST /api/runtime/pause` | 请求暂停，在下次截图或输入前退出操作流程 |
| `POST /api/runtime/resume` | 暂停后重新识别当前界面 |
| `POST /api/runtime/step` | 执行一个页面处理流程后再次暂停 |
| `POST /api/runtime/stop` | 请求停止并保存状态 |
| `POST /api/runtime/restart` | 等待停止后重新启动；旧实例未退出则返回 409 |
| `GET /api/runtime/diagnostics` | 安全暂停原因、OCR、截图 URL、最后输入和 Handler |
| `GET /api/runtime/commands/{id}` | 查询命令是否已被脚本执行或拒绝 |

控制请求返回队列命令 ID 不表示动作已经执行。可查询命令结果或等待运行状态改变。所有命令绑定 `run_id`，旧实例的命令不能操作新实例。

## AI 接管

1. 读取 diagnostics，确认 `safe_for_ai=true` 且 `state=paused_safe`。
2. 用状态中的 `run_id` 和 `pause_id` 注册客户端，得到 `lease_id`。
3. 读取诊断中的 `frame_id` 和截图，提交结构化动作。
4. 查询命令结果；点击后脚本仍保持安全暂停，并更新现场截图、OCR 和 `frame_id`。
5. 处理完后请求 `resume`，由脚本重新识别页面，交还自动运行。

注册示例：

```json
POST /api/runtime/ai-claim
{
  "client_id": "codex-diagnostics",
  "run_id": "来自status.run_id",
  "pause_id": "来自status.pause_id"
}
```

同一个客户端再次注册会续期；会话有效期为 120 秒。暂停重新发生、脚本恢复、实例重启或会话到期，都要求重新核对状态。已存在其他有效客户端时拒绝接管。

一次点击示例：

```json
POST /api/runtime/ai-action
{
  "kind": "click",
  "run_id": "来自status.run_id",
  "pause_id": "来自status.pause_id",
  "lease_id": "来自ai-claim.lease_id",
  "action_id": "客户端生成的唯一动作ID",
  "expected_frame_id": "来自status.frame_id",
  "x": 0.8,
  "y": 0.65
}
```

`x/y` 是游戏客户区比例坐标，范围为 0..1。脚本执行前重新截图；画面变化过大或诊断帧已更新时拒绝点击。该校验是缩小截图的平均像素差比较，不能替代 OCR 语义核对。点击仍遵守前台窗口和输入权限检查。

`kind` 也可以是 `inspect`、`resume`、`stop`，它们不需要 x/y。`inspect` 重新获取现场，不点击。重复 action_id 不会再次发送输入；异常后的客户端应先检查现场和命令结果，不重放该 ID。协议不支持运行任意 Python 代码或注入游戏进程。

## 恢复记录与验证范围

`runtime/checkpoint.json` 在每次鼠标或键盘输入前、输入后、决策后及页面处理后原子写入。
`last_input.phase=prepared` 表示输入尚未确认；`sent` 表示系统已接收输入，并不保证游戏接受或完成动作；`uncertain` 表示输入期间出现异常。恢复一律从新截图判断，不能按 checkpoint 重放。

输入异常时仍会尝试松开按键、恢复光标并记录 `uncertain`。如果系统拒绝释放输入，脚本无法保证释放成功，会停止后续操作；截图或 checkpoint 无法写入时也不会发送下一次输入。

`runtime/events.jsonl` 是追加事件流；`runtime/screenshots/` 保存现场；`runtime/commands/` 保存待处理命令，`runtime/acks/` 保存执行结果。这些都是本地运行数据，不纳入 Git。

离线回归覆盖：暂停中断旧流程、当前页面重新调度、输入前后记录、实例互斥、未知事件保持暂停、AI 会话与画面校验、重复动作拒绝、正常运行拒绝 AI。真实游戏所有页面的重启恢复仍须逐页实测；新增界面需要扩充脚本的 Handler，不能依赖 AI 成为正常流程的一部分。
