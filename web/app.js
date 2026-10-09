const state = {
  overview: null,
  events: [],
  totalEvents: 0,
  currentEventIndex: null,
  currentEvent: null,
  selectedBranch: null,
  isNewEvent: false,
  activeTab: "run",
  quickLookup: null,
};

const $ = (id) => document.getElementById(id);

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function api(url, options = {}) {
  const response = await fetch(url, { cache: "no-store", ...options });
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch (_) { /* plain response */ }
    throw new Error(detail);
  }
  return response.json();
}

function showToast(message, error = false) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.toggle("error", error);
  toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("show"), 2600);
}

function setTab(tab) {
  state.activeTab = tab;
  document.querySelectorAll("[data-run-only]").forEach((element) => {
    element.hidden = tab !== "run";
  });
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tab);
  });
  document.querySelectorAll(".tab-page").forEach((page) => page.classList.remove("active"));
  $("tab-" + tab).classList.add("active");
  const titles = {
    run: ["运行", "查看决策轨迹、点击目标和对应截图"],
    events: ["事件库", "浏览所有事件、分支和量化收益"],
    config: ["配置", "调整规则权重并保存到本地配置"],
  };
  $("pageTitle").textContent = titles[tab][0];
  $("pageSubtitle").textContent = titles[tab][1];
  if (tab === "events" && !$('eventRows').dataset.loaded) loadEvents();
  if (tab === "config" && !$('configGrid').dataset.loaded) loadConfig();
}

function formatTime(value) {
  if (!value) return "—";
  const text = String(value);
  return text.includes("T") ? text.replace("T", " ").slice(0, 19) : text;
}

function renderOverview(data) {
  state.overview = data;
  const summary = data.summary || {};
  const run = data.run || {};
  $("metric-run").textContent = run.id || "暂无记录";
  $("metric-actions").textContent = summary.total_actions ?? "—";
  $("metric-training").textContent = summary.train_count ?? "—";
  $("metric-events").textContent = summary.events_encountered ?? "—";
  $("metric-exit").textContent = run.exit_reason ? `结束：${run.exit_reason}` : "等待日志";
  $("decisionCount").textContent = (data.actions || []).length;
  $("updatedLabel").textContent = `更新于 ${new Date().toLocaleTimeString()}`;
  $("healthDot").style.background = "#50d69b";

  const actions = data.actions || [];
  const rows = $("decisionRows");
  if (!actions.length) {
    rows.innerHTML = '<tr><td colspan="5" class="empty-cell">暂无决策记录，运行脚本后点击刷新</td></tr>';
  } else {
    rows.innerHTML = actions.map((action, index) => `
      <tr data-action-index="${index}">
        <td>${esc(action.time)}</td>
        <td class="type-cell">${esc(action.type)}</td>
        <td title="${esc(action.decision)}">${esc(action.decision)}</td>
        <td title="${esc(action.target)}">${esc(action.target)}</td>
        <td class="${action.has_screenshot ? "shot-yes" : ""}">${action.has_screenshot ? "有" : "—"}</td>
      </tr>`).join("");
    rows.querySelectorAll("tr[data-action-index]").forEach((row) => {
      row.addEventListener("click", () => selectAction(Number(row.dataset.actionIndex), row));
    });
  }

  const logLines = actions.slice(0, 80).map((action) => {
    const detail = action.details || {};
    const extra = action.type === "decision" && detail.matched_rule ? ` rule=${detail.matched_rule}` : "";
    return `[${action.time || "--:--:--"}] ${action.type} · ${action.decision} · ${action.target}${extra}`;
  });
  $("logView").textContent = logLines.join("\n") || "暂无日志；运行脚本后点击刷新数据。";

  const runs = $("runList");
  if (!(data.logs || []).length) {
    runs.innerHTML = '<div class="empty-cell">暂无运行记录</div>';
  } else {
    runs.innerHTML = data.logs.map((item) => `
      <div class="run-item"><div><strong>${esc(item.run_id || item.name)}</strong>
        <small>${esc(formatTime(item.started_at))}</small></div>
        <span class="run-status">${esc(item.exit_reason || "未标记")}</span></div>`).join("");
  }
}

function selectAction(index, row) {
  document.querySelectorAll("#decisionRows tr").forEach((item) => item.classList.remove("selected"));
  if (row) row.classList.add("selected");
  const action = (state.overview?.actions || [])[index];
  if (!action) return;
  const image = $("decisionImage");
  const placeholder = $("screenshotPlaceholder");
  if (action.screenshot_url) {
    image.src = action.screenshot_url;
    image.hidden = false;
    placeholder.hidden = true;
    $("screenshotBadge").textContent = "已关联";
    $("screenshotBadge").classList.add("ready");
    $("screenshotCaption").textContent = `${action.type} · ${action.decision}`;
    $("screenshotPath").textContent = action.screenshot || "—";
  } else {
    image.hidden = true;
    image.removeAttribute("src");
    placeholder.hidden = false;
    $("screenshotBadge").textContent = "无截图";
    $("screenshotBadge").classList.remove("ready");
    $("screenshotCaption").textContent = "旧日志没有保存对应截图";
    $("screenshotPath").textContent = "—";
  }
}

async function loadOverview() {
  try {
    renderOverview(await api("/api/overview"));
    await loadRuntimeStatus();
  } catch (error) {
    $("healthDot").style.background = "#dc5b68";
    showToast(`读取运行数据失败：${error.message}`, true);
  }
}

function renderRuntimeStatus(status) {
  const running = Boolean(status?.running);
  const state = status?.state || status?.status?.state || "offline";
  const pill = $("runtimeStatus");
  const button = $("runControlBtn");
  const labels = { running: "运行中", paused_manual: "手动暂停", paused_safe: "安全暂停",
    starting: "启动中", recovering: "重新识别中", process_exit: "进程已退出，可重启恢复", finished: "已结束", journey_end: "旅程结束", stopped: "已停止", offline: "未运行" };
  pill.textContent = `${labels[state] || state}${running && status.pid ? ` · PID ${status.pid}` : ""}`;
  pill.title = status.reason || status?.status?.reason || "";
  pill.classList.toggle("running", running && state === "running");
  pill.classList.toggle("error", state === "paused_safe" || state === "finished");
  button.textContent = running ? "■ 停止运行" : "▶ 启动自动化";
  button.classList.toggle("stop", running);
  button.classList.toggle("primary", !running);
  button.classList.toggle("secondary", running);
  const paused = running && ["paused_manual", "paused_safe"].includes(state);
  $("pauseRuntimeBtn").hidden = !running || !["running", "recovering"].includes(state);
  $("resumeRuntimeBtn").hidden = !paused;
  $("stepRuntimeBtn").hidden = !paused;
  $("restartRuntimeBtn").hidden = !running || state === "starting";
  $("inspectRuntimeBtn").hidden = !status?.status?.screenshot;
  const reason = status?.status?.diagnostics?.error || status?.reason || "";
  $("runtimeReason").textContent = state === "paused_safe" ? `${reason} · 等待人工或 AI 处理` :
    (reason || "自动运行与恢复由脚本完成；暂停后继续会重新识别当前界面");
  renderTargetApplied(status);
}

async function loadRuntimeStatus() {
  try {
    state.runtime = await api("/api/runtime");
    renderRuntimeStatus(state.runtime);
    if (state.selectedJourneyRecord) {
      $("saveJourneyResultBtn").disabled = !state.selectedJourneyRecord.outcome.journey_ended || Boolean(state.runtime.running && state.runtime.status?.journey_id === state.selectedJourneyRecord.journey_id);
      $("scanJourneyInfoBtn").disabled = !state.selectedJourneyRecord.outcome.journey_ended || Boolean(state.runtime.running);
    }
    const recordKey = `${state.runtime.status?.journey_id || ""}:${state.runtime.state === "journey_end" ? "ended" : "active"}`;
    if (state.lastJourneyRecordKey !== recordKey && state.runtime.status?.journey_id) {
      state.lastJourneyRecordKey = recordKey;
      await loadJourneyRecords();
    }
    const status = state.runtime.status || {};
    const key = `${status.run_id || ""}:${status.round_count || 0}:${state.runtime.state}`;
    if (state.activeTab === "run" && state.lastOverviewKey !== key) {
      state.lastOverviewKey = key;
      renderOverview(await api("/api/overview"));
    }
  }
  catch (error) { showToast(`读取运行状态失败：${error.message}`, true); }
}

async function toggleRuntime() {
  const running = Boolean(state.runtime?.running);
  if (!running && !targetIsSaved()) return;
  try {
    const status = await api(running ? "/api/runtime/stop" : "/api/runtime/start", { method: "POST" });
    state.runtime = status;
    renderRuntimeStatus(status);
    showToast(running ? "已请求停止，脚本正在保存状态" : "自动化已启动，脚本将识别当前界面");
    await loadOverview();
  } catch (error) { showToast(`${running ? "停止" : "启动"}自动化失败：${error.message}`, true); }
}

async function sendRuntimeCommand(path, label) {
  if (["resume", "step", "restart"].includes(path) && !targetIsSaved()) return;
  try {
    const status = await api(`/api/runtime/${path}`, { method: "POST" });
    state.runtime = status;
    renderRuntimeStatus(status);
    showToast(label);
  } catch (error) { showToast(`${label}失败：${error.message}`, true); }
}

async function inspectRuntime() {
  try {
    const data = await api("/api/runtime/diagnostics");
    if (!data.screenshot_url) { showToast("当前没有可用的现场截图", true); return; }
    $("decisionImage").src = data.screenshot_url;
    $("decisionImage").hidden = false;
    $("screenshotPlaceholder").hidden = true;
    $("screenshotCaption").textContent = `${data.status.last_handler || "现场"} · ${data.reason || data.state}`;
    $("screenshotPath").textContent = data.status.screenshot;
    $("screenshotBadge").textContent = "暂停现场";
  } catch (error) { showToast(`读取现场失败：${error.message}`, true); }
}

function renderEvents(items, total) {
  state.events = items;
  state.totalEvents = total;
  $("eventCount").textContent = `${items.length} / ${total}`;
  const rows = $("eventRows");
  removeInlineEditor();
  if (!items.length) {
    rows.innerHTML = '<tr><td colspan="6" class="empty-cell">没有匹配的事件</td></tr>';
    if (state.isNewEvent) {
      ensureInlineEditor(null);
      populateEventEditor(state.currentEvent, state.currentEvent.options || []);
    }
    return;
  }
  rows.innerHTML = items.map((event) => `
    <tr data-event-index="${event.index}" class="${event.index === state.currentEventIndex ? "active" : ""}" title="单击展开编辑，双击聚焦事件名">
      <td title="${esc(event.name)}">${esc(event.name)}</td>
      <td>${esc(event.category)}</td><td>${esc(event.source)}</td>
      <td>${esc(event.verified)}</td><td>${esc(event.recommended)}</td><td>${esc(event.options)}</td>
    </tr>`).join("");
  rows.querySelectorAll("tr[data-event-index]").forEach((row) => {
    row.addEventListener("click", () => {
      const index = Number(row.dataset.eventIndex);
      if (index === state.currentEventIndex && !state.isNewEvent && document.querySelector("[data-inline-editor]")) {
        collapseInlineEditor();
      } else {
        loadEvent(index);
      }
    });
    row.addEventListener("dblclick", () => {
      loadEvent(Number(row.dataset.eventIndex)).then(() => $("eventName").focus());
    });
  });
  if (state.isNewEvent) {
    ensureInlineEditor(null);
    populateEventEditor(state.currentEvent, state.currentEvent.options || []);
  } else if (state.currentEventIndex !== null && state.currentEvent) {
    ensureInlineEditor(state.currentEventIndex);
    populateEventEditor(state.currentEvent, state.currentEvent.options || []);
  }
}

async function loadEvents() {
  try {
    const query = encodeURIComponent($("eventSearch").value || "");
    const data = await api(`/api/events?query=${query}`);
    renderEvents(data.items || [], data.total || 0);
    $("eventRows").dataset.loaded = "1";
  } catch (error) { showToast(`读取事件库失败：${error.message}`, true); }
}

async function loadEvent(index) {
  try {
    const data = await api(`/api/events/${index}`);
    state.currentEventIndex = index;
    state.currentEvent = structuredClone(data.event);
    state.isNewEvent = false;
    state.selectedBranch = null;
    ensureInlineEditor(index);
    populateEventEditor(data.event, data.options || data.event.options || []);
    document.querySelectorAll("#eventRows tr[data-event-index]").forEach((row) =>
      row.classList.toggle("active", Number(row.dataset.eventIndex) === index));
  } catch (error) { showToast(`读取事件失败：${error.message}`, true); }
}

function removeInlineEditor() {
  document.querySelector("[data-inline-editor]")?.remove();
}

function collapseInlineEditor() {
  removeInlineEditor();
  state.currentEventIndex = null;
  state.currentEvent = null;
  state.selectedBranch = null;
  document.querySelectorAll("#eventRows tr[data-event-index]").forEach((row) => row.classList.remove("active"));
}

function ensureInlineEditor(index) {
  removeInlineEditor();
  const template = $("eventEditorTemplate");
  const rows = $("eventRows");
  if (!template || !rows) return false;
  const fragment = template.content.cloneNode(true);
  const target = index === null ? null : rows.querySelector(`tr[data-event-index="${index}"]`);
  if (target) target.after(fragment);
  else rows.prepend(fragment);
  $("updateBranchBtn").addEventListener("click", updateCurrentBranch);
  $("addBranchBtn").addEventListener("click", addBranch);
  $("deleteBranchBtn").addEventListener("click", deleteBranch);
  $("saveEventBtn").addEventListener("click", saveEvent);
  return true;
}

function populateEventEditor(event, options) {
  if (!$("eventEditor")) return;
  $("eventName").value = event.event_name || event.id || "";
  $("eventCategory").value = event.category || "";
  $("eventRecommended").value = event.recommended_option ?? "";
  $("eventEditorHint").textContent = `${event.id || "新事件"} · ${options.length} 个分支`;
  $("eventSourceBadge").textContent = event.document_verified ? "文档核验" : (event.status === "manual" ? "手动新增" : "本地规则");
  $("eventSourceBadge").classList.toggle("ready", Boolean(event.document_verified));
  renderBranches(options);
  if (state.selectedBranch !== null && options[state.selectedBranch]) selectBranch(state.selectedBranch);
}

function prepareNewEvent() {
  removeInlineEditor();
  state.currentEventIndex = null;
  state.currentEvent = {
    id: "",
    event_name: "",
    category: "手动事件",
    source_type: "local_rule",
    document_verified: false,
    recommended_option: 1,
    options: [{ index: 1, keyword: "", alias: [], effect_text: "" }],
  };
  state.isNewEvent = true;
  state.selectedBranch = 0;
  ensureInlineEditor(null);
  populateEventEditor(state.currentEvent, state.currentEvent.options);
  selectBranch(0);
  $("eventName").focus();
}

function renderBranches(options) {
  const rows = $("branchRows");
  if (!options.length) {
    rows.innerHTML = '<tr><td colspan="4" class="empty-cell">暂无分支</td></tr>';
    return;
  }
  rows.innerHTML = options.map((option, index) => `
    <tr data-branch-index="${index}" class="${state.selectedBranch === index ? "active" : ""}">
      <td>${esc(option.index ?? index + 1)}</td><td title="${esc(option.keyword || "")}">${esc(option.keyword || "—")}</td>
      <td title="${esc(option.effect_text || "")}">${esc(option.effect_text || "—")}</td>
      <td title="${esc(option.quantified_summary || "未量化")}">${esc(option.quantified_summary || "未量化")}</td>
    </tr>`).join("");
  rows.querySelectorAll("tr[data-branch-index]").forEach((row) =>
    row.addEventListener("click", () => selectBranch(Number(row.dataset.branchIndex))));
}

function selectBranch(index) {
  const options = state.currentEvent?.options || [];
  const option = options[index];
  if (!option) return;
  state.selectedBranch = index;
  $("branchIndex").value = option.index ?? index + 1;
  $("branchKeyword").value = option.keyword || "";
  $("branchAlias").value = (option.alias || []).join(",");
  $("branchEffect").value = option.effect_text || "";
  renderBranches(options);
}

function updateCurrentBranch() {
  if (!state.currentEvent || state.selectedBranch === null) {
    showToast("请先选择一个分支", true); return;
  }
  const option = state.currentEvent.options[state.selectedBranch];
  const number = Number($("branchIndex").value);
  if (!Number.isInteger(number) || number < 1) { showToast("分支序号必须是正整数", true); return; }
  option.index = number;
  option.keyword = $("branchKeyword").value.trim();
  option.alias = $("branchAlias").value.split(",").map((item) => item.trim()).filter(Boolean);
  option.effect_text = $("branchEffect").value.trim();
  renderBranches(state.currentEvent.options);
  showToast("分支已更新，保存事件后写入 JSON");
}

function addBranch() {
  if (!state.currentEvent) { showToast("请先选择一个事件", true); return; }
  const options = state.currentEvent.options || (state.currentEvent.options = []);
  const next = Math.max(0, ...options.map((option) => Number(option.index) || 0)) + 1;
  options.push({ index: next, keyword: "新分支", alias: [], effect_text: "" });
  state.selectedBranch = options.length - 1;
  renderBranches(options);
  selectBranch(state.selectedBranch);
}

function deleteBranch() {
  if (!state.currentEvent || state.selectedBranch === null) { showToast("请先选择一个分支", true); return; }
  state.currentEvent.options.splice(state.selectedBranch, 1);
  state.selectedBranch = null;
  renderBranches(state.currentEvent.options);
  showToast("分支已删除，保存事件后写入 JSON");
}

async function saveEvent() {
  if (!state.currentEvent) { showToast("请先选择一个事件", true); return; }
  const eventName = $("eventName").value.trim();
  if (!eventName) { showToast("事件名不能为空", true); $("eventName").focus(); return; }
  const recommended = Number($("eventRecommended").value);
  if (!Number.isInteger(recommended) || recommended < 1 ||
      ((state.currentEvent.options || []).length && recommended > state.currentEvent.options.length)) {
    showToast("默认分支必须是当前分支中的正整数", true); return;
  }
  state.currentEvent.event_name = eventName;
  state.currentEvent.category = $("eventCategory").value.trim() || "手动事件";
  state.currentEvent.recommended_option = recommended;
  try {
    const endpoint = state.isNewEvent ? "/api/events" : `/api/events/${state.currentEventIndex}`;
    const method = state.isNewEvent ? "POST" : "PUT";
    const data = await api(endpoint, {
      method, headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.currentEvent),
    });
    showToast(state.isNewEvent ? "新事件已添加到事件库" : "事件已保存到 document_verified.json");
    state.isNewEvent = false;
    await loadEvents();
    state.currentEventIndex = data.index ?? state.currentEventIndex;
    await loadEvent(state.currentEventIndex);
  } catch (error) { showToast(`保存事件失败：${error.message}`, true); }
}

function renderConfig(specs) {
  const grouped = {};
  specs.forEach((spec) => (grouped[spec.group] ||= []).push(spec));
  $("configGrid").innerHTML = Object.entries(grouped).map(([group, items]) => `
    <article class="config-card"><h3>${esc(group)}</h3>
      ${items.map((item) => `<div class="config-row"><div><span>${esc(item.label)}</span><small>${esc(item.path)}</small></div>
        <input class="config-input" data-path="${esc(item.path)}" value="${esc(item.value)}"></div>`).join("")}
    </article>`).join("");
  $("configGrid").dataset.loaded = "1";
}

async function loadConfig() {
  try { renderConfig((await api("/api/config")).specs || []); }
  catch (error) { showToast(`读取配置失败：${error.message}`, true); }
}

async function saveConfig() {
  const updates = {};
  document.querySelectorAll(".config-input").forEach((input) => { updates[input.dataset.path] = input.value.trim(); });
  try {
    const data = await api("/api/config", {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ updates }),
    });
    renderConfig(data.specs || []);
    showToast("配置已保存到 config.yaml");
  } catch (error) { showToast(`保存配置失败：${error.message}`, true); }
}

async function copyLog() {
  try {
    await navigator.clipboard.writeText($("logView").textContent);
    showToast("日志摘要已复制");
  } catch (_) { showToast("浏览器不允许访问剪贴板，请手动复制", true); }
}

function targetIsSaved() {
  if (state.targetDirty || !state.targetData?.target || !state.targetData.precheck?.ready) {
    $("journeyTargetPanel").open = true;
    showToast("请先预检查并保存本轮目标", true);
    return false;
  }
  return true;
}

function targetForm() {
  return { character: $("targetCharacter").value.trim(), playstyle: $("targetPlaystyle").value,
    combo_code: $("targetCombo").value, training_direction: $("targetTrainingDirection").value,
    has_partner: $("targetHasPartner").checked,
    actual_setup: { character: $("setupCharacter").value.trim() || null,
      journey_records: setupEntries("setupJourneyRecords", "journey_records"),
      support_cards: setupEntries("setupSupportCards", "support_cards") } };
}

function entryLines(value) {
  const text = value.trim();
  return text === "无" ? [] : text ? text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean) : null;
}

function setupEntries(id, key) {
  const names = entryLines($(id).value);
  const saved = state.targetData?.target?.actual_setup?.[key];
  if (saved && JSON.stringify(names) === JSON.stringify(saved.map((item) => item.name))) return saved;
  return names;
}

function targetDirty() {
  state.targetDirty = true;
  $("targetBadge").textContent = "未保存";
  $("targetSummary").textContent = "目标已更改，请重新预检查并保存";
  $("targetResult").textContent = "当前编辑尚未保存。";
  $("targetResult").className = "target-result";
}

function renderTargetRecommendations() {
  const catalog = state.targetData?.catalog;
  if (!catalog) return;
  const name = $("targetCharacter").value.trim();
  const character = catalog.characters.find((item) => [item.name, ...(item.aliases || [])].includes(name));
  const codes = character?.recommendations?.[$("targetPlaystyle").value] || [];
  const area = $("targetRecommendations");
  area.innerHTML = codes.length ? `<span>表格推荐 · 可任选其一：</span>${codes.map((code) => {
    const combo = catalog.combos.find((item) => item.code === code);
    return `<button class="button ghost" data-target-preset="${esc(code)}">${esc(combo.label)}</button>`;
  }).join("")}` : "该角色推荐尚未收录；可手动选择目标组合。";
  area.querySelectorAll("[data-target-preset]").forEach((button) => button.addEventListener("click", () => {
    $("targetCombo").value = button.dataset.targetPreset;
    updateTargetCombo(true);
    targetDirty();
  }));
}

function updateTargetCombo(suggest = false) {
  const combo = state.targetData?.catalog.combos.find((item) => item.code === $("targetCombo").value);
  const directions = { attack: "攻击向", survival: "生存向", comprehensive: "综合向", tactical: "战术向" };
  $("targetEffect").hidden = !combo;
  $("targetEffect").textContent = combo ? `${combo.label}\n${combo.effect_text}${combo.note ? `\n${combo.note}` : ""}\n脚本自动规划：三月选择${directions[combo.event_direction]}声援。${combo.special ? `取得${combo.partner_material}特殊装备。` : `组合需要${combo.initial_material}起始装备 ＋ ${combo.partner_material}后续装备。`}` : "";
  if (suggest && combo) {
    $("targetHasPartner").checked = false;
  }
}

function renderTargetCheck(result) {
  const directions = { attack: "攻击向", survival: "生存向", comprehensive: "综合向", tactical: "战术向" };
  const messages = result.ready ? [`预检查通过。脚本将根据 ${result.target.combo_code} 自动选择${directions[result.target.event_direction]}声援。`, ...(result.warnings || [])] : (result.errors || ["预检查未通过"]);
  $("targetResult").textContent = messages.join("\n");
  $("targetResult").className = `target-result ${result.ready ? "ready" : "error"}`;
}

function renderTargetApplied(runtime) {
  if (!state.targetData || state.targetDirty) return;
  const saved = state.targetData.target;
  if (!saved) return;
  const active = runtime?.status?.journey_target;
  const applied = active?.revision === saved.revision;
  const oldScript = runtime?.running && !("journey_target" in (runtime.status || {}));
  $("targetBadge").textContent = applied ? "已生效" : oldScript ? "待重启加载" : "已保存";
  const progress = runtime?.status?.journey_target_state;
  $("targetSummary").textContent = `${saved.character} · ${saved.combo_code}${applied && progress?.has_partner ? " · 材料已齐" : ""}`;
  if (oldScript) $("targetResult").textContent = "目标已保存。当前脚本尚未加载此功能，请将游戏切回前台后使用「重启恢复」。";
}

async function loadJourneyTarget() {
  try {
    const data = await api("/api/journey-target");
    state.targetData = data;
    state.targetDirty = false;
    $("targetCharacterList").innerHTML = data.catalog.characters.map((item) => `<option value="${esc(item.name)}"></option>`).join("");
    $("targetCombo").innerHTML = '<option value="">请选择目标组合</option>' + data.catalog.combos.map((item) => `<option value="${esc(item.code)}">${esc(item.label)}</option>`).join("");
    $("targetCoverage").textContent = `${data.catalog.coverage} · 核对于 ${data.catalog.verified_at}`;
    const target = data.target || {};
    $("targetCharacter").value = target.character || "";
    $("targetPlaystyle").value = target.playstyle || "pve";
    $("targetCombo").value = target.combo_code || "";
    $("targetTrainingDirection").value = target.training_direction || "";
    $("targetHasPartner").checked = Boolean(data.progress?.has_partner ?? target.has_partner);
    const setup = target.actual_setup || {};
    $("setupCharacter").value = setup.character || "";
    for (const [id, key] of [["setupJourneyRecords", "journey_records"], ["setupSupportCards", "support_cards"]]) {
      $(id).value = setup[key] ? setup[key].map((item) => item.name).join("\n") || "无" : "";
    }
    renderTargetRecommendations();
    updateTargetCombo();
    if (data.target) renderTargetCheck(data.precheck);
    renderTargetApplied(state.runtime);
  } catch (error) { showToast(`读取旅程目标失败：${error.message}`, true); }
}

async function checkOrSaveTarget(save = false) {
  const button = $(save ? "saveTargetBtn" : "precheckTargetBtn");
  button.disabled = true;
  try {
    const result = await api(save ? "/api/journey-target" : "/api/journey-target/precheck", {
      method: save ? "PUT" : "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(targetForm()),
    });
    if (save) {
      await loadJourneyTarget();
      showToast("本轮目标已保存");
    } else renderTargetCheck(result);
  } catch (error) { showToast(error.message, true); $("targetResult").textContent = error.message; $("targetResult").className = "target-result error"; }
  finally { button.disabled = false; }
}

for (const id of ["targetCharacter", "targetPlaystyle", "targetTrainingDirection", "targetHasPartner", "setupCharacter", "setupJourneyRecords", "setupSupportCards"]) {
  $(id).addEventListener(["targetCharacter", "setupCharacter", "setupJourneyRecords", "setupSupportCards"].includes(id) ? "input" : "change", () => {
    targetDirty();
    if (["targetCharacter", "targetPlaystyle"].includes(id)) renderTargetRecommendations();
  });
}
$("targetCombo").addEventListener("change", () => { updateTargetCombo(true); targetDirty(); });
$("precheckTargetBtn").addEventListener("click", () => checkOrSaveTarget());
$("saveTargetBtn").addEventListener("click", () => checkOrSaveTarget(true));

function renderJourneyRecord() {
  const record = state.journeyRecords?.find((item) => item.journey_id === $("journeyRecordSelect").value);
  state.selectedJourneyRecord = record || null;
  $("saveJourneyResultBtn").disabled = !record?.outcome.journey_ended || Boolean(state.runtime?.running && state.runtime.status?.journey_id === record?.journey_id);
  $("scanJourneyInfoBtn").disabled = !record?.outcome.journey_ended || Boolean(state.runtime?.running);
  $("journeyResultFields").hidden = !record;
  const scan = record?.latest_screen_scan;
  const evidenceScreenshot = scan?.screenshot || record?.outcome.final_observation?.screenshot;
  $("journeyEndScreenshot").hidden = !evidenceScreenshot;
  $("journeyEndOcr").textContent = (scan?.ocr || record?.outcome.final_observation?.ocr || []).map((item) => item.text).join("\n") || "尚未记录";
  if (!record) return;
  const target = record.target_history.at(-1);
  const sourceNames = { user_declared: "人工填写", screen_confirmed: "已核对画面", unobserved: "未记录" };
  const setupText = (key) => {
    const entry = record.actual_setup[key];
    const value = entry.value;
    const text = value === null ? "未记录" : Array.isArray(value) ? value.map((item) => item.name).join("、") || "无" : value;
    return `${text}（${sourceNames[entry.source] || entry.source}）`;
  };
  $("journeyRecordSummary").textContent = `${target?.character || "目标未记录"} · ${record.outcome.journey_ended ? "已结束" : "尚未结束"}`;
  $("journeyRecordInfo").textContent = [`目标：${target ? `${target.character} · ${target.combo_code}` : "未记录"}`,
    `实际角色：${setupText("character")}`, `旅程记录：${setupText("journey_records")}`, `支援卡：${setupText("support_cards")}`,
    `运行过程：${record.outcome.execution_mode === "exploration" ? "含未知事件探索" : "正常规则"} · 探索次数：${record.exploration?.attempts || 0}`,
    `关联运行：${record.segments.length} 次 · 待核对字段：${record.data_quality.missing_fields.length} 项`,
    ...record.data_quality.warnings].join("\n");
  $("journeyResultFields").querySelectorAll("[data-final-stat]").forEach((input) => { input.value = record.outcome.final_stats[input.dataset.finalStat].value ?? ""; });
  $("resultCharacter").value = record.actual_setup.character.value || "";
  for (const [id, key] of [["resultJourneyRecords", "journey_records"], ["resultSupportCards", "support_cards"]]) {
    const entries = record.actual_setup[key].value;
    $(id).value = entries ? entries.map((item) => item.name).join("\n") || "无" : "";
  }
  $("resultRank").value = record.outcome.rank.value ?? "";
  $("resultPotential").value = record.outcome.potential_points.value ?? "";
  $("resultAppraisal").value = record.outcome.appraisal_result.value || "";
  const skills = record.outcome.equipped_skills.value;
  $("resultSkills").value = skills ? skills.join("\n") || "无" : "";
  if (evidenceScreenshot) $("journeyEndScreenshot").href = `/api/screenshot?path=${encodeURIComponent(evidenceScreenshot)}`;
  const achieved = record.outcome.target_skill_achieved.value;
  $("journeyResultMessage").textContent = !record.outcome.journey_ended ? "本轮旅程尚未结束，最终结果保持待核对。" :
    `目标技能${achieved === null ? "尚未核对" : achieved ? "已达成" : "未达成"}；${record.outcome.execution_mode === "exploration" ? "本轮含探索过程" : "本轮按正常规则完成"}。` +
    (scan ? `最近一次画面扫描：${formatTime(scan.observed_at)}；仍待核对 ${record.data_quality.missing_fields.length} 项。` : "可读取当前放大镜页面自动补录。");
}

async function loadJourneyRecords() {
  try {
    const data = await api("/api/journey-records");
    const selected = $("journeyRecordSelect").value;
    state.journeyRecords = data.records;
    $("journeyRecordSelect").innerHTML = data.records.length ? data.records.map((record) => {
      const target = record.target_history.at(-1);
      return `<option value="${esc(record.journey_id)}">${esc(record.started_at)} · ${esc(target?.character || "角色未记录")} · ${record.outcome.journey_ended ? "已结束" : "未结束"}</option>`;
    }).join("") : '<option value="">暂无记录</option>';
    if (data.records.some((item) => item.journey_id === selected)) $("journeyRecordSelect").value = selected;
    else if (data.records.some((item) => item.journey_id === data.active_journey_id)) $("journeyRecordSelect").value = data.active_journey_id;
    renderJourneyRecord();
  } catch (error) { showToast(`读取旅程记录失败：${error.message}`, true); }
}

async function saveJourneyResult() {
  const record = state.selectedJourneyRecord;
  if (!record) return;
  const body = { final_stats: {}, actual_setup: { character: $("resultCharacter").value.trim() || null } };
  for (const [id, key] of [["resultJourneyRecords", "journey_records"], ["resultSupportCards", "support_cards"]]) {
    const names = entryLines($(id).value);
    const existing = record.actual_setup[key].value;
    body.actual_setup[key] = existing && JSON.stringify(names) === JSON.stringify(existing.map((item) => item.name)) ? existing : names;
  }
  $("journeyResultFields").querySelectorAll("[data-final-stat]").forEach((input) => { if (input.value !== "") body.final_stats[input.dataset.finalStat] = Number(input.value); });
  for (const [id, key] of [["resultRank", "rank"], ["resultPotential", "potential_points"]]) {
    if ($(id).value !== "") body[key] = Number($(id).value);
  }
  if ($("resultAppraisal").value) body.appraisal_result = $("resultAppraisal").value;
  const skills = entryLines($("resultSkills").value);
  if (skills !== null) body.equipped_skills = skills;
  $("saveJourneyResultBtn").disabled = true;
  try {
    await api(`/api/journey-records/${encodeURIComponent(record.journey_id)}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    await loadJourneyRecords();
    showToast("旅程结果已补录");
  } catch (error) { $("journeyResultMessage").textContent = error.message; }
  finally { $("saveJourneyResultBtn").disabled = !record.outcome.journey_ended || Boolean(state.runtime?.running && state.runtime.status?.journey_id === record.journey_id); }
}

async function scanJourneyInfo() {
  const record = state.selectedJourneyRecord;
  if (!record?.outcome.journey_ended) return;
  const button = $("scanJourneyInfoBtn");
  button.disabled = true;
  $("journeyResultMessage").textContent = "正在验证前台游戏并读取旅程信息，请保持放大镜页面不动…";
  try {
    const data = await api(`/api/journey-records/${encodeURIComponent(record.journey_id)}/scan-current`, { method: "POST" });
    await loadJourneyRecords();
    const setupCount = Object.keys(data.scan?.actual_setup || {}).length;
    const outcomeCount = Object.keys(data.scan?.outcome || {}).length;
    showToast(`已读取旅程信息，自动补录 ${setupCount + outcomeCount} 组字段`);
  } catch (error) {
    $("journeyResultMessage").textContent = error.message;
    showToast(`读取旅程信息失败：${error.message}`, true);
  } finally {
    if (state.selectedJourneyRecord) button.disabled = !state.selectedJourneyRecord.outcome.journey_ended || Boolean(state.runtime?.running);
  }
}

function renderQuickLookup(data) {
  state.quickLookup = data || null;
  const match = data?.match;
  const rescuer = data?.rescuer;
  const arcanum = data?.arcanum;
  const details = rescuer?.roster_details || [];
  const lines = [];
  if (rescuer) {
    const character = rescuer.character?.value || "未确认";
    lines.push(`救援者：${character} · 已读取可见角色详情 ${details.length} 条`);
    if (rescuer.title) lines.push(`当前标题：${rescuer.title}`);
    if (rescuer.level) lines.push(`当前等级：Lv.${rescuer.level}${rescuer.level_max ? `/${rescuer.level_max}` : ""}`);
    if (rescuer.rarity) lines.push(`稀有度：${rescuer.rarity}`);
    if (details.length) lines.push(`角色列表：${details.map((item) => `${item.character || "未知"} Lv.${item.level ?? "?"}`).join("、")}`);
  }
  if (arcanum) {
    const card = arcanum.cards?.[0] || {};
    const name = card.name?.value || "卡名待确认";
    lines.push(`阿尔克那：${name} · ${card.rarity || "稀有度未知"} · Lv.${card.level ?? "?"}`);
    const effects = Object.values(card.effects || {}).flat().map((item) => item.text).filter(Boolean);
    if (effects.length) lines.push(`效果：${effects.join("；")}`);
  }
  if (match) {
    lines.push(`对应关系：${match.status === "matched" ? "已匹配" : "部分匹配"}`);
    lines.push(`旅程角色字段：${match.journey_end_mapping?.["actual_setup.character"]?.value || "未确认"}`);
    const cards = match.journey_end_mapping?.["actual_setup.support_cards"]?.value;
    lines.push(`旅程支援卡字段：${cards?.map((item) => item.name).join("、") || "未确认"}`);
    lines.push("救援者当前属性不会写入旅程最终属性字段。" );
    if ((match.uncertain || []).length) lines.push(`待人工确认：${match.uncertain.join("、")}`);
  }
  $("quickLookupResult").textContent = lines.join("\n") || "尚未读取";
  $("quickLookupBadge").textContent = match ? (match.status === "matched" ? "已匹配" : "部分匹配") : (rescuer || arcanum ? "已读取" : "未读取");
  $("quickLookupBadge").classList.toggle("ready", Boolean(match));
  $("quickLookupSummary").textContent = match ? `最近匹配：${formatTime(match.observed_at)}` : "打开对应界面后分别读取，脚本会遍历可见角色行";
}

async function loadQuickLookup() {
  try { renderQuickLookup(await api("/api/quick-lookup")); }
  catch (error) { showToast(`读取快查状态失败：${error.message}`, true); }
}

async function quickLookupAction(path, label) {
  const button = $(path.includes("rescuer") ? "scanRescuerBtn" : "scanArcanumBtn");
  button.disabled = true;
  $("quickLookupMessage").textContent = `${label}；请保持 StarSavior 在前台…`;
  try {
    const data = await api(`/api/quick-lookup/${path}`, { method: "POST" });
    renderQuickLookup(data);
    $("quickLookupMessage").textContent = `${label}完成，截图和 OCR 已保存。`;
    showToast(`${label}完成`);
  } catch (error) {
    $("quickLookupMessage").textContent = error.message;
    showToast(`${label}失败：${error.message}`, true);
  } finally { button.disabled = false; }
}

async function matchQuickLookup() {
  const button = $("matchQuickLookupBtn");
  button.disabled = true;
  try {
    const data = await api("/api/quick-lookup/match", { method: "POST" });
    renderQuickLookup(data);
    $("quickLookupMessage").textContent = data.match?.status === "matched" ? "已生成角色、支援卡与旅程结束字段对应关系。" : "已生成部分对应关系，未确认字段已保留。";
    showToast("对应关系已生成");
  } catch (error) { $("quickLookupMessage").textContent = error.message; showToast(`生成对应关系失败：${error.message}`, true); }
  finally { button.disabled = false; }
}

$("journeyRecordSelect").addEventListener("change", renderJourneyRecord);
$("refreshJourneyRecordsBtn").addEventListener("click", loadJourneyRecords);
$("scanJourneyInfoBtn").addEventListener("click", scanJourneyInfo);
$("saveJourneyResultBtn").addEventListener("click", saveJourneyResult);
$("scanRescuerBtn").addEventListener("click", () => quickLookupAction("scan-rescuer", "救援者扫描"));
$("scanArcanumBtn").addEventListener("click", () => quickLookupAction("scan-arcanum", "阿尔克那扫描"));
$("matchQuickLookupBtn").addEventListener("click", matchQuickLookup);

document.querySelectorAll(".nav-item").forEach((button) => button.addEventListener("click", () => setTab(button.dataset.tab)));
$("refreshBtn").addEventListener("click", async () => { await loadOverview(); if (state.activeTab === "events") await loadEvents(); showToast("数据已刷新"); });
$("runControlBtn").addEventListener("click", toggleRuntime);
$("pauseRuntimeBtn").addEventListener("click", () => sendRuntimeCommand("pause", "脚本已请求暂停"));
$("resumeRuntimeBtn").addEventListener("click", () => sendRuntimeCommand("resume", "已请求继续，脚本将重新识别"));
$("stepRuntimeBtn").addEventListener("click", () => sendRuntimeCommand("step", "脚本将处理一个页面后暂停"));
$("restartRuntimeBtn").addEventListener("click", () => sendRuntimeCommand("restart", "脚本已请求重启恢复"));
$("inspectRuntimeBtn").addEventListener("click", inspectRuntime);
$("eventSearch").addEventListener("input", loadEvents);
$("newEventBtn").addEventListener("click", prepareNewEvent);
$("saveConfigBtn").addEventListener("click", saveConfig);
$("copyLogBtn").addEventListener("click", copyLog);

loadOverview();
loadJourneyTarget();
loadJourneyRecords();
loadQuickLookup();
setInterval(loadRuntimeStatus, 2000);
