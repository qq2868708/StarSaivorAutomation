const state = {
  overview: null,
  events: [],
  totalEvents: 0,
  currentEventIndex: null,
  currentEvent: null,
  selectedBranch: null,
  activeTab: "run",
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
  const pill = $("runtimeStatus");
  const button = $("runControlBtn");
  pill.textContent = running ? `运行中 · PID ${status.pid}` :
    (status?.returncode !== null && status?.returncode !== undefined ? `已停止 · ${status.returncode}` : "未运行");
  pill.classList.toggle("running", running);
  pill.classList.toggle("error", !running && status?.returncode !== null && status?.returncode !== 0);
  button.textContent = running ? "■ 停止运行" : "▶ 启动自动化";
  button.classList.toggle("stop", running);
  button.classList.toggle("primary", !running);
  button.classList.toggle("secondary", running);
}

async function loadRuntimeStatus() {
  try {
    state.runtime = await api("/api/runtime");
    renderRuntimeStatus(state.runtime);
  }
  catch (error) { showToast(`读取运行状态失败：${error.message}`, true); }
}

async function toggleRuntime() {
  const running = Boolean(state.runtime?.running);
  try {
    const status = await api(running ? "/api/runtime/stop" : "/api/runtime/start", { method: "POST" });
    state.runtime = status;
    renderRuntimeStatus(status);
    showToast(running ? "自动化已停止" : "自动化已启动，脚本正在等待游戏界面");
    await loadOverview();
  } catch (error) { showToast(`${running ? "停止" : "启动"}自动化失败：${error.message}`, true); }
}

function renderEvents(items, total) {
  state.events = items;
  state.totalEvents = total;
  $("eventCount").textContent = `${items.length} / ${total}`;
  const rows = $("eventRows");
  if (!items.length) {
    rows.innerHTML = '<tr><td colspan="6" class="empty-cell">没有匹配的事件</td></tr>';
    return;
  }
  rows.innerHTML = items.map((event) => `
    <tr data-event-index="${event.index}" class="${event.index === state.currentEventIndex ? "active" : ""}">
      <td title="${esc(event.name)}">${esc(event.name)}</td>
      <td>${esc(event.category)}</td><td>${esc(event.source)}</td>
      <td>${esc(event.verified)}</td><td>${esc(event.recommended)}</td><td>${esc(event.options)}</td>
    </tr>`).join("");
  rows.querySelectorAll("tr[data-event-index]").forEach((row) => {
    row.addEventListener("click", () => loadEvent(Number(row.dataset.eventIndex)));
  });
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
    state.selectedBranch = null;
    $("eventEmpty").hidden = true;
    $("eventEditor").hidden = false;
    $("eventName").value = data.event.event_name || data.event.id || "";
    $("eventRecommended").value = data.event.recommended_option ?? "";
    $("eventEditorHint").textContent = `${data.event.id || "未命名"} · ${data.event.options?.length || 0} 个分支`;
    $("eventSourceBadge").textContent = data.event.document_verified ? "文档核验" : "本地规则";
    $("eventSourceBadge").classList.toggle("ready", Boolean(data.event.document_verified));
    renderBranches(data.options || data.event.options || []);
    document.querySelectorAll("#eventRows tr[data-event-index]").forEach((row) =>
      row.classList.toggle("active", Number(row.dataset.eventIndex) === index));
  } catch (error) { showToast(`读取事件失败：${error.message}`, true); }
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
  if (state.currentEventIndex === null || !state.currentEvent) { showToast("请先选择一个事件", true); return; }
  state.currentEvent.event_name = $("eventName").value.trim();
  state.currentEvent.recommended_option = Number($("eventRecommended").value);
  try {
    await api(`/api/events/${state.currentEventIndex}`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.currentEvent),
    });
    showToast("事件已保存到 document_verified.json");
    await loadEvents();
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

document.querySelectorAll(".nav-item").forEach((button) => button.addEventListener("click", () => setTab(button.dataset.tab)));
$("refreshBtn").addEventListener("click", async () => { await loadOverview(); if (state.activeTab === "events") await loadEvents(); showToast("数据已刷新"); });
$("runControlBtn").addEventListener("click", toggleRuntime);
$("eventSearch").addEventListener("input", loadEvents);
$("updateBranchBtn").addEventListener("click", updateCurrentBranch);
$("addBranchBtn").addEventListener("click", addBranch);
$("deleteBranchBtn").addEventListener("click", deleteBranch);
$("saveEventBtn").addEventListener("click", saveEvent);
$("saveConfigBtn").addEventListener("click", saveConfig);
$("copyLogBtn").addEventListener("click", copyLog);

loadOverview();
setInterval(loadRuntimeStatus, 2000);
