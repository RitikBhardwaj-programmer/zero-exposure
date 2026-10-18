/* Zero Exposure dashboard front end: one state object, filled by server-sent events from a live run or a replay. */
"use strict";

const HOSPITAL_LABELS = ["Cleveland", "Hungarian", "Switzerland", "VA"];
const ATTACK_LABELS = { none: "No attack", labelflip: "Label flip", boost: "Boost", flipboost: "Label flip + boost" };
const AGG_LABELS = { fedavg: "Mean", median: "Median", trimmed: "Trimmed mean" };
const $ = (id) => document.getElementById(id);

const state = {
  config: null,         // GET /api/config
  source: "live",       // "live" | "replay"
  stream: null,         // EventSource
  inspectTimer: null,
  sessionKind: null,    // what the panels show now: "live" | "replay"
  sessionName: "",      // run id or replay name
  baseline: null,
  rounds: [],
  attack: null,
  totalRounds: 0,
  inspect: null,
  inspectRound: 1,
  sweeps: null,
  running: false,
};

/* ---------------------------------------------------------------------------------------------------- helpers */

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const pct = (x) => (x == null ? "–" : `${(x * 100).toFixed(1)}%`);
const num = (x, d = 2) => (x == null ? "–" : Number(x).toFixed(d));
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function setStatus(text, kind = "") {
  const el = $("status");
  el.textContent = text;
  el.className = "status" + (kind ? " " + kind : "");
}

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  let body = null;
  try { body = await res.json(); } catch (_) { body = null; }
  if (!res.ok) {
    const detail = body && body.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : res.statusText;
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return body;
}

function epsBand(e) {
  if (e == null) return { cls: "weak", text: "no formal privacy (noise 0)" };
  if (e < 1) return { cls: "strong", text: "strong privacy (< 1)" };
  if (e <= 10) return { cls: "moderate", text: "moderate privacy (1–10)" };
  return { cls: "weak", text: "weak privacy (> 10)" };
}

function hospitalColor(i) { return css(`--h${i}`); }

/* ---------------------------------------------------------------------------------------------------- theme */

function storedTheme() { try { return localStorage.getItem("ze-theme"); } catch (_) { return null; } }
function applyTheme(theme) {
  if (theme) document.documentElement.setAttribute("data-theme", theme);
  else document.documentElement.removeAttribute("data-theme");
}
function currentTheme() {
  const t = document.documentElement.getAttribute("data-theme");
  if (t) return t;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
function toggleTheme() {
  const next = currentTheme() === "dark" ? "light" : "dark";
  applyTheme(next);
  try { localStorage.setItem("ze-theme", next); } catch (_) { /* private mode: theme just isn't remembered */ }
  restyleCharts();
  renderInspector();
}

/* ---------------------------------------------------------------------------------------------------- charts */

const charts = {};

// Draws ε privacy bands behind the epsilon chart.
const bandPlugin = {
  id: "epsBands",
  beforeDatasetsDraw(chart) {
    if (chart.canvas.id !== "c-eps") return;
    const { ctx, chartArea: a, scales: { y } } = chart;
    if (!a || !y) return;
    const band = (lo, hi, color) => {
      const top = y.getPixelForValue(Math.min(hi, y.max)), bottom = y.getPixelForValue(Math.max(lo, y.min));
      if (bottom <= top) return;
      ctx.fillStyle = color;
      ctx.fillRect(a.left, top, a.right - a.left, bottom - top);
    };
    ctx.save();
    band(0, 1, css("--band-strong"));
    band(1, 10, css("--band-moderate"));
    band(10, Math.max(y.max, 10), css("--band-weak"));
    ctx.restore();
  },
};

// Draws ± std whiskers for datasets that carry `errors` (one value per point).
const whiskerPlugin = {
  id: "whiskers",
  afterDatasetsDraw(chart) {
    const { ctx } = chart;
    chart.data.datasets.forEach((ds, di) => {
      if (!ds.errors) return;
      const meta = chart.getDatasetMeta(di);
      if (meta.hidden) return;
      const y = chart.scales[meta.yAxisID];
      ctx.save();
      ctx.strokeStyle = ds.borderColor;
      ctx.lineWidth = 1.5;
      meta.data.forEach((el, i) => {
        const err = ds.errors[i];
        const raw = ds.data[i];
        const v = raw == null ? null : typeof raw === "object" ? raw.y : raw;
        if (err == null || v == null || !err) return;
        const x = el.x, t = y.getPixelForValue(v + err), b = y.getPixelForValue(v - err);
        ctx.beginPath();
        ctx.moveTo(x, t); ctx.lineTo(x, b);
        ctx.moveTo(x - 4, t); ctx.lineTo(x + 4, t);
        ctx.moveTo(x - 4, b); ctx.lineTo(x + 4, b);
        ctx.stroke();
      });
      ctx.restore();
    });
  },
};

// Writes a short label at the end of each line that asks for one (direct labels, so identity is never color alone).
const endLabelPlugin = {
  id: "endLabels",
  afterDatasetsDraw(chart) {
    const { ctx, chartArea: a } = chart;
    if (!a || chart.width < 420) return;
    ctx.save();
    ctx.font = `600 11px ${css("--font")}`;
    ctx.fillStyle = css("--text-2");
    ctx.textBaseline = "middle";
    const used = [];
    chart.data.datasets.forEach((ds, di) => {
      if (!ds.endLabel) return;
      const meta = chart.getDatasetMeta(di);
      if (meta.hidden) return;
      let last = null;
      meta.data.forEach((el, i) => { if (ds.data[i] != null && !el.skip) last = el; });
      if (!last) return;
      let y = last.y;
      while (used.some((u) => Math.abs(u - y) < 12)) y += 12;
      used.push(y);
      ctx.fillText(ds.endLabel, Math.min(last.x + 6, a.right + 4), y);
    });
    ctx.restore();
  },
};

function baseOptions(extra = {}) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 250 },
    interaction: { mode: "index", intersect: false },
    layout: { padding: (c) => ({ right: c.chart.width < 420 ? 8 : 72 }) },
    plugins: {
      legend: { position: "bottom", labels: { boxWidth: 14, boxHeight: 3, usePointStyle: false, color: css("--text-2") } },
      tooltip: { backgroundColor: css("--surface"), titleColor: css("--text"), bodyColor: css("--text-2"), borderColor: css("--border"),
                 borderWidth: 1, padding: 10, bodyFont: { family: css("--mono") } },
    },
    scales: {
      x: { grid: { color: css("--grid"), drawTicks: false }, ticks: { color: css("--muted") }, border: { color: css("--border") } },
      y: { grid: { color: css("--grid"), drawTicks: false }, ticks: { color: css("--muted") }, border: { display: false } },
    },
    ...extra,
  };
}

function initCharts() {
  if (!window.Chart) return;
  Chart.defaults.font.family = css("--font");
  Chart.register(bandPlugin, whiskerPlugin, endLabelPlugin);

  const accDatasets = [
    { label: "Global model", data: [], borderColor: css("--teal"), backgroundColor: css("--teal"), borderWidth: 3, pointRadius: 0,
      pointHoverRadius: 5, tension: 0.25, endLabel: "Global", order: 0 },
    ...HOSPITAL_LABELS.map((name, i) => ({ label: name, data: [], borderColor: hospitalColor(i), backgroundColor: hospitalColor(i),
      borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, tension: 0.25, endLabel: name, order: 1 })),
    { label: "Centralized (no privacy)", data: [], borderColor: css("--ref"), borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0,
      endLabel: "Central", order: 2, refKey: "central" },
    { label: "Local-only (average)", data: [], borderColor: css("--ref"), borderDash: [2, 3], borderWidth: 1.5, pointRadius: 0,
      endLabel: "Local", order: 2, refKey: "local" },
  ];
  charts.acc = new Chart($("c-acc"), {
    type: "line",
    data: { labels: [], datasets: accDatasets },
    options: baseOptions({
      scales: { ...baseOptions().scales,
        x: { ...baseOptions().scales.x, title: { display: true, text: "round", color: css("--muted") } },
        y: { ...baseOptions().scales.y, min: 0, max: 1, ticks: { color: css("--muted"), callback: (v) => `${Math.round(v * 100)}%` } } },
      plugins: { ...baseOptions().plugins,
        tooltip: { ...baseOptions().plugins.tooltip, callbacks: { title: (it) => `Round ${it[0].label}`,
          label: (c) => ` ${c.dataset.label}: ${pct(c.parsed.y)}` } } },
    }),
  });

  charts.eps = new Chart($("c-eps"), {
    type: "line",
    data: { labels: [], datasets: HOSPITAL_LABELS.map((name, i) => ({ label: name, data: [], borderColor: hospitalColor(i),
      backgroundColor: hospitalColor(i), borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, tension: 0.2, endLabel: name })) },
    options: baseOptions({
      scales: { ...baseOptions().scales,
        x: { ...baseOptions().scales.x, title: { display: true, text: "round", color: css("--muted") } },
        y: { ...baseOptions().scales.y, min: 0, suggestedMax: 2, title: { display: true, text: "ε", color: css("--muted") } } },
      plugins: { ...baseOptions().plugins,
        tooltip: { ...baseOptions().plugins.tooltip, callbacks: { title: (it) => `Round ${it[0].label}`,
          label: (c) => ` ${c.dataset.label}: ε ${num(c.parsed.y)} · ${epsBand(c.parsed.y).text}` } } },
    }),
  });

  charts.sweep = new Chart($("c-sweep"), {
    type: "line",
    data: { datasets: [] },
    options: baseOptions({
      interaction: { mode: "nearest", intersect: false },
      scales: { x: { ...baseOptions().scales.x, type: "linear", title: { display: true, text: "noise multiplier (more noise = more privacy)", color: css("--muted") } },
                y: { ...baseOptions().scales.y, title: { display: true, text: "global accuracy", color: css("--muted") },
                     ticks: { color: css("--muted"), callback: (v) => `${Math.round(v * 100)}%` } } },
      plugins: { ...baseOptions().plugins, tooltip: { ...baseOptions().plugins.tooltip, callbacks: {
        title: (it) => `noise ${it[0].parsed.x}`,
        label: (c) => {
          const g = c.raw.g;
          const eps = g.epsilon_max.mean == null ? "unbounded" : `${num(g.epsilon_max.mean)} ± ${num(g.epsilon_max.std)}`;
          return ` ${c.dataset.label}: ${pct(g.fed_acc.mean)} ± ${pct(g.fed_acc.std)} · ε ${eps} · ${g.n} run(s)`;
        } } } },
    }),
  });

  charts.attack = new Chart($("c-attack"), {
    type: "bar",
    data: { labels: [], datasets: [] },
    options: baseOptions({
      layout: { padding: { right: 8 } },
      interaction: { mode: "nearest", intersect: true },
      scales: { x: { ...baseOptions().scales.x, grid: { display: false } },
                y: { ...baseOptions().scales.y, min: 0, max: 1, ticks: { color: css("--muted"), callback: (v) => `${Math.round(v * 100)}%` } } },
      plugins: { ...baseOptions().plugins, tooltip: { ...baseOptions().plugins.tooltip, callbacks: {
        label: (c) => ` ${c.dataset.label}: ${pct(c.parsed.y)} ± ${pct(c.dataset.errors[c.dataIndex])}` } } },
    }),
  });
}

function restyleCharts() {
  if (!window.Chart) return;
  for (const chart of Object.values(charts)) {
    const o = chart.options;
    o.plugins.legend.labels.color = css("--text-2");
    Object.assign(o.plugins.tooltip, { backgroundColor: css("--surface"), titleColor: css("--text"), bodyColor: css("--text-2"), borderColor: css("--border") });
    for (const s of Object.values(o.scales)) {
      s.grid.color = css("--grid");
      s.ticks.color = css("--muted");
      if (s.border) s.border.color = css("--border");
      if (s.title) s.title.color = css("--muted");
    }
  }
  if (charts.acc) {
    charts.acc.data.datasets.forEach((ds, i) => {
      const c = i === 0 ? css("--teal") : i <= 4 ? hospitalColor(i - 1) : css("--ref");
      ds.borderColor = c; ds.backgroundColor = c;
    });
    charts.eps.data.datasets.forEach((ds, i) => { ds.borderColor = hospitalColor(i); ds.backgroundColor = hospitalColor(i); });
  }
  renderSweeps();
  Object.values(charts).forEach((c) => c.update("none"));
}

/* ---------------------------------------------------------------------------------------------------- session */

function resetSession(kind, name) {
  state.sessionKind = kind;
  state.sessionName = name;
  state.baseline = null;
  state.rounds = [];
  state.attack = null;
  state.totalRounds = 0;
  state.inspect = null;
  state.inspectRound = 1;
  $("source-tag").textContent = kind === "replay" ? `Replay of a recorded run: ${name}` : `Live run ${name}`;
  renderTraining();
  renderInspector();
  renderAttackCards();
}

function closeStream() {
  if (state.stream) { state.stream.close(); state.stream = null; }
  if (state.inspectTimer) { clearInterval(state.inspectTimer); state.inspectTimer = null; }
}

function onEvent(ev) {
  switch (ev.type) {
    case "baseline":
      state.baseline = ev;
      state.totalRounds = (ev.config && ev.config.rounds) || state.totalRounds;
      state.inspectRound = (ev.config && ev.config["inspect-round"]) || 1;
      break;
    case "round":
      state.totalRounds = ev.total_rounds || state.totalRounds;
      state.rounds = state.rounds.filter((r) => r.round !== ev.round).concat([ev]).sort((a, b) => a.round - b.round);
      break;
    case "attack":
      state.attack = ev;
      renderAttackCards();
      break;
    case "error":
      setStatus(`The run failed: ${ev.message}`, "err");
      break;
    case "done":
      break;
    default:
      return;
  }
  renderTraining();
  if (state.sessionKind === "replay") renderInspector();
}

function listen(url, onEnd) {
  closeStream();
  const es = new EventSource(url);
  state.stream = es;
  es.onmessage = (m) => {
    let ev;
    try { ev = JSON.parse(m.data); } catch (_) { return; }
    onEvent(ev);
  };
  es.addEventListener("end", (m) => {
    let info = null;
    try { info = JSON.parse(m.data); } catch (_) { info = null; }
    es.close();
    if (state.stream === es) state.stream = null;
    onEnd(info);
  });
  es.onerror = () => {
    if (es.readyState === EventSource.CLOSED) return;
    // The server ends the stream after done / error / stop; a broken connection retries on its own.
  };
}

/* ---------------------------------------------------------------------------------------------------- live runs */

function formParams() {
  const f = $("run-form");
  return {
    noise: Number(f.noise.value),
    strategy: f.strategy.value,
    secagg: f.secagg.checked,
    rounds: parseInt(f.rounds.value, 10),
    seed: parseInt(f.seed.value, 10),
    attack: f.attack.value,
    attacker: parseInt(f.attacker.value, 10),
    agg: f.agg.value,
  };
}

function setRunning(running) {
  state.running = running;
  $("start-btn").disabled = running || state.source !== "live";
  $("stop-btn").disabled = !running;
  $("play-btn").disabled = running;
}

async function pollInspect(runId) {
  try {
    state.inspect = await api(`/api/runs/${runId}/inspect`);
    renderInspector();
  } catch (_) { /* next poll */ }
}

function followRun(run) {
  resetSession("live", run.id);
  setRunning(run.status === "running");
  setStatus(run.status === "running" ? "Training… each hospital trains on its own patients, then the server merges." : `Run ${run.status}.`);
  pollInspect(run.id);
  state.inspectTimer = setInterval(() => pollInspect(run.id), 2000);
  listen(`/api/runs/${run.id}/events`, () => {
    closeStream();
    pollInspect(run.id);
    waitFinished(run.id);
  });
}

// The stream ends at the done / error event; the process may need a few more seconds to shut down (Ray teardown).
async function waitFinished(runId, tries = 0) {
  let run = null;
  try { run = await api("/api/runs/current"); } catch (_) { run = null; }
  if (run && run.id === runId && run.status === "running" && tries < 150) {
    setTimeout(() => waitFinished(runId, tries + 1), 2000);
    return;
  }
  setRunning(false);
  if (!run || run.id !== runId || state.sessionName !== runId) return;
  if (run.status === "done") setStatus("Run finished: every hospital kept its patients; only masked updates moved.", "ok");
  else if (run.status === "stopped") setStatus("Run stopped.");
  else if (run.status === "failed" && !$("status").classList.contains("err")) setStatus("The run failed. Its stderr.log in the run folder says why.", "err");
}

async function startRun(e) {
  e.preventDefault();
  const params = formParams();
  setStatus("Starting…");
  $("start-btn").disabled = true;
  try {
    const run = await api("/api/runs", { method: "POST", body: JSON.stringify(params) });
    followRun(run);
  } catch (err) {
    setRunning(false);
    setStatus(err.status === 409 ? "A run is already active. Stop it first." : `Not started: ${err.message}`, "err");
  }
}

async function stopRun() {
  $("stop-btn").disabled = true;
  setStatus("Stopping the run and all its processes…");
  try {
    const run = await api("/api/runs/current/stop", { method: "POST" });
    setStatus(`Run ${run.status}.`);
  } catch (err) {
    setStatus(`Could not stop: ${err.message}`, "err");
  }
  setRunning(false);
}

/* ---------------------------------------------------------------------------------------------------- replays */

async function loadReplays() {
  let list = [];
  try { list = await api("/api/replays"); } catch (_) { list = []; }
  const sel = $("f-replay");
  sel.innerHTML = "";
  for (const r of list) {
    const opt = document.createElement("option");
    opt.value = r.name;
    const c = r.config || {};
    opt.textContent = `${r.name} · ${c.rounds ?? "?"} rounds · ${c.secagg ? "SecAgg+" : "no SecAgg+"} · ${ATTACK_LABELS[c.attack] || c.attack || ""}`;
    sel.appendChild(opt);
  }
  return list;
}

async function playReplay(name, speed, preview = false) {
  closeStream();
  resetSession("replay", name);
  try { state.inspect = await api(`/api/replays/${encodeURIComponent(name)}/inspect`); } catch (_) { state.inspect = null; }
  renderInspector();
  setStatus(preview ? `Preview: replaying the recorded run "${name}" (no training). Start a live run any time.`
                    : `Replaying "${name}" at ${speed}× (no training).`);
  listen(`/api/replays/${encodeURIComponent(name)}/events?speed=${speed}`, () => {
    if (state.sessionName === name && !state.running) setStatus(`Replay of "${name}" finished.`, "ok");
  });
}

/* ---------------------------------------------------------------------------------------------------- rendering: training */

function renderTraining() {
  const rounds = state.rounds;
  const last = rounds[rounds.length - 1];
  $("k-acc").textContent = last ? pct(last.acc) : "–";
  $("k-f1").textContent = last ? num(last.f1) : "–";
  const eps = last ? last.epsilon : null;
  const noNoise = eps && eps.some((e) => e == null) && last.round > 0;
  const maxEps = eps && !noNoise ? Math.max(...eps.filter((e) => e != null)) : null;
  $("k-eps").textContent = !last ? "–" : noNoise ? "∞" : num(maxEps);
  const band = !last ? null : epsBand(noNoise ? null : maxEps);
  $("k-eps-band").innerHTML = band ? `<span class="band ${band.cls}">${esc(band.text)}</span>` : "worst hospital's privacy loss so far";
  $("k-round").textContent = last ? `${last.round} / ${state.totalRounds || last.total_rounds}` : "–";
  $("k-round-note").textContent = last && last.round === 0 ? "round 0 = untrained starting model" : "rounds done of planned";
  $("eps-note").hidden = !noNoise;

  if (!charts.acc) return;
  const total = Math.max(state.totalRounds, last ? last.round : 0);
  const labels = Array.from({ length: total + 1 }, (_, i) => i);
  const byRound = new Map(rounds.map((r) => [r.round, r]));
  const series = (fn) => labels.map((i) => (byRound.has(i) ? fn(byRound.get(i)) : null));
  const acc = charts.acc;
  acc.data.labels = labels;
  acc.data.datasets[0].data = series((r) => r.acc);
  for (let h = 0; h < 4; h++) acc.data.datasets[1 + h].data = series((r) => r.per_hospital_acc[h]);
  const b = state.baseline;
  const localMean = b ? b.local_per_hospital_acc.reduce((s, v) => s + (v || 0), 0) / b.local_per_hospital_acc.length : null;
  acc.data.datasets[5].data = labels.map(() => (b ? b.central_acc : null));
  acc.data.datasets[6].data = labels.map(() => localMean);
  acc.update();
  $("c-acc").setAttribute("aria-label", last
    ? `Accuracy per round: global model ${pct(last.acc)} after round ${last.round}` + (b ? `, centralized reference ${pct(b.central_acc)}` : "")
    : "Accuracy per round: no data yet");

  const epsChart = charts.eps;
  epsChart.data.labels = labels;
  for (let h = 0; h < 4; h++) epsChart.data.datasets[h].data = series((r) => r.epsilon[h]);
  epsChart.update();
  $("c-eps").setAttribute("aria-label", last ? `Epsilon per hospital after round ${last.round}: ` +
    HOSPITAL_LABELS.map((n, i) => `${n} ${num(last.epsilon[i])}`).join(", ") : "Epsilon per hospital: no data yet");

  renderAccTable(labels, byRound);
}

function renderAccTable(labels, byRound) {
  const t = $("t-acc");
  const b = state.baseline;
  let html = "<thead><tr><th scope=\"col\">Round</th><th scope=\"col\">Global</th>" +
    HOSPITAL_LABELS.map((n) => `<th scope="col">${n}</th>`).join("") + "<th scope=\"col\">Max ε</th></tr></thead><tbody>";
  for (const i of labels) {
    const r = byRound.get(i);
    if (!r) continue;
    const e = r.epsilon.some((x) => x == null) ? "∞" : num(Math.max(...r.epsilon));
    html += `<tr><td>${r.round}</td><td>${pct(r.acc)}</td>${r.per_hospital_acc.map((a) => `<td>${pct(a)}</td>`).join("")}<td>${e}</td></tr>`;
  }
  if (b) {
    html += `<tr><td>Centralized</td><td>${pct(b.central_acc)}</td><td colspan="5"></td></tr>`;
    html += `<tr><td>Local-only</td><td></td>${b.local_per_hospital_acc.map((a) => `<td>${pct(a)}</td>`).join("")}<td></td></tr>`;
  }
  t.innerHTML = html + "</tbody>";
}

/* ---------------------------------------------------------------------------------------------------- rendering: inspector */

function sizeCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 300;
  // The logical height comes from the HTML height attribute, but canvas.height (below) overwrites that same attribute with the
  // pixel size: keep the original once, or every redraw multiplies the height by devicePixelRatio again.
  if (!canvas.dataset.h) canvas.dataset.h = canvas.getAttribute("height") || "90";
  const h = Number(canvas.dataset.h) || 90;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function emptyCanvas(canvas, text) {
  const { ctx, w, h } = sizeCanvas(canvas);
  ctx.fillStyle = css("--muted");
  ctx.font = `13px ${css("--font")}`;
  ctx.textAlign = "center";
  ctx.fillText(text, w / 2, h / 2 + 4);
}

// Signed values as bars around a zero line (model weights).
function drawSigned(canvas, values, color) {
  const { ctx, w, h } = sizeCanvas(canvas);
  const m = Math.max(1e-9, ...values.map((v) => Math.abs(v || 0)));
  const mid = h / 2, bw = w / values.length;
  ctx.fillStyle = color;
  values.forEach((v, i) => {
    const bh = ((v || 0) / m) * (mid - 4);
    ctx.fillRect(i * bw, bh >= 0 ? mid - bh : mid, Math.max(bw - 0.3, 0.6), Math.max(Math.abs(bh), 0.6));
  });
  ctx.fillStyle = css("--border");
  ctx.fillRect(0, mid, w, 1);
  ctx.fillStyle = css("--muted");
  ctx.font = `11px ${css("--mono")}`;
  ctx.textAlign = "left";
  ctx.fillText(`±${m.toFixed(2)}`, 2, 11);
}

// Unsigned 32-bit values scaled to [0, 1] (what left the hospital).
function drawUnsigned(canvas, values, color) {
  const { ctx, w, h } = sizeCanvas(canvas);
  const bw = w / values.length;
  ctx.fillStyle = color;
  values.forEach((v, i) => {
    const bh = (v / 4294967296) * (h - 6);
    ctx.fillRect(i * bw, h - bh, Math.max(bw - 0.3, 0.6), bh);
  });
  ctx.fillStyle = css("--border");
  ctx.fillRect(0, h - 1, w, 1);
}

function drawHistogram(canvas, masked, color) {
  const { ctx, w, h } = sizeCanvas(canvas);
  const vals = masked.slice(1);
  const counts = new Array(32).fill(0);
  vals.forEach((v) => { counts[Math.min(31, Math.floor(v / 134217728))] += 1; });
  const expected = vals.length / 32;
  const top = Math.max(expected * 2, ...counts);
  const pad = 18, bw = w / 32;
  ctx.fillStyle = color;
  counts.forEach((c, i) => {
    const bh = (c / top) * (h - pad - 4);
    ctx.fillRect(i * bw + 1, h - pad - bh, Math.max(bw - 2, 1), bh);
  });
  const ey = h - pad - (expected / top) * (h - pad - 4);
  ctx.strokeStyle = css("--text-2");
  ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(0, ey); ctx.lineTo(w, ey); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = css("--muted");
  ctx.font = `11px ${css("--font")}`;
  ctx.textAlign = "right";
  ctx.fillText(`even fill: ${expected.toFixed(1)} per bin`, w - 2, Math.max(ey - 5, 11));
  ctx.textAlign = "left";
  ctx.fillText("0", 0, h - 4);
  ctx.textAlign = "right";
  ctx.fillText("2³²", w - 2, h - 4);
}

function renderInspector() {
  const ins = state.inspect;
  const lastRound = state.rounds.length ? state.rounds[state.rounds.length - 1].round : -1;
  // A replay shows the inspected round only once the playback reaches it; a live run shows files as they appear.
  const ready = ins && (state.sessionKind !== "replay" || lastRound >= state.inspectRound);
  const meta = ready ? ins.meta : null;
  const secagg = meta ? meta.secagg : state.baseline && state.baseline.config ? state.baseline.config.secagg : null;
  $("sv-round").textContent = meta ? `round ${meta.round} · ${meta.n_examples} training patients at Cleveland` : "";

  if (ready && ins.plain) {
    drawSigned($("v-plain"), ins.plain, css("--teal"));
    $("v-plain").setAttribute("aria-label", `Plain update: ${ins.plain.length} weights`);
  } else emptyCanvas($("v-plain"), "waiting for the inspected round");

  if (ready && ins.masked) {
    drawUnsigned($("v-masked"), ins.masked.slice(1), css("--gold"));
    $("masked-cap").textContent = `The same update after SecAgg+ masking: ${ins.masked.length - 1} numbers that look random.`;
    $("v-masked").setAttribute("aria-label", "Masked update: values spread evenly over the whole 32-bit range");
    drawHistogram($("v-hist"), ins.masked, css("--gold"));
  } else if (ready && secagg === 0) {
    drawSigned($("v-masked"), ins.plain || [], css("--bad"));
    $("masked-cap").textContent = "SecAgg+ is off: the server receives the plain update above, every weight readable.";
    emptyCanvas($("v-hist"), "nothing masked: SecAgg+ is off");
  } else {
    emptyCanvas($("v-masked"), "waiting for the inspected round");
    $("masked-cap").textContent = "The same update after SecAgg+ masking.";
    emptyCanvas($("v-hist"), "waiting for the masked update");
  }

  if (ready && ins.aggregate) drawSigned($("v-agg"), ins.aggregate, css("--teal"));
  else emptyCanvas($("v-agg"), "waiting for the merge");

  const v = $("chi-verdict");
  const chi = ins && ins.chi2;
  if (ready && chi && chi.stat != null) {
    const ok = chi.stat < chi.critical_99;
    v.className = "verdict " + (ok ? "safe" : "exposed");
    v.innerHTML = ok
      ? `<b>Indistinguishable from random noise.</b> χ² = ${num(chi.stat, 1)} is below ${num(chi.critical_99, 1)} (99% level, ${chi.df} degrees of freedom): the server cannot read this hospital's update.`
      : `<b>Not uniform.</b> χ² = ${num(chi.stat, 1)} exceeds ${num(chi.critical_99, 1)} (99% level): these numbers carry visible structure.`;
  } else if (ready && secagg === 0) {
    v.className = "verdict exposed";
    v.innerHTML = "<b>Exposed.</b> Without SecAgg+ the server sees each hospital's update in the clear.";
  } else {
    v.className = "verdict";
    v.textContent = "Waiting for the inspected round…";
  }
  renderTimeline(ready ? ins.stages : null);
}

function renderTimeline(stages) {
  const el = $("timeline");
  const all = (stages || []).flat().filter((s) => typeof s.t === "number");
  if (!all.length) {
    el.innerHTML = "<p class=\"muted\">Each hospital's steps appear here during the inspected round.</p>";
    return;
  }
  const t0 = Math.min(...all.map((s) => s.t)), t1 = Math.max(...all.map((s) => s.t));
  const span = Math.max(t1 - t0, 1e-6);
  let html = "";
  stages.forEach((rows, pid) => {
    const pts = rows.filter((s) => typeof s.t === "number").map((s) => ({ ...s, x: 4 + ((s.t - t0) / span) * 88 }));
    // Steps closer than 6% of the track (e.g. trained -> masked -> sent within milliseconds) share one label, so labels never overlap.
    const groups = [];
    pts.forEach((p) => {
      const g = groups[groups.length - 1];
      if (g && p.x - g[0].x < 6) g.push(p); else groups.push([p]);
    });
    const dots = pts.map((s) =>
      `<span class="tl-dot ${esc(s.stage)}" style="left:${s.x}%" title="${esc(s.stage)} at +${(s.t - t0).toFixed(2)} s"></span>`).join("") +
      groups.map((g, k) => {
        const cls = g.some((s) => s.stage === "masked") ? "masked" : esc(g[0].stage);
        return `<span class="tl-label ${cls}" style="left:${g[0].x}%${k % 2 === 1 ? ";bottom:-26px" : ""}">` +
          `${g.map((s) => esc(s.stage)).join(" → ")}</span>`;
      }).join("");
    const desc = rows.map((s) => `${s.stage} at +${(s.t - t0).toFixed(2)} s`).join(", ") || "no steps recorded";
    html += `<div class="tl-row" role="listitem" aria-label="${esc(HOSPITAL_LABELS[pid])}: ${esc(desc)}">` +
      `<span class="tl-name"><i style="background:${hospitalColor(pid)}"></i>${HOSPITAL_LABELS[pid]}</span>` +
      `<div class="tl-track" style="margin-bottom:18px">${dots}</div></div>`;
  });
  html += `<div class="tl-axis" aria-hidden="true"><span>0 s</span><span>+${span.toFixed(1)} s</span></div>`;
  el.innerHTML = html;
}

/* ---------------------------------------------------------------------------------------------------- rendering: attacks */

function renderAttackCards() {
  const a = state.attack;
  const p = $("poison-body"), m = $("mia-body");
  if (!a) {
    const msg = "<p class=\"muted\">Appears after the last round.</p>";
    p.innerHTML = msg; m.innerHTML = msg;
    return;
  }
  const attacker = HOSPITAL_LABELS[a.attacker] || a.attacker_name;
  if (a.attack === "none") {
    p.innerHTML = `<p>No hospital attacked in this run. Average accuracy of the hospitals other than ${esc(attacker)}: <b>${pct(a.honest_acc)}</b>.</p>`;
  } else {
    const facts = [
      `Attack: <b>${esc(ATTACK_LABELS[a.attack] || a.attack)}</b> by ${esc(attacker)}${a.boost ? ` (update scaled ×${num(a.boost, 1)})` : ""}`,
      `Averaging: <b>${esc(AGG_LABELS[a.agg] || a.agg)}</b>, SecAgg+ ${a.secagg ? "on" : "off"}`,
    ];
    if (a.clipped_rounds != null) facts.push(`SecAgg+ clipping cut the attacker's update in <b>${a.clipped_rounds}</b> round(s)` +
      (a.max_scaled_attacker != null ? ` (largest scaled weight ${num(a.max_scaled_attacker)})` : ""));
    p.innerHTML = `<div class="duo">
        <div class="stat"><b>${pct(a.honest_acc)}</b><span>honest hospitals' accuracy</span><div class="meter"><i style="width:${(a.honest_acc || 0) * 100}%"></i></div></div>
        <div class="stat"><b>${pct(a.attacker_acc)}</b><span>accuracy on ${esc(attacker)}'s own patients</span><div class="meter"><i style="width:${(a.attacker_acc || 0) * 100}%"></i></div></div>
      </div><ul class="facts">${facts.map((f) => `<li>${f}</li>`).join("")}</ul>`;
  }
  const target = HOSPITAL_LABELS[a.mia_hospital] || a.mia_hospital_name;
  if (a.mia_auc_update == null && a.mia_auc_global == null) {
    m.innerHTML = `<p class="muted">The membership test did not run in this run (it needs the side channel folder).</p>`;
  } else {
    const verdict = (auc) => auc == null ? "–" : auc < 0.6 ? "close to a coin flip" : auc < 0.75 ? "some leakage" : "clear leakage";
    m.innerHTML = `<p>The curious server takes ${a.mia_n ?? "?"} of ${esc(target)}'s training patients and as many unseen ones, and guesses which were used.
        <b>AUC 0.5 = coin flip</b> (nothing learned), 1.0 = always right.</p>
      <div class="duo">
        <div class="stat"><b>${num(a.mia_auc_update)}</b><span>AUC from ${esc(target)}'s own last update: ${verdict(a.mia_auc_update)}</span>
          <div class="meter gold"><i style="width:${Math.max(0, ((a.mia_auc_update || 0.5) - 0.5) * 200)}%"></i></div></div>
        <div class="stat"><b>${num(a.mia_auc_global)}</b><span>AUC from the final shared model: ${verdict(a.mia_auc_global)}</span>
          <div class="meter gold"><i style="width:${Math.max(0, ((a.mia_auc_global || 0.5) - 0.5) * 200)}%"></i></div></div>
      </div>
      <p class="facts">Advantage (true-positive minus false-positive rate): update ${num(a.mia_adv_update)}, shared model ${num(a.mia_adv_global)}.
        With SecAgg+ on, the server never even holds the single-hospital update; the first number is the worst case without it.</p>`;
  }
}

/* ---------------------------------------------------------------------------------------------------- rendering: sweeps */

async function loadSweeps() {
  try { state.sweeps = await api("/api/sweeps"); } catch (_) { state.sweeps = null; }
  renderSweeps();
}

function renderSweeps() {
  const groups = (state.sweeps && state.sweeps.groups) || [];
  $("sw-tag").textContent = state.sweeps && state.sweeps.n_rows ? `${state.sweeps.n_rows} finished runs from ${state.sweeps.files.join(", ")}` : "";
  if (!charts.sweep) return;

  // Privacy vs accuracy: attack-free groups; one line per (strategy, SecAgg+, averaging). Fixed color order, max 3 lines.
  const clean = groups.filter((g) => g.attack === "none" && g.fed_acc.mean != null);
  const lines = new Map();
  for (const g of clean) {
    const key = `${g.strategy === "fedprox" ? "FedProx" : AGG_LABELS[g.agg] || g.agg}${g.secagg ? " + SecAgg+" : ""}`;
    if (!lines.has(key)) lines.set(key, []);
    lines.get(key).push(g);
  }
  const palette = [css("--teal"), css("--h0"), css("--h1")];
  charts.sweep.data.datasets = [...lines.entries()].slice(0, 3).map(([label, gs], i) => {
    gs.sort((a, b) => a.noise - b.noise);
    return { label, data: gs.map((g) => ({ x: g.noise, y: g.fed_acc.mean, g })), errors: gs.map((g) => g.fed_acc.std),
      borderColor: palette[i], backgroundColor: palette[i], borderWidth: 2, pointRadius: 4, pointHoverRadius: 6 };
  });
  charts.sweep.update();
  $("sweep-empty").hidden = clean.length > 0;
  $("c-sweep").setAttribute("aria-label", clean.length ? `Privacy versus accuracy over ${clean.length} settings` : "Privacy versus accuracy: no sweep results");

  // Attacks: honest accuracy by attack (x) and averaging rule (series).
  const atk = groups.filter((g) => g.honest_acc.mean != null);
  const attacks = Object.keys(ATTACK_LABELS).filter((k) => atk.some((g) => g.attack === k));
  const aggs = Object.keys(AGG_LABELS).filter((k) => atk.some((g) => g.agg === k));
  const aggColor = [css("--teal"), css("--h0"), css("--h1")];
  charts.attack.data.labels = attacks.map((k) => ATTACK_LABELS[k]);
  charts.attack.data.datasets = aggs.map((agg, i) => {
    const pick = (k) => {
      const gs = atk.filter((g) => g.attack === k && g.agg === agg);
      if (!gs.length) return { m: null, s: null };
      const best = gs.reduce((a, b) => (b.honest_acc.n > a.honest_acc.n ? b : a));
      return { m: best.honest_acc.mean, s: best.honest_acc.std };
    };
    const vals = attacks.map(pick);
    return { label: AGG_LABELS[agg], data: vals.map((v) => v.m), errors: vals.map((v) => v.s), backgroundColor: aggColor[i % 3],
      borderColor: css("--text-2"), borderWidth: 0, borderRadius: 4, borderSkipped: "start", maxBarThickness: 46 };
  });
  charts.attack.update();
  $("attack-empty").hidden = atk.length > 0;

  const rows = groups.filter((g) => g.attack !== "none" || g.mia_auc_update.mean != null);
  $("t-attack").innerHTML = rows.length ? "<thead><tr><th scope=\"col\">Attack</th><th scope=\"col\">Averaging</th><th scope=\"col\">SecAgg+</th>" +
    "<th scope=\"col\">Noise</th><th scope=\"col\">Honest acc.</th><th scope=\"col\">Attacker acc.</th><th scope=\"col\">MIA AUC update</th>" +
    "<th scope=\"col\">MIA AUC model</th><th scope=\"col\">Runs</th></tr></thead><tbody>" + rows.map((g) =>
      `<tr><td>${esc(ATTACK_LABELS[g.attack] || g.attack)}</td><td>${esc(AGG_LABELS[g.agg] || g.agg)}</td><td>${g.secagg ? "on" : "off"}</td>` +
      `<td>${num(g.noise, 1)}</td><td>${pct(g.honest_acc.mean)} ± ${pct(g.honest_acc.std)}</td><td>${pct(g.attacker_acc.mean)}</td>` +
      `<td>${num(g.mia_auc_update.mean)}</td><td>${num(g.mia_auc_global.mean)}</td><td>${g.n}</td></tr>`).join("") + "</tbody>"
    : "<tbody><tr><td>No attack results yet.</td></tr></tbody>";
}

/* ---------------------------------------------------------------------------------------------------- console wiring */

function syncConsole() {
  const f = $("run-form");
  $("o-noise").textContent = Number(f.noise.value).toFixed(1);
  const median = f.agg.value !== "fedavg";
  const wasMedian = state.wasMedian;
  if (median) {
    f.secagg.checked = false;
    f.secagg.disabled = true;
    f.strategy.value = "fedavg";
    f.strategy.disabled = true;
    $("h-secagg").textContent = "Off: the median needs every hospital's update in the clear, and SecAgg+ hides exactly that.";
  } else {
    if (wasMedian) f.secagg.checked = true; // leaving the median: protection comes back on by default
    f.secagg.disabled = false;
    f.strategy.disabled = false;
    $("h-secagg").textContent = "Median needs every update in the clear, so it turns SecAgg+ off.";
  }
  $("t-secagg").textContent = f.secagg.checked ? "On: server sees only the sum" : "Off: server sees every update";
  f.attacker.disabled = f.attack.value === "none";
  state.wasMedian = median;
}

function setSource(source) {
  state.source = source;
  $("run-form").hidden = source !== "live";
  $("replay-form").hidden = source !== "replay";
  setRunning(state.running);
}

function fillConsole(cfg) {
  const f = $("run-form");
  f.noise.max = cfg.limits.noise_max;
  f.rounds.max = cfg.limits.rounds_max;
  f.seed.max = cfg.limits.seed_max;
  const sel = f.attacker;
  sel.innerHTML = "";
  cfg.hospitals.forEach((name, i) => {
    if (i === 0) return; // hospital 0 is the membership-test target
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = HOSPITAL_LABELS[i] || name;
    sel.appendChild(opt);
  });
  sel.value = "3";
  $("mode-badge").textContent = cfg.mode === "deploy" ? "deployment" : "simulation";
  $("mode-badge").title = cfg.mode === "deploy" ? "Runs go to the Flower SuperLink (flwr run)" : "Runs use Flower's simulation engine on this machine";
}

function watchNav() {
  const links = [...document.querySelectorAll("nav a")];
  const obs = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#${e.target.id}`));
    });
  }, { rootMargin: "-40% 0px -55% 0px" });
  document.querySelectorAll("main section[id]").forEach((s) => obs.observe(s));
}

async function init() {
  applyTheme(storedTheme());
  initCharts();
  $("theme-btn").addEventListener("click", toggleTheme);
  const f = $("run-form");
  f.addEventListener("input", syncConsole);
  f.addEventListener("change", syncConsole);
  f.addEventListener("submit", startRun);
  $("stop-btn").addEventListener("click", stopRun);
  document.querySelectorAll("input[name=source]").forEach((r) => r.addEventListener("change", () => setSource(r.value)));
  $("play-btn").addEventListener("click", () => {
    const name = $("f-replay").value;
    if (name) playReplay(name, Number($("f-speed").value));
  });
  let resizeTimer = null;
  window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderInspector, 150); });
  watchNav();

  try {
    state.config = await api("/api/config");
    fillConsole(state.config);
  } catch (err) {
    setStatus(`Cannot reach the dashboard server: ${err.message}`, "err");
  }
  syncConsole();
  renderInspector();
  loadSweeps();
  const replays = await loadReplays();

  let current = null;
  try { current = await api("/api/runs/current"); } catch (_) { current = null; }
  if (current && current.status === "running") {
    followRun(current);
  } else if (replays.length) {
    // Golden preview: every panel gets real data at once, without training. Plays the whole run in about 20 seconds.
    const pick = replays.find((r) => r.name === "golden") || replays[0];
    const speed = Math.min(1000, Math.max(1, Math.ceil((pick.duration || 20) / 20)));
    $("f-replay").value = pick.name;
    playReplay(pick.name, speed, true);
  } else {
    setStatus("Ready. Choose settings and start a run.");
  }
  setRunning(!!(current && current.status === "running"));
}

document.addEventListener("DOMContentLoaded", init);
