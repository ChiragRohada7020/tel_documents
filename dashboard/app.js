/* ============================================================
   J.A.R.V.I.S. — Document Vault Dashboard Logic
   Vanilla JS · talks to the FastAPI backend (web_app.py)
   ============================================================ */

"use strict";

/* ---------------- helpers ---------------- */
const $ = (id) => document.getElementById(id);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&").replace(/</g, "<").replace(/>/g, ">")
    .replace(/"/g, """).replace(/'/g, "&#39;");
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function fmtDate(d) {
  return d.toLocaleDateString(undefined, { weekday: "short", year: "numeric", month: "short", day: "numeric" }).toUpperCase();
}

function daysUntil(iso) {
  if (!iso) return null;
  const d = new Date(iso + "T00:00:00");
  if (isNaN(d)) return null;
  return Math.ceil((d - new Date(new Date().toDateString())) / 86400000);
}

/* ---------------- state ---------------- */
const LS = {
  base: "jarvis_api_base",
  key: "jarvis_api_key",
  user: "jarvis_user_id",
  sound: "jarvis_sound",
};

const state = {
  base: localStorage.getItem(LS.base) || "",
  apiKey: localStorage.getItem(LS.key) || "",
  userId: localStorage.getItem(LS.user) || "",
  sound: localStorage.getItem(LS.sound) !== "off",
  docs: [],
  bootedAt: Date.now(),
  currentDocId: null,
  confirmAction: null,
};

/* ---------------- audio (WebAudio blips) ---------------- */
let _audioCtx = null;
function beep(freq = 1200, dur = 0.05, type = "square", gain = 0.04) {
  if (!state.sound) return;
  try {
    _audioCtx = _audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const o = _audioCtx.createOscillator();
    const g = _audioCtx.createGain();
    o.type = type; o.frequency.value = freq;
    g.gain.setValueAtTime(gain, _audioCtx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.0001, _audioCtx.currentTime + dur);
    o.connect(g); g.connect(_audioCtx.destination);
    o.start(); o.stop(_audioCtx.currentTime + dur);
  } catch (_) { /* audio unavailable */ }
}
const sfx = {
  click: () => beep(1400, 0.04),
  ok: () => { beep(880, 0.07); setTimeout(() => beep(1320, 0.09), 70); },
  err: () => beep(180, 0.22, "sawtooth", 0.06),
  send: () => beep(1000, 0.05, "triangle"),
};

/* ---------------- API layer ---------------- */
class ApiError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

async function api(path, { method = "GET", body = null, form = null, auth = true } = {}) {
  const base = state.base.replace(/\/+$/, "");
  const headers = {};
  if (auth && state.apiKey) headers["X-API-Key"] = state.apiKey;

  const t0 = performance.now();
  let res;
  try {
    res = await fetch(base + path, {
      method,
      headers,
      body: form ? form : body ? JSON.stringify(body) : undefined,
    });
  } catch (e) {
    throw new ApiError(0, "Network unreachable — is the server running?");
  }
  const ms = Math.round(performance.now() - t0);
  setLatency(ms);

  if (!res.ok) {
    let detail = res.statusText;
    try { const j = await res.json(); detail = j.detail || detail; } catch (_) {}
    if (res.status === 401) { openModal("modal-settings"); }
    throw new ApiError(res.status, detail);
  }
  const ct = res.headers.get("content-type") || "";
  return ct.includes("json") ? res.json() : res.text();
}

/* ---------------- toasts ---------------- */
function toast(msg, kind = "info", ttl = 3800) {
  const el = document.createElement("div");
  el.className = "toast" + (kind !== "info" ? " " + kind : "");
  el.textContent = msg;
  $("toasts").appendChild(el);
  setTimeout(() => { el.classList.add("out"); setTimeout(() => el.remove(), 350); }, ttl);
}

/* ---------------- status chips / diagnostics ---------------- */
function setChip(id, on, label) {
  const chip = $(id);
  chip.classList.toggle("on", !!on);
  chip.classList.toggle("off", !on);
  chip.querySelector("b").textContent = label;
}
function setLatency(ms) {
  $("chip-latency").querySelector("b").textContent = ms + " ms";
  $("diag-latency").textContent = ms + " ms";
}

async function healthCheck() {
  try {
    await api("/health", { auth: false });
    setChip("chip-api", true, "ONLINE");
  } catch (_) {
    setChip("chip-api", false, "OFFLINE");
  }
}

/* ---------------- boot sequence ---------------- */
const BOOT_LINES = [
  "J.A.R.V.I.S. KERNEL v2.6 …………………… LOADED",
  "MOUNTING DOCUMENT VAULT INTERFACE …… OK",
  "CALIBRATING ARC REACTOR ARRAY ………… OK",
  "LINKING GROQ AI CORE ……………………… STANDBY",
  "SCANNING TELEGRAM ARCHIVE BRIDGE …… OK",
  "ALL SYSTEMS NOMINAL. WELCOME BACK.",
];

async function runBoot() {
  const log = $("boot-log");
  const fill = $("boot-fill");
  log.textContent = "";
  for (let i = 0; i < BOOT_LINES.length; i++) {
    const line = BOOT_LINES[i];
    for (let c = 0; c <= line.length; c += 3) {
      log.textContent = log.textContent.split("\n").slice(0, i).concat([line.slice(0, c)]).join("\n");
      await sleep(4);
    }
    log.textContent = log.textContent.split("\n").slice(0, i).concat([line]).join("\n") + "\n";
    fill.style.width = Math.round(((i + 1) / BOOT_LINES.length) * 100) + "%";
    beep(900 + i * 90, 0.03);
    await sleep(90);
  }
  await sleep(420);
  $("boot-screen").classList.add("fade-out");
  setTimeout(() => $("boot-screen").classList.add("hidden"), 750);
  $("app").classList.remove("hidden");
  afterBoot();
}

/* ---------------- init ---------------- */
function startClock() {
  const tick = () => {
    const n = new Date();
    $("clock").textContent = n.toLocaleTimeString(undefined, { hour12: false });
    $("datestamp").textContent = fmtDate(n);
    const up = Math.floor((Date.now() - state.bootedAt) / 1000);
    const hh = String(Math.floor(up / 3600)).padStart(2, "0");
    const mm = String(Math.floor((up % 3600) / 60)).padStart(2, "0");
    const ss = String(up % 60).padStart(2, "0");
    $("diag-uptime").textContent = `${hh}:${mm}:${ss}`;
  };
  tick();
  setInterval(tick, 1000);
}

function afterBoot() {
  startClock();
  bindEvents();
  updateSoundIcon();
  healthCheck();
  setInterval(healthCheck, 15000);

  if (!state.apiKey || !state.userId) {
    toast("Connection not configured. Opening settings…", "warn");
    openModal("modal-settings");
  } else {
    loadVault().catch((e) => toast(e.message, "err"));
  }
}

/* ---------------- vault data ---------------- */
async function loadVault() {
  if (!state.userId) return;
  const data = await api(`/api/v1/documents?user_id=${encodeURIComponent(state.userId)}&limit=50`);
  state.docs = data.items || [];
  renderStats();
  renderCategories();
  renderDocs();
  setChip("chip-vault", true, (data.total ?? state.docs.length) + " FILES");
  $("last-sync").textContent = "LAST SYNC: " + new Date().toLocaleTimeString(undefined, { hour12: false });
  sfx.ok();
}

function renderStats() {
  const docs = state.docs;
  const photos = docs.filter((d) => d.file_type === "photo").length;
  const expiring = docs.filter((d) => {
    const dd = daysUntil(d.expiry_date);
    return dd !== null && dd >= 0 && dd <= 30;
  }).length;
  $("stat-total").textContent = docs.length;
  $("stat-docs").textContent = docs.length - photos;
  $("stat-photos").textContent = photos;
  $("stat-expiring").textContent = expiring;
  $("diag-user").textContent = state.userId ? "#" + state.userId : "NOT SET";
}

function renderCategories() {
  const counts = {};
  for (const d of state.docs) {
    const c = (d.category || "uncategorized").trim() || "uncategorized";
    counts[c] = (counts[c] || 0) + 1;
  }
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 10);
  const wrap = $("category-bars");
  if (!entries.length) {
    wrap.innerHTML = '<div class="empty-hint">No categories detected yet.<br />Upload documents to populate the matrix.</div>';
    return;
  }
  const max = Math.max(...entries.map((e) => e[1]));
  wrap.innerHTML = entries.map(([name, n]) => `
    <div class="cat-row">
      <div class="cat-name">${esc(name)}</div>
      <div class="cat-track"><div class="cat-fill" style="width:${Math.round((n / max) * 100)}%"></div></div>
      <div class="cat-count">${n}</div>
    </div>`).join("");
}

function renderDocs() {
  const list = $("doc-list");
  const q = ($("doc-filter").value || "").toLowerCase().trim();
  const docs = state.docs.filter((d) => {
    if (!q) return true;
    const hay = [d.ai_title, d.filename, d.description, d.category].join(" ").toLowerCase();
    return hay.includes(q);
  });

  if (!docs.length) {
    list.innerHTML = `<div class="empty-hint">${q ? "No documents match this filter." :
      "Archive empty. Configure the connection and upload your first document."}</div>`;
    return;
  }

  list.innerHTML = docs.map((d) => {
    const icon = d.file_type === "photo" ? "🖼️" : "📄";
    const name = d.ai_title || d.filename;
    const dd = daysUntil(d.expiry_date);
    let expiryTag = "";
    if (dd !== null) {
      const cls = dd <= 7 ? "expiry soon" : "expiry";
      const label = dd < 0 ? "EXPIRED" : dd === 0 ? "EXPIRES TODAY" : `${dd}D LEFT`;
      expiryTag = `<span class="tag ${cls}">⏳ ${label}</span>`;
    }
    const catTag = d.category ? `<span class="tag cat">${esc(d.category)}</span>` : "";
    return `
      <div class="doc-card" data-id="${esc(d._id)}">
        <div class="doc-top">
          <span class="doc-icon">${icon}</span>
          <span class="doc-name">${esc(name)}</span>
        </div>
        <div class="doc-tags">${catTag}${expiryTag}</div>
      </div>`;
  }).join("");

  $$(".doc-card").forEach((card) => {
    card.addEventListener("click", () => openDocDetail(card.dataset.id));
  });
}

/* ---------------- semantic search ---------------- */
async function runSemanticSearch() {
  const q = ($("doc-filter").value || "").trim();
  if (!q) { toast("Type a query in the filter box first.", "warn"); return; }
  if (!state.userId) { openModal("modal-settings"); return; }
  const panel = $("search-results");
  panel.classList.remove("hidden");
  panel.innerHTML = '<div class="sr-item"><div class="sr-src">SCANNING VAULT…</div></div>';
  sfx.send();
  try {
    const data = await api(`/api/v1/search?user_id=${encodeURIComponent(state.userId)}&q=${encodeURIComponent(q)}&limit=5`);
    const items = data.items || [];
    if (!items.length) {
      panel.innerHTML = '<div class="sr-item"><div class="sr-src">NO MATCHES IN VAULT</div></div>';
      return;
    }
    panel.innerHTML =
      '<div class="sr-close" id="sr-close">✕ CLOSE SCAN</div>' +
      items.map((it) => {
        const score = typeof it.score === "number"
          ? (it.score <= 1 ? (it.score * 100).toFixed(1) + "%" : it.score.toFixed(2))
          : "—";
        return `
          <div class="sr-item">
            <div class="sr-src">▸ ${esc(it.source || "unknown")} · MATCH ${score}</div>
            <div class="sr-content">${esc(String(it.content || "").slice(0, 220))}…</div>
          </div>`;
      }).join("");
    $("sr-close").addEventListener("click", () => panel.classList.add("hidden"));
    sfx.ok();
  } catch (e) {
    panel.innerHTML = `<div class="sr-item"><div class="sr-src">SCAN FAILED: ${esc(e.message)}</div></div>`;
    sfx.err();
  }
}

/* ---------------- chat ---------------- */
function addMsg(role, text, sources) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  const who = role === "ai" ? "J.A.R.V.I.S." : "OPERATOR";
  wrap.innerHTML = `
    <div class="msg-head">${who}</div>
    <div class="msg-body"></div>
    ${sources && sources.length ? '<div class="msg-sources">' +
      sources.map((s) => `<span class="src-chip">📄 ${esc(s)}</span>`).join("") + "</div>" : ""}`;
  wrap.querySelector(".msg-body").textContent = text;
  $("chat-log").appendChild(wrap);
  $("chat-log").scrollTop = $("chat-log").scrollHeight;
  return wrap.querySelector(".msg-body");
}

async function typewrite(el, text) {
  el.classList.add("typing-cursor");
  const step = text.length > 700 ? 6 : text.length > 250 ? 3 : 1;
  for (let i = 0; i <= text.length; i += step) {
    el.textContent = text.slice(0, i);
    $("chat-log").scrollTop = $("chat-log").scrollHeight;
    await sleep(12);
  }
  el.textContent = text;
  el.classList.remove("typing-cursor");
}

function reactor(mode) {
  const r = $("reactor");
  r.classList.toggle("thinking", mode === "thinking");
  $("reactor-state").textContent = mode === "thinking" ? "PROCESSING" : "STANDBY";
}

async function sendChat(e) {
  e.preventDefault();
  const input = $("chat-input");
  const text = input.value.trim();
  if (!text) return;
  if (!state.apiKey || !state.userId) { openModal("modal-settings"); return; }

  input.value = "";
  addMsg("user", text);
  sfx.send();
  reactor("thinking");
  $("chat-send").disabled = true;

  try {
    const data = await api("/api/v1/chat", {
      method: "POST",
      body: { user_id: Number(state.userId), message: text },
    });
    const body = addMsg("ai", "");
    await typewrite(body, data.answer || "(empty response)");
    if (data.sources && data.sources.length) {
      // attach source chips under the freshly added ai message
      const last = $("chat-log").lastElementChild;
      const div = document.createElement("div");
      div.className = "msg-sources";
      div.innerHTML = data.sources.map((s) => `<span class="src-chip">📄 ${esc(s)}</span>`).join("");
      last.appendChild(div);
    }
    sfx.ok();
  } catch (err) {
    addMsg("ai", "⚠ Transmission failed: " + err.message);
    sfx.err();
  } finally {
    reactor("standby");
    $("chat-send").disabled = false;
    $("chat-input").focus();
  }
}

/* ---------------- document detail ---------------- */
async function openDocDetail(id) {
  if (!state.userId) return;
  sfx.click();
  try {
    const d = await api(`/api/v1/documents/${encodeURIComponent(id)}?user_id=${encodeURIComponent(state.userId)}`);
    state.currentDocId = id;
    $("doc-modal-title").textContent = (d.ai_title || d.filename || "DOCUMENT").toUpperCase();
    const meta = [];
    meta.push(["FILE", d.filename]);
    if (d.category) meta.push(["CATEGORY", d.category]);
    if (d.document_type) meta.push(["TYPE", d.document_type]);
    if (d.expiry_date) meta.push(["EXPIRY", d.expiry_date]);
    if (Array.isArray(d.tags) && d.tags.length) meta.push(["TAGS", d.tags.join(", ")]);
    if (d.description) meta.push(["DESCRIPTION", d.description]);
    $("doc-meta").innerHTML = meta.map(([k, v]) =>
      `<span class="meta-item">${esc(k)}<b>${esc(v)}</b></span>`).join("");
    $("doc-text").textContent = (d.text || "(no indexed text)").slice(0, 4000);
    $("note-input").value = "";
    openModal("modal-doc");
  } catch (e) {
    toast("Could not open document: " + e.message, "err");
    sfx.err();
  }
}

async function saveNote() {
  const note = $("note-input").value.trim();
  if (!note || !state.currentDocId) return;
  try {
    await api(`/api/v1/documents/${encodeURIComponent(state.currentDocId)}/notes`, {
      method: "POST",
      body: { user_id: Number(state.userId), detail: note },
    });
    toast("Note indexed into the vault.", "ok");
    sfx.ok();
    closeModal("modal-doc");
    loadVault().catch(() => {});
  } catch (e) {
    toast("Note failed: " + e.message, "err");
    sfx.err();
  }
}

function askDeleteDoc() {
  if (!state.currentDocId) return;
  state.confirmAction = async () => {
    try {
      await api(`/api/v1/documents/${encodeURIComponent(state.currentDocId)}?user_id=${encodeURIComponent(state.userId)}`, { method: "DELETE" });
      toast("Document purged from vault.", "ok");
      sfx.ok();
      closeModal("modal-doc");
      loadVault().catch(() => {});
    } catch (e) {
      toast("Delete failed: " + e.message, "err");
      sfx.err();
    }
  };
  $("confirm-text").textContent = "This will permanently remove the document metadata and its indexed knowledge. Proceed?";
  openModal("modal-confirm");
}

/* ---------------- upload ---------------- */
async function handleFile(file) {
  if (!file) return;
  if (!state.apiKey || !state.userId) { openModal("modal-settings"); return; }

  const status = $("upload-status");
  status.className = "upload-status";
  status.classList.remove("hidden");
  status.textContent = `▸ TRANSMITTING ${file.name} (${(file.size / 1048576).toFixed(1)} MB)…`;

  const fd = new FormData();
  fd.append("user_id", state.userId);
  fd.append("description", $("upload-desc").value.trim());
  fd.append("file", file);

  try {
    const res = await api("/api/v1/documents/upload", { method: "POST", form: fd });
    status.className = "upload-status ok";
    status.textContent = `▸ INDEXED: ${res.stored_as || res.filename} · ${res.chunks} chunks`;
    toast(`Uploaded & indexed: ${res.stored_as || res.filename}`, "ok");
    $("upload-desc").value = "";
    sfx.ok();
    loadVault().catch(() => {});
    setTimeout(() => status.classList.add("hidden"), 6000);
  } catch (e) {
    status.className = "upload-status error";
    status.textContent = "▸ UPLOAD FAILED: " + e.message;
    toast("Upload failed: " + e.message, "err");
    sfx.err();
  }
}

function bindUpload() {
  const dz = $("dropzone");
  const fi = $("file-input");
  dz.addEventListener("click", () => fi.click());
  fi.addEventListener("change", () => { handleFile(fi.files[0]); fi.value = ""; });
  ["dragenter", "dragover"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("dragover"); }));
  dz.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));
}

/* ---------------- modals ---------------- */
function openModal(id) { $(id).classList.remove("hidden"); }
function closeModal(id) { $(id).classList.add("hidden"); }

function openSettings() {
  $("set-base").value = state.base;
  $("set-key").value = state.apiKey;
  $("set-user").value = state.userId;
  openModal("modal-settings");
}

function saveSettings() {
  state.base = $("set-base").value.trim();
  state.apiKey = $("set-key").value.trim();
  state.userId = String(parseInt($("set-user").value, 10) || "");
  localStorage.setItem(LS.base, state.base);
  localStorage.setItem(LS.key, state.apiKey);
  localStorage.setItem(LS.user, state.userId);
  closeModal("modal-settings");
  toast("Connection parameters stored.", "ok");
  sfx.ok();
  healthCheck();
  if (state.apiKey && state.userId) loadVault().catch((e) => toast(e.message, "err"));
}

/* ---------------- sound toggle ---------------- */
function updateSoundIcon() {
  $("btn-sound").textContent = state.sound ? "🔊" : "🔇";
}
function toggleSound() {
  state.sound = !state.sound;
  localStorage.setItem(LS.sound, state.sound ? "on" : "off");
  updateSoundIcon();
  if (state.sound) sfx.ok();
}

/* ---------------- event binding ---------------- */
function bindEvents() {
  $("chat-form").addEventListener("submit", sendChat);
  $("btn-semantic").addEventListener("click", runSemanticSearch);
  $("doc-filter").addEventListener("keydown", (e) => { if (e.key === "Enter") runSemanticSearch(); });
  $("doc-filter").addEventListener("input", renderDocs);
  $("btn-settings").addEventListener("click", openSettings);
  $("btn-sound").addEventListener("click", toggleSound);
  $("btn-save-settings").addEventListener("click", saveSettings);
  $("btn-add-note").addEventListener("click", saveNote);
  $("note-input").addEventListener("keydown", (e) => { if (e.key === "Enter") saveNote(); });
  $("btn-delete-doc").addEventListener("click", askDeleteDoc);
  $("btn-confirm-yes").addEventListener("click", async () => {
    closeModal("modal-confirm");
    if (state.confirmAction) { const fn = state.confirmAction; state.confirmAction = null; await fn(); }
  });
  bindUpload();

  // close buttons + overlay click + Esc
  $$("[data-close]").forEach((b) =>
    b.addEventListener("click", () => b.closest(".modal-overlay").classList.add("hidden")));
  $$(".modal-overlay").forEach((ov) =>
    ov.addEventListener("click", (e) => { if (e.target === ov) ov.classList.add("hidden"); }));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") $$(".modal-overlay").forEach((ov) => ov.classList.add("hidden"));
  });

  // subtle click sounds on buttons
  $$("button").forEach((b) => b.addEventListener("mousedown", sfx.click));
}

/* ---------------- go ---------------- */
window.addEventListener("DOMContentLoaded", runBoot);