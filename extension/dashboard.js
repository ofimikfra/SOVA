// ─────────────────────────────────────────────
//  SOVA Dashboard — dashboard.js
//
//  Works in two contexts:
//    1. Chrome extension tab — messages route
//       through background.js (no direct WS)
//    2. PyWebView desktop app — uses pywebview
//       API for start/stop, direct WS for data
// ─────────────────────────────────────────────

const WS_URL          = "ws://localhost:8765";
const RECONNECT_DELAY = 3000;
const MAX_FEED_ITEMS  = 20;

// ── State ─────────────────────────────────────
let socket          = null;
let engineRunning   = false;
let isExtension     = (typeof chrome !== "undefined" && !!chrome.runtime?.id);
let isDesktopApp    = (typeof window.pywebview !== "undefined");

// ── DOM refs ──────────────────────────────────
const statusPill   = document.getElementById("status-pill");
const statusText   = document.getElementById("status-text");
const descText     = document.getElementById("description-text");
const confLabel    = document.getElementById("conf-label");
const confBar      = document.getElementById("conf-bar");
const confPct      = document.getElementById("conf-pct");
const confTrack    = confBar.parentElement;
const feed         = document.getElementById("feed");
const srLive       = document.getElementById("sr-live");
const startBtn     = document.getElementById("start-btn");

const ttsToggle    = document.getElementById("tts-toggle");
const flushSlider  = document.getElementById("flush-interval");
const intervalDisp = document.getElementById("interval-display");
const volumeSlider = document.getElementById("tts-volume");
const volumeDisp   = document.getElementById("tts-volume-display");
const modelSelect  = document.getElementById("ollama-model");
const saveBtn      = document.getElementById("save-btn");
const saveStatus   = document.getElementById("save-status");


// ── Status ────────────────────────────────────

function setStatus(s) {
  statusPill.dataset.status = s;
  statusPill.ariaLabel      = `Connection status: ${s}`;
  statusText.textContent    = s === "connected" ? "Connected" : "Not connected";
}


// ── Result handler ────────────────────────────

function handleResult(msg) {
  // support both new `summary` field and legacy `description` key
  const summary   = msg.summary ?? msg.description ?? "No description available.";
  const conf      = msg.sentimentConf ?? 0;
  const sentiment = msg.sentiment ?? "neutral";

  // Latest description
  descText.textContent = summary;
  descText.classList.remove("placeholder");

  // Confidence bar
  const pct = Math.round(conf * 100);
  confBar.style.width = pct + "%";
  confTrack.setAttribute("aria-valuenow", pct);
  confPct.textContent   = pct + "%";
  confLabel.textContent = confTier(conf);

  // Screen reader announcement
  srLive.textContent = summary;

  // Feed
  addFeedItem(summary, sentiment);
}

function confTier(conf) {
  if (conf < 0.65) return "Low confidence";
  if (conf < 0.85) return "Medium confidence";
  return "High confidence";
}


// ── Feed ──────────────────────────────────────

function addFeedItem(text, sentiment) {
  const time = new Date().toLocaleTimeString([], {
    hour: "2-digit", minute: "2-digit"
  });

  const li = document.createElement("li");
  li.className         = "feed-item";
  li.dataset.sentiment = sentiment ?? "neutral";
  li.innerHTML = `
    <span class="feed-time" aria-label="Time: ${time}">${time}</span>
    <span class="feed-text">${escapeHtml(text)}</span>
  `;

  feed.prepend(li);

  while (feed.children.length > MAX_FEED_ITEMS) {
    feed.removeChild(feed.lastChild);
  }
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}


// ── Config ────────────────────────────────────

function applyConfig(cfg) {
  if (cfg.tts_enabled    !== undefined) ttsToggle.checked     = cfg.tts_enabled;
  if (cfg.flush_interval !== undefined) {
    flushSlider.value              = cfg.flush_interval;
    intervalDisp.textContent       = cfg.flush_interval + "s";
    flushSlider.setAttribute("aria-valuenow", cfg.flush_interval);
  }
  if (cfg.tts_volume !== undefined) {
    volumeSlider.value              = cfg.tts_volume;
    volumeDisp.textContent       = Math.round(cfg.tts_volume * 100)  + "%";
    volumeSlider.setAttribute("aria-valuenow", cfg.tts_volume);
  }
  if (cfg.ollama_model   !== undefined) modelSelect.value     = cfg.ollama_model;
}

function loadLocalSettings() {
  const raw = localStorage.getItem("sova_config");
  if (raw) {
    try { applyConfig(JSON.parse(raw)); } catch (_) {}
  }
}

flushSlider.addEventListener("input", () => {
  intervalDisp.textContent = flushSlider.value + "s";
  flushSlider.setAttribute("aria-valuenow", flushSlider.value * 100);
});

// Volume slider — send immediately and play a test sound
volumeSlider.addEventListener("input", () => {
  const val = parseFloat(volumeSlider.value);
  volumeDisp.textContent = Math.round(val * 100) + "%";
  volumeSlider.setAttribute("aria-valuenow", val);
  sendVolume(val);
});

function sendVolume(val) {
  const msg = { type: "settings", tts_volume: val };
  if (isExtension) {
    chrome.runtime.sendMessage(msg);
  } else if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(msg));
  }
}

// ── Settings save ─────────────────────────────

saveBtn.addEventListener("click", () => {
  const settings = {
    type:           "settings",
    tts_enabled:    ttsToggle.checked,
    tts_volume:     parseFloat(ttsVolumeSlider.value),
    flush_interval: parseInt(flushSlider.value, 10),
    ollama_model:   modelSelect.value,
  };

  localStorage.setItem("sova_config", JSON.stringify(settings));

  if (isExtension) {
    // Route through background.js — it owns the WS connection
    chrome.runtime.sendMessage(settings, (resp) => {
      if (chrome.runtime.lastError || !resp?.ok) {
        showSaveStatus("✓ Saved locally — SOVA not connected");
      } else {
        showSaveStatus("✓ Saved and applied");
      }
    });

  } else if (socket && socket.readyState === WebSocket.OPEN) {
    // Desktop app or direct browser — use WebSocket directly
    socket.send(JSON.stringify(settings));
    showSaveStatus("✓ Saved and applied");

  } else {
    showSaveStatus("✓ Saved locally — SOVA not connected");
  }
});

function showSaveStatus(msg) {
  saveStatus.textContent = msg;
  setTimeout(() => { saveStatus.textContent = ""; }, 3000);
}


// ── Extension context ─────────────────────────
// background.js forwards results and config here via chrome.runtime.onMessage

function initExtension() {
  // Register this tab with background.js so it routes results here,
  // regardless of whether the tab was opened via openDashboard() or directly.
  chrome.runtime.sendMessage({ type: "register_dashboard_tab" }, (resp) => {
    if (chrome.runtime.lastError) return;
    setStatus(resp?.connected ? "connected" : "disconnected");
  });

  // Listen for messages forwarded from background.js
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === "result")        handleResult(msg);
    if (msg.type === "config")        applyConfig(msg);
    if (msg.type === "sova_status")   setStatus(msg.connected ? "connected" : "disconnected");
    if (msg.type === "engine_status") setEngineState(msg.running);
  });

  // Ask background for current config (status is handled by registration response above)
  chrome.runtime.sendMessage({ type: "get_config" });
}


// ── Desktop app context (PyWebView) ───────────
// Direct WebSocket + pywebview API for start/stop

function initDesktopApp() {
  // Show start/stop button
  if (startBtn) startBtn.style.display = "block";

  // Sync initial engine state via pywebview API (no WS needed)
  window.pywebview.api.get_status().then((s) => {
    setEngineState(s.running);
    setStatus(s.running ? "connected" : "disconnected");
  });

  // Start / Stop button — calls pywebview API directly
  startBtn?.addEventListener("click", async () => {
    if (!engineRunning) {
      const result = await window.pywebview.api.start();
      if (result.ok) { setEngineState(true); setStatus("connected"); }
    } else {
      const result = await window.pywebview.api.stop();
      if (result.ok) setEngineState(false);
    }
  });

  // app.py pushes results here via window.evaluate_js() — no WS connection needed.
  // This keeps the PyWebView window out of the WS client list entirely.
  window.__sovaReceive = (msg) => {
    if (msg.type === "result")        { setStatus("connected"); handleResult(msg); }
    if (msg.type === "config")        applyConfig(msg);
    if (msg.type === "engine_status") setEngineState(msg.running);
  };
}

function setEngineState(running) {
  engineRunning = running;
  if (!startBtn) return;
  startBtn.textContent     = running ? "Stop SOVA"  : "Start SOVA";
  startBtn.ariaLabel       = running ? "Stop SOVA engine" : "Start SOVA engine";
  startBtn.dataset.running = running;
}

function connectWS() {
  socket = new WebSocket(WS_URL);

  socket.addEventListener("open", () => {
    setStatus("connected");
    socket.send(JSON.stringify({ type: "get_config" }));
  });

  socket.addEventListener("message", (e) => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.type === "result") handleResult(msg);
      if (msg.type === "config") applyConfig(msg);
    } catch (_) {}
  });

  socket.addEventListener("close", () => {
    setStatus("disconnected");
    setTimeout(connectWS, RECONNECT_DELAY);
  });

  socket.addEventListener("error", () => socket.close());
}


// ── Boot ──────────────────────────────────────

loadLocalSettings();

if (isExtension) {
  initExtension();
} else {
  // PyWebView desktop app: pywebview API is injected async, so we listen for
  // the ready event. We also check synchronously in case it's already there.
  // We NEVER call connectWS() here — the desktop dashboard receives data via
  // window.__sovaReceive (evaluate_js), not via WebSocket.
  window.addEventListener("pywebviewready", initDesktopApp);
  if (typeof window.pywebview !== "undefined") {
    initDesktopApp();
  }
  // If neither fires (e.g. opened as file:// directly in a browser for testing),
  // just show disconnected state — no WS connection is made.
}