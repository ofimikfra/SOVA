const WS_URL = "ws://localhost:8765";
let socket = null;

// ── DOM refs ──────────────────────────────────
const statusEl       = document.getElementById("status");
const latest         = document.getElementById("latest");
const openDash       = document.getElementById("open-dash");
const saveBtn        = document.getElementById("save-btn");
const saveStatus     = document.getElementById("save-status");
const volSlider      = document.getElementById("tts-volume");
const volDisplay     = document.getElementById("vol-display");
const intervalSlider = document.getElementById("flush-interval");
const intervalDisp   = document.getElementById("interval-display");
const cueExpression  = document.getElementById("cue-expression");
const cueGesture     = document.getElementById("cue-gesture");
const cueAction      = document.getElementById("cue-action");
const sentimentBadge = document.getElementById("sentiment-badge");
const confFill       = document.getElementById("conf-fill");
const confPct        = document.getElementById("conf-pct");


// ── Open dashboard ────────────────────────────

openDash.addEventListener("click", () => {
  chrome.runtime.sendMessage({ type: "open_dashboard" });
});


// ── Sliders ───────────────────────────────────

volSlider.addEventListener("input", () => {
  updateVolDisplay(parseFloat(volSlider.value));
});

intervalSlider.addEventListener("input", () => {
  intervalDisp.textContent = intervalSlider.value + "s";
});

function updateVolDisplay(val) {
  volDisplay.textContent = Math.round(val * 100) + "%";
}


// ── Save settings ─────────────────────────────

saveBtn.addEventListener("click", () => {
  const settings = {
    type:           "settings",
    tts_volume:     parseFloat(volSlider.value),
    flush_interval: parseInt(intervalSlider.value, 10),
  };
  chrome.runtime.sendMessage(settings, (resp) => {
    const ok = !chrome.runtime.lastError && resp?.ok;
    showSaveStatus(ok ? "✓ Saved" : "✓ Saved locally");
  });
});

function showSaveStatus(msg) {
  saveStatus.textContent = msg;
  setTimeout(() => { saveStatus.textContent = ""; }, 2000);
}


// ── Apply config from backend ─────────────────

function applyConfig(cfg) {
  if (cfg.tts_volume !== undefined) {
    volSlider.value = cfg.tts_volume;
    updateVolDisplay(cfg.tts_volume);
  }
  if (cfg.flush_interval !== undefined) {
    intervalSlider.value     = cfg.flush_interval;
    intervalDisp.textContent = cfg.flush_interval + "s";
  }
}


// ── Result handler ────────────────────────────

function handleResult(msg) {
  // Description
  const text = msg.summary ?? msg.description;
  if (text) latest.textContent = text;

  // Non-verbal cues
  if (msg.expression) cueExpression.textContent = `😐 ${msg.expression}`;
  if (msg.gesture)    cueGesture.textContent    = `🤚 ${msg.gesture}`;
  if (msg.action)     cueAction.textContent     = `🧍 ${msg.action}`;

  // Sentiment
  const sentiment = (msg.sentiment ?? "neutral").toLowerCase();
  const conf      = msg.sentimentConf ?? 0;
  sentimentBadge.textContent          = sentiment;
  sentimentBadge.dataset.sentiment    = sentiment;
  confFill.style.width                = Math.round(conf * 100) + "%";
  confPct.textContent                 = Math.round(conf * 100) + "%";
}


// ── Messages from background ──────────────────

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "sova_status") setConnected(msg.connected);
  if (msg.type === "result")      handleResult(msg);
  if (msg.type === "config")      applyConfig(msg);
});


// ── WebSocket ─────────────────────────────────

function connect() {
  socket = new WebSocket(WS_URL);

  socket.addEventListener("open", () => {
    setConnected(true);
    // ── FIX: request config so sliders reflect current backend state ──
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
    setConnected(false);
    setTimeout(connect, 3000);
  });

  socket.addEventListener("error", () => socket.close());
}

function setConnected(connected) {
  statusEl.textContent = connected ? "Connected" : "Not connected";
  statusEl.classList.toggle("connected", connected);
}

connect();