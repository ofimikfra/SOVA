// ─────────────────────────────────────────────
//  SOVA Dashboard — dashboard.js
// ─────────────────────────────────────────────

const WS_URL          = "ws://localhost:8765";
const RECONNECT_DELAY = 3000;
const MAX_FEED_ITEMS  = 20;

// ── State ─────────────────────────────────────
let socket             = null;
let engineRunning      = false;
let _lastEngineRunning = null;
let isExtension        = (typeof chrome !== "undefined" && !!chrome.runtime?.id);
let isDesktopApp       = (typeof window.pywebview !== "undefined");

// ── DOM refs ──────────────────────────────────
const statusPill    = document.getElementById("status-pill");
const statusText    = document.getElementById("status-text");
const descText      = document.getElementById("description-text");
const confLabel     = document.getElementById("conf-label");
const confBar       = document.getElementById("conf-bar");
const confPct       = document.getElementById("conf-pct");
const confTrack     = confBar.parentElement;
const feed          = document.getElementById("feed");
const srLive        = document.getElementById("sr-live");
const startBtn      = document.getElementById("start-btn");
const ttsToggle     = document.getElementById("tts-toggle");
const flushSlider   = document.getElementById("flush-interval");
const intervalDisp  = document.getElementById("interval-display");
const volumeSlider  = document.getElementById("tts-volume");
const volumeDisp    = document.getElementById("tts-volume-display");
const modelSelect   = document.getElementById("ollama-model");
const saveBtn       = document.getElementById("save-btn");
const saveStatus    = document.getElementById("save-status");
const helpBtn       = document.getElementById("help-btn");
const helpModal     = document.getElementById("help-modal");
const helpCloseBtn  = document.getElementById("help-close-btn");
const modalBackdrop = document.getElementById("modal-backdrop");
const srLivePolite  = document.getElementById("sr-live-polite");


// ── Narrator (Web Speech API) ─────────────────

const _synth = window.speechSynthesis;

function _narrate(text, { interrupt = true } = {}) {
  if (!_synth) return;
  if (interrupt) _synth.cancel();
  const utt = new SpeechSynthesisUtterance(text);
  utt.rate   = 0.95;
  utt.pitch  = 1.0;
  utt.volume = 1.0;
  _synth.speak(utt);
}

function _runWelcomeTour() {
  if (sessionStorage.getItem("sova_toured")) return;
  sessionStorage.setItem("sova_toured", "1");

  const lines = [
    "Welcome to SOVA.",
    "Press Space to start or stop SOVA.",
    "Press D to jump to the latest description.",
    "Press F to browse the description feed.",
    "Press G to open settings.",
    "Press T to toggle text-to-speech on or off.",
    "Press plus or minus to raise or lower the volume.",
    "Press R to re-read the latest description.",
    "Press H at any time to hear these shortcuts again.",
    "Press Tab and Shift Tab to move between controls.",
    "SOVA is ready.",
  ];

  lines.forEach(line => {
    const utt = new SpeechSynthesisUtterance(line);
    utt.rate   = 0.95;
    utt.pitch  = 1.0;
    utt.volume = 1.0;
    _synth.speak(utt);
  });
}

function _readShortcuts() {
  _narrate(
    "Keyboard shortcuts. " +
    "Space: start or stop SOVA. " +
    "D: latest description. " +
    "F: description feed. " +
    "G: settings. " +
    "T: toggle text-to-speech. " +
    "Plus or minus: volume up or down. " +
    "R: re-read last description. " +
    "H: repeat shortcuts. " +
    "Escape: close this panel."
  );
}


// ── Help modal ────────────────────────────────

function openHelpModal() {
  helpModal.hidden = false;
  helpModal.querySelector(".modal-box").focus();
  _readShortcuts();
}

function closeHelpModal() {
  helpModal.hidden = true;
  helpBtn.focus();
  _synth.cancel();
}

helpBtn.addEventListener("click", openHelpModal);
helpCloseBtn.addEventListener("click", closeHelpModal);
modalBackdrop.addEventListener("click", closeHelpModal);

helpModal.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { closeHelpModal(); return; }
  if (e.key !== "Tab") return;
  const focusable = [...helpModal.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  )];
  const first = focusable[0];
  const last  = focusable[focusable.length - 1];
  if (e.shiftKey && document.activeElement === first) {
    e.preventDefault(); last.focus();
  } else if (!e.shiftKey && document.activeElement === last) {
    e.preventDefault(); first.focus();
  }
});


// ── Global keyboard shortcuts ─────────────────

document.addEventListener("keydown", (e) => {
  const tag = document.activeElement?.tagName;
  const isTyping = (tag === "INPUT" && document.activeElement.type !== "range")
                || tag === "SELECT"
                || tag === "TEXTAREA";

  if (!helpModal.hidden) return;

  if (e.ctrlKey && e.key === "s") {
    e.preventDefault();
    saveSettings();
    return;
  }

  if (isTyping) return;

  switch (e.key) {
    case " ":
      e.preventDefault();
      if (startBtn && startBtn.style.display !== "none") startBtn.click();
      else _narrate("Start button is not available in extension mode.");
      break;

    case "d": case "D":
      document.getElementById("section-description")?.scrollIntoView({ behavior: "smooth" });
      descText.focus();
      _narrate("Latest description: " + (descText.textContent || "No description yet."));
      break;

    case "f": case "F":
      document.getElementById("section-feed")?.scrollIntoView({ behavior: "smooth" });
      feed.focus();
      _narrate("Description feed.");
      break;

    case "g": case "G":
      document.getElementById("section-settings")?.scrollIntoView({ behavior: "smooth" });
      ttsToggle.focus();
      _narrate("Settings.");
      break;

    case "t": case "T":
      ttsToggle.click();
      _narrate("Text-to-speech " + (ttsToggle.checked ? "on." : "off."));
      break;

    case "+": case "=": {
      const newVol = Math.min(1, parseFloat(volumeSlider.value) + 0.05);
      volumeSlider.value = newVol;
      volumeSlider.dispatchEvent(new Event("input"));
      _narrate(`Volume ${Math.round(newVol * 100)} percent.`);
      break;
    }

    case "-": case "_": {
      const newVol = Math.max(0, parseFloat(volumeSlider.value) - 0.05);
      volumeSlider.value = newVol;
      volumeSlider.dispatchEvent(new Event("input"));
      _narrate(`Volume ${Math.round(newVol * 100)} percent.`);
      break;
    }

    case "r": case "R":
      _narrate(descText.textContent || "No description yet.");
      break;

    case "h": case "H":
      openHelpModal();
      break;

    case "Escape":
      if (!helpModal.hidden) closeHelpModal();
      break;
  }
});


// ── Offline settings helpers ──────────────────

function _getLocalSettings() {
  try {
    const raw = localStorage.getItem("sova_config");
    return raw ? JSON.parse(raw) : null;
  } catch (_) { return null; }
}

function _persistLocal(partial) {
  const current = _getLocalSettings() || {};
  localStorage.setItem("sova_config", JSON.stringify({ ...current, ...partial }));
}

function flushSettingsToBackend() {
  const saved = _getLocalSettings();
  if (!saved) return;

  const payload = { type: "settings" };
  if (saved.tts_enabled    !== undefined) payload.tts_enabled    = saved.tts_enabled;
  if (saved.tts_volume     !== undefined) payload.tts_volume     = saved.tts_volume;
  if (saved.flush_interval !== undefined) payload.flush_interval = saved.flush_interval;
  if (saved.ollama_model   !== undefined) payload.ollama_model   = saved.ollama_model;

  if (isExtension) {
    chrome.runtime.sendMessage(payload);
  } else if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(payload));
  }
  console.log("[SOVA] Flushed offline settings to backend:", payload);
}


// ── Status ────────────────────────────────────

function setStatus(s) {
  statusPill.dataset.status = s;
  statusPill.setAttribute("aria-label", `Connection status: ${s}`);
  statusText.textContent = {
    connected:    "Connected",
    disconnected: "Not connected",
    starting:     "Starting up...",
    stopping:     "Stopping...",
  }[s] ?? "Not connected";
}


// ── Engine state ──────────────────────────────

function setEngineState(running) {
  if (_lastEngineRunning === running) return;
  _lastEngineRunning = running;
  engineRunning = running;

  if (startBtn) {
    startBtn.textContent     = running ? "Stop SOVA"  : "Start SOVA";
    startBtn.ariaLabel       = running ? "Stop SOVA engine" : "Start SOVA engine";
    startBtn.dataset.running = running;
  }

  setStatus(running ? "connected" : "disconnected");
  srLive.textContent = running ? "SOVA is connected." : "SOVA is disconnected.";
  _narrate(running ? "SOVA is connected." : "SOVA is disconnected.");
}


// ── Result handler ────────────────────────────

function handleResult(msg) {
  const summary   = msg.summary ?? msg.description ?? "No description available.";
  const conf      = msg.sentimentConf ?? 0;
  const sentiment = msg.sentiment ?? "neutral";

  descText.textContent = summary;
  descText.classList.remove("placeholder");

  const pct = Math.round(conf * 100);
  confBar.style.width = pct + "%";
  confTrack.setAttribute("aria-valuenow", pct);
  confPct.textContent   = pct + "%";
  confLabel.textContent = confTier(conf);

  srLive.textContent = summary;
  addFeedItem(summary, sentiment);
}

function confTier(conf) {
  if (conf < 0.65) return "Low confidence";
  if (conf < 0.85) return "Medium confidence";
  return "High confidence";
}


// ── Feed ──────────────────────────────────────

function addFeedItem(text, sentiment) {
  const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const li = document.createElement("li");
  li.className         = "feed-item";
  li.dataset.sentiment = sentiment ?? "neutral";
  li.innerHTML = `
    <span class="feed-time" aria-label="Time: ${time}">${time}</span>
    <span class="feed-text">${escapeHtml(text)}</span>
  `;
  feed.prepend(li);
  while (feed.children.length > MAX_FEED_ITEMS) feed.removeChild(feed.lastChild);
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}


// ── Config ────────────────────────────────────

function applyConfig(cfg) {
  if (cfg.tts_enabled    !== undefined) ttsToggle.checked = cfg.tts_enabled;
  if (cfg.flush_interval !== undefined) {
    flushSlider.value            = cfg.flush_interval;
    intervalDisp.textContent     = cfg.flush_interval + "s";
    flushSlider.setAttribute("aria-valuenow",  cfg.flush_interval);
    flushSlider.setAttribute("aria-valuetext", cfg.flush_interval + " seconds");
  }
  if (cfg.tts_volume !== undefined) {
    volumeSlider.value           = cfg.tts_volume;
    const pct                    = Math.round(cfg.tts_volume * 100);
    volumeDisp.textContent       = pct + "%";
    volumeSlider.setAttribute("aria-valuenow",  pct);
    volumeSlider.setAttribute("aria-valuetext", pct + " percent");
  }
  if (cfg.ollama_model !== undefined) modelSelect.value = cfg.ollama_model;
}

function loadLocalSettings() {
  const saved = _getLocalSettings();
  if (saved) applyConfig(saved);
}

flushSlider.addEventListener("input", () => {
  const v = flushSlider.value;
  intervalDisp.textContent = v + "s";
  flushSlider.setAttribute("aria-valuenow",  v);
  flushSlider.setAttribute("aria-valuetext", v + " seconds");
});

volumeSlider.addEventListener("input", () => {
  const val = parseFloat(volumeSlider.value);
  const pct = Math.round(val * 100);
  volumeDisp.textContent = pct + "%";
  volumeSlider.setAttribute("aria-valuenow",  pct);
  volumeSlider.setAttribute("aria-valuetext", pct + " percent");
  sendVolume(val);
});

function sendVolume(val) {
  const msg = { type: "settings", tts_volume: val };
  if (isExtension) chrome.runtime.sendMessage(msg);
  else if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(msg));
  _persistLocal({ tts_volume: val });
}


// ── Settings save ─────────────────────────────

saveBtn.addEventListener("click", saveSettings);

function saveSettings() {
  const settings = {
    type:           "settings",
    tts_enabled:    ttsToggle.checked,
    tts_volume:     parseFloat(volumeSlider.value),
    flush_interval: parseInt(flushSlider.value, 10),
    ollama_model:   modelSelect.value,
  };

  localStorage.setItem("sova_config", JSON.stringify(settings));

  if (isExtension) {
    chrome.runtime.sendMessage(settings, (resp) => {
      const msg = chrome.runtime.lastError || !resp?.ok
        ? "✓ Saved — will apply when SOVA connects"
        : "✓ Saved and applied";
      showSaveStatus(msg);
      _narrate(msg.replace("✓ ", ""));
    });

  } else if (typeof window.pywebview !== "undefined") {
    window.pywebview.api.save_config(settings).then((resp) => {
      const msg = resp?.ok ? "✓ Saved and applied" : "✓ Saved locally";
      showSaveStatus(msg);
      _narrate(msg.replace("✓ ", ""));
    });

  } else if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(settings));
    showSaveStatus("✓ Saved and applied");
    _narrate("Saved and applied.");

  } else {
    showSaveStatus("✓ Saved — will apply when SOVA connects");
    _narrate("Saved. Will apply when SOVA connects.");
  }
}

function showSaveStatus(msg) {
  saveStatus.textContent = msg;
  setTimeout(() => { saveStatus.textContent = ""; }, 3000);
}


// ── Extension context ─────────────────────────

function initExtension() {
  chrome.runtime.sendMessage({ type: "register_dashboard_tab" }, (resp) => {
    if (chrome.runtime.lastError) return;
    setStatus(resp?.connected ? "connected" : "disconnected");
    if (resp?.connected) {
      flushSettingsToBackend();
      chrome.runtime.sendMessage({ type: "get_config" });
    }
  });

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === "result")      handleResult(msg);
    if (msg.type === "config")      applyConfig(msg);
    if (msg.type === "sova_status") {
      setStatus(msg.connected ? "connected" : "disconnected");
      if (msg.connected) {
        flushSettingsToBackend();
        chrome.runtime.sendMessage({ type: "get_config" });
      }
    }
    if (msg.type === "engine_status") setEngineState(msg.running);
  });
}


// ── Desktop app context (PyWebView) ───────────

function initDesktopApp() {
  if (startBtn) startBtn.style.display = "block";

  window.pywebview.api.get_status().then((s) => {
    setEngineState(s.running);
  });

  startBtn?.addEventListener("click", async () => {
    if (!engineRunning) {
      setStatus("starting");
      srLive.textContent = "SOVA is starting up. Please wait.";
      _narrate("SOVA is starting up. Please wait.");
      startBtn.disabled = true;

      const result = await window.pywebview.api.start();
      startBtn.disabled = false;

      if (!result.ok) {
        setStatus("disconnected");
        srLive.textContent = "SOVA failed to start.";
        _narrate("SOVA failed to start.");
        _lastEngineRunning = null;
      }
      // On success: engine_status:true arrives via __sovaReceive → setEngineState(true)

    } else {
      setStatus("stopping");
      srLive.textContent = "SOVA is stopping.";
      _narrate("Stopping SOVA.");

      startBtn.disabled = true;
      await window.pywebview.api.stop();
      startBtn.disabled = false;
      // engine_status:false arrives via __sovaReceive → setEngineState(false)
    }
  });

  window.__sovaReceive = (msg) => {
    if (msg.type === "result")        handleResult(msg);
    if (msg.type === "config")        applyConfig(msg);
    if (msg.type === "engine_status") setEngineState(msg.running);
  };
}

function connectWS() {
  socket = new WebSocket(WS_URL);
  socket.addEventListener("open", () => {
    setStatus("connected");
    flushSettingsToBackend();
    socket.send(JSON.stringify({ type: "get_config" }));
  });
  socket.addEventListener("message", (e) => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.type === "result")        handleResult(msg);
      if (msg.type === "config")        applyConfig(msg);
      if (msg.type === "engine_status") setEngineState(msg.running);
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
  window.addEventListener("pywebviewready", initDesktopApp);
  if (typeof window.pywebview !== "undefined") initDesktopApp();
}

setTimeout(_runWelcomeTour, 800);