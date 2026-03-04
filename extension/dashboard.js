// ── Config ────────────────────────────────────────────────────────────
const WS_URL            = "ws://localhost:8765";
const RECONNECT_DELAY   = 3000;
const MAX_FEED_ITEMS    = 20;

// ── State ─────────────────────────────────────────────────────────────
let socket        = null;
let latestSentiment = "neutral";
let latestConf      = 0;

// ── DOM refs ──────────────────────────────────────────────────────────
const statusPill    = document.getElementById("status-pill");
const statusText    = document.getElementById("status-text");
const descText      = document.getElementById("description-text");
const confLabel     = document.getElementById("conf-label");
const confBar       = document.getElementById("conf-bar");
const confPct       = document.getElementById("conf-pct");
const confTrack     = confBar.parentElement;
const feed          = document.getElementById("feed");
const srLive        = document.getElementById("sr-live");

const ttsToggle     = document.getElementById("tts-toggle");
const flushSlider   = document.getElementById("flush-interval");
const intervalDisp  = document.getElementById("interval-display");
const modelSelect   = document.getElementById("ollama-model");
const saveBtn       = document.getElementById("save-btn");
const saveStatus    = document.getElementById("save-status");

// ── Desktop app bridge ────────────────────────────────────────────────────
// window.pywebview is only defined when running inside the PyWebView app.
// When opened directly in a browser (extension context) it's undefined
// and the Start button stays hidden — so the same file works in both contexts.

const startBtn = document.getElementById("start-btn");
let engineRunning = false;

async function initDesktopApp() {
    if (typeof window.pywebview === "undefined") return;

    // Show the start button — we're inside the desktop app
    startBtn.style.display = "block";

    // Sync initial state
    const status = await window.pywebview.api.get_status();
    setEngineState(status.running);
}

function setEngineState(running) {
    engineRunning = running;
    startBtn.textContent          = running ? "Stop SOVA" : "Start SOVA";
    startBtn.ariaLabel            = running ? "Stop SOVA engine" : "Start SOVA engine";
    startBtn.dataset.running      = running;
}

startBtn.addEventListener("click", async () => {
    if (typeof window.pywebview === "undefined") return;

    if (!engineRunning) {
    const result = await window.pywebview.api.start();
    if (result.ok) setEngineState(true);
    } else {
    const result = await window.pywebview.api.stop();
    if (result.ok) setEngineState(false);
    }
});

// Wait for pywebview to be ready before calling the API
window.addEventListener("pywebviewready", initDesktopApp);
// Fallback — pywebviewready sometimes fires before listener attached
if (typeof window.pywebview !== "undefined") initDesktopApp();

// ── WebSocket ─────────────────────────────────────────────────────────
function connect() {
    socket = new WebSocket(WS_URL);

    socket.addEventListener("open", () => {
    setStatus("connected");
    // Request current config on connect
    socket.send(JSON.stringify({ type: "get_config" }));
    });

    socket.addEventListener("message", (e) => {
    try {
        const msg = JSON.parse(e.data);
        if (msg.type === "result")  handleResult(msg);
        if (msg.type === "config")  applyConfig(msg);
    } catch (_) {}
    });

    socket.addEventListener("close", () => {
    setStatus("disconnected");
    setTimeout(connect, RECONNECT_DELAY);
    });

    socket.addEventListener("error", () => socket.close());
}

// ── Status ────────────────────────────────────────────────────────────
function setStatus(s) {
    statusPill.dataset.status    = s;
    statusPill.ariaLabel         = `Connection status: ${s}`;
    statusText.textContent       = s === "connected" ? "Connected" : "Not connected";
}

// ── Handle result from SOVA ───────────────────────────────────────────
function handleResult(msg) {
    const summary  = msg.summary  ?? "No description available.";
    const conf     = msg.sentimentConf ?? 0;
    const sentiment= msg.sentiment ?? "neutral";

    latestSentiment = sentiment;
    latestConf      = conf;

    // Update latest description card
    descText.textContent = summary;
    descText.classList.remove("placeholder");

    // Confidence bar
    const pct = Math.round(conf * 100);
    confBar.style.width            = pct + "%";
    confTrack.setAttribute("aria-valuenow", pct);
    confPct.textContent            = pct + "%";
    confLabel.textContent          = confTier(conf);

    // Announce to screen readers
    srLive.textContent = summary;

    // Add to feed
    addFeedItem(summary, sentiment, conf);
}

function confTier(conf) {
    if (conf < 0.65) return "Low confidence";
    if (conf < 0.85) return "Medium confidence";
    return "High confidence";
}

// ── Feed ──────────────────────────────────────────────────────────────
function addFeedItem(text, sentiment, conf) {
    const now  = new Date();
    const time = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    const li = document.createElement("li");
    li.className = "feed-item";
    li.dataset.sentiment = sentiment;
    li.innerHTML = `
    <span class="feed-time" aria-label="Time: ${time}">${time}</span>
    <span class="feed-text">${escapeHtml(text)}</span>
    `;

    feed.prepend(li);

    // Trim old items
    while (feed.children.length > MAX_FEED_ITEMS) {
    feed.removeChild(feed.lastChild);
    }
}

function escapeHtml(str) {
    return str.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

// ── Settings ──────────────────────────────────────────────────────────
flushSlider.addEventListener("input", () => {
    intervalDisp.textContent        = flushSlider.value + "s";
    flushSlider.ariaValueNow        = flushSlider.value;
});

function applyConfig(cfg) {
    if (cfg.tts_enabled    !== undefined) ttsToggle.checked  = cfg.tts_enabled;
    if (cfg.flush_interval !== undefined) {
    flushSlider.value              = cfg.flush_interval;
    intervalDisp.textContent       = cfg.flush_interval + "s";
    }
    if (cfg.ollama_model   !== undefined) modelSelect.value  = cfg.ollama_model;
}

function loadLocalSettings() {
    // Fall back to chrome.storage if available (extension context),
    // otherwise use localStorage for dev/file:// context
    const store = (typeof chrome !== "undefined" && chrome.storage)
    ? null   // handled via WS get_config
    : null;

    const raw = localStorage.getItem("sova_config");
    if (raw) {
    try { applyConfig(JSON.parse(raw)); } catch(_) {}
    }
}

saveBtn.addEventListener("click", () => {
    const settings = {
    type:           "settings",
    tts_enabled:    ttsToggle.checked,
    flush_interval: parseInt(flushSlider.value, 10),
    ollama_model:   modelSelect.value,
    };

    // Save locally
    localStorage.setItem("sova_config", JSON.stringify(settings));

    // Send to local app if connected
    if (socket && socket.readyState === WebSocket.OPEN) {
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

// ── Boot ──────────────────────────────────────────────────────────────
loadLocalSettings();
connect();
