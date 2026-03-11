// ─────────────────────────────────────────────
//  SOVA Popup
//  All communication goes through background.js.
//  No direct WebSocket — background.js owns it.
// ─────────────────────────────────────────────

const dot      = document.getElementById("dot");
const status   = document.getElementById("status");
const latest   = document.getElementById("latest");
const startBtn = document.getElementById("start-btn");

let _wsConnected   = false;
let _engineRunning = false;


// ── UI ────────────────────────────────────────

function setConnected(connected) {
  _wsConnected = connected;
  dot.classList.toggle("connected", connected);
  status.textContent = connected ? "Connected" : "Not connected";
  status.classList.toggle("connected", connected);
  refreshButton();
}

function setEngineRunning(running) {
  _engineRunning = running;
  refreshButton();
}

function refreshButton() {
  if (!_wsConnected) {
    startBtn.disabled        = true;
    startBtn.textContent     = "Open SOVA app first";
    startBtn.dataset.running = "false";
    return;
  }
  startBtn.disabled        = false;
  startBtn.textContent     = _engineRunning ? "Stop SOVA" : "Start SOVA";
  startBtn.dataset.running = _engineRunning ? "true" : "false";
  startBtn.ariaLabel       = _engineRunning ? "Stop SOVA engine" : "Start SOVA engine";
}


// ── Start / Stop ──────────────────────────────

startBtn.addEventListener("click", () => {
  if (!_wsConnected) return;
  chrome.runtime.sendMessage(
    { type: _engineRunning ? "stop_engine" : "start_engine" },
    () => { if (chrome.runtime.lastError) return; }
  );
  // Optimistic update — confirmed by engine_status message
  setEngineRunning(!_engineRunning);
});


// ── Messages from background.js ───────────────

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "sova_status")   setConnected(msg.connected);
  if (msg.type === "engine_status") setEngineRunning(msg.running);
  if (msg.type === "result") {
    const text = msg.summary ?? msg.description;
    if (text) latest.textContent = text;
  }
});


// ── Boot ─────────────────────────────────────

chrome.runtime.sendMessage({ type: "get_status" }, (resp) => {
  if (chrome.runtime.lastError) return;
  setConnected(resp?.connected ?? false);
});

chrome.runtime.sendMessage({ type: "get_engine_status" }, (resp) => {
  if (chrome.runtime.lastError) return;
  setEngineRunning(resp?.running ?? false);
});