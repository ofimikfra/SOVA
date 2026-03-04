// ─────────────────────────────────────────────
//  SOVA Content Script — Google Meet
//  Receives analysis results from background.js
//  and renders them in the Meet overlay.
//
//  Audio is now captured directly by the SOVA
//  desktop app via ScreenCaptureKit / WASAPI
// ─────────────────────────────────────────────

let overlayEl = null;


// ── Overlay ───────────────────────────────────

function createOverlay() {
  overlayEl = document.createElement("div");
  overlayEl.id = "sova-overlay";
  Object.assign(overlayEl.style, {
    position:      "fixed",
    bottom:        "80px",
    right:         "16px",
    zIndex:        "99999",
    background:    "rgba(0,0,0,0.72)",
    color:         "#fff",
    fontFamily:    "monospace",
    fontSize:      "13px",
    lineHeight:    "1.6",
    padding:       "10px 14px",
    borderRadius:  "10px",
    pointerEvents: "none",
    minWidth:      "200px",
    transition:    "opacity 0.3s",
  });
  overlayEl.innerHTML = "SOVA — connecting...";
  document.body.appendChild(overlayEl);
}

function updateOverlay({ status, expression, gesture, action, sentiment }) {
  if (!overlayEl) return;

  if (status === "disconnected") {
    overlayEl.innerHTML      = "SOVA — ⚠️ not connected";
    overlayEl.style.opacity  = "0.5";
    return;
  }

  if (status === "connected" && !expression) {
    overlayEl.innerHTML      = "SOVA — ✅ connected";
    overlayEl.style.opacity  = "1";
    return;
  }

  overlayEl.style.opacity = "1";
  overlayEl.innerHTML = [
    `😐 <b>${expression ?? "—"}</b>`,
    `🤚 ${gesture  ?? "—"}`,
    `🧍 ${action   ?? "—"}`,
    sentiment ? `💬 ${sentiment}` : "",
  ]
    .filter(Boolean)
    .join("<br>");
}

function handleResult(msg) {
  if (msg.dashboardVisible === false) {
    overlayEl.style.opacity = "0";
    return;
  }
  updateOverlay({
    expression: msg.expression,
    gesture:    msg.gesture,
    action:     msg.action,
    sentiment:  msg.sentiment,
  });
}


// ── Listen for messages from background.js ────

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "result") {
    handleResult(msg);
  }
  if (msg.type === "sova_status") {
    updateOverlay({ status: msg.connected ? "connected" : "disconnected" });
  }
});


// ── Boot ──────────────────────────────────────

createOverlay();

// Register this tab with background.js so it knows
// where to forward results from the SOVA app
chrome.runtime.sendMessage({ type: "register_meet_tab" }, (response) => {
  if (response?.connected) {
    updateOverlay({ status: "connected" });
  }
});