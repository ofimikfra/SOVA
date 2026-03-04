// ─────────────────────────────────────────────
//  SOVA Content Script — Google Meet
//  Relays captions to background.js which holds
//  the WebSocket connection to the local app.
// ─────────────────────────────────────────────

const CAPTION_SELECTOR  = '[jsname="tgaKEf"]';
const RECONNECT_DELAY   = 3000;

let lastCaption = "";
let overlayEl   = null;
let _connected  = false;


// ── Register this tab with background.js ─────

function register() {
  chrome.runtime.sendMessage({ type: "register_meet_tab" }, (resp) => {
    if (chrome.runtime.lastError) {
      // Background worker may still be starting — retry
      setTimeout(register, RECONNECT_DELAY);
      return;
    }
    _connected = resp?.connected ?? false;
    updateOverlay({ status: _connected ? "connected" : "disconnected" });
    console.log(`[SOVA] Registered with background. WS connected: ${_connected}`);
  });
}


// ── Listen for messages from background.js ───

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "sova_status") {
    _connected = msg.connected;
    updateOverlay({ status: _connected ? "connected" : "disconnected" });
  }
  if (msg.type === "result") {
    handleResult(msg);
  }
});


// ── Send caption via background.js ───────────

function sendCaption(text) {
  chrome.runtime.sendMessage({ type: "caption", text }).catch(() => {});
}


// ── Caption Scraping ─────────────────────────

function attachCaptionObserver() {
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      for (const node of mutation.addedNodes) {
        if (node.nodeType !== Node.ELEMENT_NODE) continue;
        const captionEls = node.matches(CAPTION_SELECTOR)
          ? [node]
          : [...node.querySelectorAll(CAPTION_SELECTOR)];
        for (const el of captionEls) {
          const text = el.innerText?.trim();
          if (text && text !== lastCaption) {
            lastCaption = text;
            sendCaption(text);
            console.log(`[SOVA] Caption: "${text}"`);
          }
        }
      }

      if (mutation.type === "characterData" || mutation.type === "childList") {
        const target = mutation.target.closest?.(CAPTION_SELECTOR)
          ?? (mutation.target.matches?.(CAPTION_SELECTOR) ? mutation.target : null);
        if (target) {
          const text = target.innerText?.trim();
          if (text && text !== lastCaption) {
            lastCaption = text;
            sendCaption(text);
            console.log(`[SOVA] Caption: "${text}"`);
          }
        }
      }
    }
  });

  observer.observe(document.body, {
    childList:     true,
    subtree:       true,
    characterData: true,
  });

  console.log("[SOVA] Caption observer attached");
}


// ── Overlay ──────────────────────────────────

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
    overlayEl.innerHTML    = "SOVA — ⚠️ not connected";
    overlayEl.style.opacity = "0.5";
    return;
  }
  if (status === "connected" && !expression) {
    overlayEl.innerHTML    = "SOVA — ✅ connected";
    overlayEl.style.opacity = "1";
    return;
  }

  overlayEl.style.opacity = "1";
  overlayEl.innerHTML = [
    `😐 <b>${expression ?? "—"}</b>`,
    `🤚 ${gesture   ?? "—"}`,
    `🧍 ${action    ?? "—"}`,
    sentiment ? `💬 ${sentiment}` : "",
  ].filter(Boolean).join("<br>");
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


// ── Keepalive ping ────────────────────────────

function ping() {
  chrome.runtime.sendMessage({ type: "ping" }).catch(() => {});
}
setInterval(ping, 20_000);


// ── Boot ─────────────────────────────────────

createOverlay();
register();
attachCaptionObserver();