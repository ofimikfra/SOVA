// ─────────────────────────────────────────────
//  SOVA Content Script — Google Meet
//  Scrapes live captions and relays them to the
//  local SOVA app via WebSocket.
// ─────────────────────────────────────────────

const WS_URL = "ws://localhost:8765";
const RECONNECT_DELAY_MS = 3000;

// Google Meet caption container selector.
// Meet renders captions inside a div with this attribute.
const CAPTION_SELECTOR = '[jsname="tgaKEf"]';

let socket = null;
let lastCaption = "";   // deduplicate — Meet updates the same element in-place
let overlayEl   = null;


// ── WebSocket ────────────────────────────────

function connect() {
  socket = new WebSocket(WS_URL);

  socket.addEventListener("open", () => {
    console.log("[SOVA] Connected to local app");
    updateOverlay({ status: "connected" });
  });

  socket.addEventListener("message", (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === "result") {
        handleResult(msg);
      }
    } catch (e) {
      console.warn("[SOVA] Bad message from server:", e);
    }
  });

  socket.addEventListener("close", () => {
    console.warn("[SOVA] Disconnected — retrying in 3 s...");
    updateOverlay({ status: "disconnected" });
    setTimeout(connect, RECONNECT_DELAY_MS);
  });

  socket.addEventListener("error", () => {
    // 'close' fires right after, which handles the retry
    socket.close();
  });
}

function sendCaption(text) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "caption", text }));
  }
}


// ── Caption Scraping ─────────────────────────

function attachCaptionObserver() {
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      // Watch for new caption nodes being added
      for (const node of mutation.addedNodes) {
        if (node.nodeType !== Node.ELEMENT_NODE) continue;

        // The caption text lives inside jsname="tgaKEf" spans
        const captionEls = node.matches(CAPTION_SELECTOR)
          ? [node]
          : [...node.querySelectorAll(CAPTION_SELECTOR)];

        for (const el of captionEls) {
          const text = el.innerText?.trim();
          if (text && text !== lastCaption) {
            lastCaption = text;
            sendCaption(text);
          }
        }
      }

      // Also watch for text changes inside existing caption nodes
      if (
        mutation.type === "characterData" ||
        mutation.type === "childList"
      ) {
        const target = mutation.target.closest?.(CAPTION_SELECTOR)
          ?? (mutation.target.matches?.(CAPTION_SELECTOR)
              ? mutation.target
              : null);

        if (target) {
          const text = target.innerText?.trim();
          if (text && text !== lastCaption) {
            lastCaption = text;
            sendCaption(text);
          }
        }
      }
    }
  });

  observer.observe(document.body, {
    childList:  true,
    subtree:    true,
    characterData: true,
  });

  console.log("[SOVA] Caption observer attached");
}


// ── Overlay ──────────────────────────────────

function createOverlay() {
  overlayEl = document.createElement("div");
  overlayEl.id = "sova-overlay";
  Object.assign(overlayEl.style, {
    position:        "fixed",
    bottom:          "80px",
    right:           "16px",
    zIndex:          "99999",
    background:      "rgba(0,0,0,0.72)",
    color:           "#fff",
    fontFamily:      "monospace",
    fontSize:        "13px",
    lineHeight:      "1.6",
    padding:         "10px 14px",
    borderRadius:    "10px",
    pointerEvents:   "none",
    minWidth:        "200px",
    transition:      "opacity 0.3s",
  });
  overlayEl.innerHTML = "SOVA — connecting...";
  document.body.appendChild(overlayEl);
}

function updateOverlay({ status, expression, gesture, action, sentiment }) {
  if (!overlayEl) return;

  if (status === "disconnected") {
    overlayEl.innerHTML = "SOVA — ⚠️ not connected";
    overlayEl.style.opacity = "0.5";
    return;
  }

  if (status === "connected" && !expression) {
    overlayEl.innerHTML = "SOVA — ✅ connected";
    overlayEl.style.opacity = "1";
    return;
  }

  overlayEl.style.opacity = "1";
  overlayEl.innerHTML = [
    `😐 <b>${expression ?? "—"}</b>`,
    `🤚 ${gesture ?? "—"}`,
    `🧍 ${action  ?? "—"}`,
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


// ── Boot ─────────────────────────────────────

createOverlay();
connect();
attachCaptionObserver();