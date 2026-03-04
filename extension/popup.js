const WS_URL = "ws://localhost:8765";
let socket = null;

const dot      = document.getElementById("dot");
const status   = document.getElementById("status");
const latest   = document.getElementById("latest");
const openDash = document.getElementById("open-dash");

// ── Open dashboard via background.js ──────────
// This ensures we reuse an existing tab rather than opening duplicates.
openDash.addEventListener("click", () => {
  chrome.runtime.sendMessage({ type: "open_dashboard" });
});

// ── Listen for status forwarded from content.js ──
chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "sova_status") {
    setConnected(msg.connected);
  }
});

// ── Direct WebSocket for latest description ───
// Popup connects directly for live description updates.
function connect() {
  socket = new WebSocket(WS_URL);

  socket.addEventListener("open", () => setConnected(true));

  socket.addEventListener("message", (e) => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.type === "result") {
        // support either `summary` (preferred) or legacy `description`
        const text = msg.summary ?? msg.description;
        if (text) {
          latest.textContent = text;
        }
      }
    } catch (_) {}
  });

  socket.addEventListener("close", () => {
    setConnected(false);
    setTimeout(connect, 3000);
  });

  socket.addEventListener("error", () => socket.close());
}

function setConnected(connected) {
  dot.classList.toggle("connected", connected);
  status.textContent = connected ? "Connected" : "Not connected";
  status.classList.toggle("connected", connected);
}

connect();