const WS_URL = "ws://localhost:8765";
let socket = null;

const dot        = document.getElementById("dot");
const status     = document.getElementById("status");
const latest     = document.getElementById("latest");
const openDash   = document.getElementById("open-dash");
const volSlider  = document.getElementById("tts-volume");
const volDisplay = document.getElementById("vol-display");

// ── Open dashboard ────────────────────────────
openDash.addEventListener("click", () => {
  chrome.runtime.sendMessage({ type: "open_dashboard" });
});

// ── Volume slider ─────────────────────────────
volSlider.addEventListener("input", () => {
  updateVolumeDisplay(parseFloat(volSlider.value));
  sendVolume(parseFloat(volSlider.value));
});

function setVolume(val) {
  val = Math.max(0, Math.min(1, parseFloat(val.toFixed(2))));
  volSlider.value = val;
  updateVolumeDisplay(val);
  sendVolume(val);
}

function updateVolumeDisplay(val) {
  volDisplay.textContent = Math.round(val * 100) + "%";
}

function sendVolume(val) {
  chrome.runtime.sendMessage({ type: "settings", tts_volume: val });
}

// ── Status from background ────────────────────
chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "sova_status") setConnected(msg.connected);
  if (msg.type === "config" && msg.tts_volume !== undefined) {
    volSlider.value = msg.tts_volume;
    updateVolumeDisplay(msg.tts_volume);
  }
});

// ── WebSocket for live descriptions ──────────
function connect() {
  socket = new WebSocket(WS_URL);
  socket.addEventListener("open", () => setConnected(true));
  socket.addEventListener("message", (e) => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.type === "result") {
        const text = msg.summary ?? msg.description;
        if (text) latest.textContent = text;
      }
      if (msg.type === "config" && msg.tts_volume !== undefined) {
        volSlider.value = msg.tts_volume;
        updateVolumeDisplay(msg.tts_volume);
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