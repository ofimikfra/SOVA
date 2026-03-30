// ─────────────────────────────────────────────
//  SOVA Content Script — Google Meet
//  Captures the active speaker video tile and
//  sends frames to background.js for analysis.
//  Also renders results in the Meet overlay.
// ─────────────────────────────────────────────

const CAPTURE_FPS      = 5;
const CAPTURE_INTERVAL = 1000 / CAPTURE_FPS;  // 200 ms
const FRAME_WIDTH      = 640;
const FRAME_HEIGHT     = 480;
const JPEG_QUALITY     = 0.7;

let overlayEl     = null;
let captureTimer  = null;
let activeVideoEl = null;  // currently tracked <video> element

// Offscreen canvas — reused every frame, never added to DOM
const _canvas = new OffscreenCanvas(FRAME_WIDTH, FRAME_HEIGHT);
const _ctx    = _canvas.getContext("2d");


// ── Active speaker detection ──────────────────
//
// Find all <video> elements that are actively
// playing and sort by rendered area — the largest
// visible tile is the active speaker in Meet's
// spotlight / auto layout.

function findActiveSpeakerVideo() {
  const videos = [...document.querySelectorAll("video")].filter((v) =>
    v.readyState  >= 2 &&   // has data
    !v.paused              &&   // is playing
    v.videoWidth  >  0     &&   // has real content
    v.videoHeight >  0     &&
    v.offsetWidth >  0     &&   // is visible
    v.offsetHeight > 0
  );

  if (videos.length === 0) return null;

  // Largest rendered area = active speaker tile
  videos.sort((a, b) =>
    (b.offsetWidth * b.offsetHeight) - (a.offsetWidth * a.offsetHeight)
  );

  return videos[0];
}


// ── Frame capture ─────────────────────────────

function captureFrame() {
  const video = findActiveSpeakerVideo();
  if (!video) return;

  if (video !== activeVideoEl) {
    console.log("[SOVA] Active speaker changed — tracking new tile");
    activeVideoEl = video;
  }

  try {
    _ctx.drawImage(video, 0, 0, FRAME_WIDTH, FRAME_HEIGHT);

    _canvas
      .convertToBlob({ type: "image/jpeg", quality: JPEG_QUALITY })
      .then((blob) => blob.arrayBuffer())
      .then((buf) => {
        // ArrayBuffer → base64
        const bytes  = new Uint8Array(buf);
        let binary   = "";
        for (let i = 0; i < bytes.byteLength; i++) {
          binary += String.fromCharCode(bytes[i]);
        }
        chrome.runtime.sendMessage({
          type: "frame",
          data: btoa(binary),
        }).catch(() => {});
      })
      .catch(() => {});

  } catch (e) {
    // Canvas tainted (CORS) — shouldn't happen with Meet's own video,
    // but stop cleanly if it does
    console.warn("[SOVA] Frame capture error (stopping):", e);
    stopCapture();
  }
}

function startCapture() {
  if (captureTimer) return;
  captureTimer = setInterval(captureFrame, CAPTURE_INTERVAL);
  console.log(`[SOVA] Frame capture started at ${CAPTURE_FPS} fps`);
}

function stopCapture() {
  if (!captureTimer) return;
  clearInterval(captureTimer);
  captureTimer = null;
  console.log("[SOVA] Frame capture stopped");
}


// ── DOM observer ──────────────────────────────
// Clears activeVideoEl if Meet removes it from
// the DOM (speaker change, layout switch).

new MutationObserver(() => {
  if (activeVideoEl && !document.contains(activeVideoEl)) {
    activeVideoEl = null;
  }
}).observe(document.body, { childList: true, subtree: true });


// ── Overlay ───────────────────────────────────

function createOverlay() {
  overlayEl = document.createElement("div");
  overlayEl.id = "sova-overlay";
  Object.assign(overlayEl.style, {
    position:      "fixed",
    top:        "80px",
    left:         "16px",
    zIndex:        "99999",
    background:    "rgba(0,0,0,0.72)",
    color:         "#fff",
    fontFamily:    "monospace",
    fontSize:      "13px",
    lineHeight:    "1.6",
    padding:       "10px 14px",
    borderRadius:  "10px",
    pointerEvents: "none",
    minWidth:      "260px",
    transition:    "opacity 0.3s",
  });
  overlayEl.innerHTML = "SOVA — connecting...";
  document.body.appendChild(overlayEl);
}

function updateOverlay({ status, expression, gesture, action, sentiment, description }) {
  if (!overlayEl) return;

  if (status === "disconnected") {
    overlayEl.innerHTML     = "SOVA — ⚠️ not connected";
    overlayEl.style.opacity = "0.5";
    stopCapture();
    return;
  }

  if (status === "connected" && !expression) {
    overlayEl.innerHTML     = "SOVA — ✅ connected";
    overlayEl.style.opacity = "1";
    startCapture();
    return;
  }

  overlayEl.style.opacity = "1";
  overlayEl.innerHTML = [
    `😐 <b>${expression ?? "—"}</b>`,
    `🤚 ${gesture  ?? "—"}`,
    `🧍 ${action   ?? "—"}`,
    sentiment    ? `💬 ${sentiment}`    : "",
    description  ? `<hr style="border-color:rgba(255,255,255,0.2);margin:6px 0">📝 <i>${description}</i>` : "",
  ]
    .filter(Boolean)
    .join("<br>");
}

function handleResult(msg) {
  overlayEl.style.direction = msg.rtl ? "rtl" : "ltr";
  if (msg.dashboardVisible === false) {
    overlayEl.style.opacity = "0";
    return;
  }
  updateOverlay({
    expression:  msg.expression,
    gesture:     msg.gesture,
    action:      msg.action,
    sentiment:   msg.sentiment,
    description: msg.description,
  });
}


// ── Messages from background.js ───────────────

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "result")      handleResult(msg);
  if (msg.type === "sova_status") {
    updateOverlay({ status: msg.connected ? "connected" : "disconnected" });
  }
});


// ── Boot ──────────────────────────────────────

createOverlay();

// Register with background.js — it needs our tab ID to route
// results back to us, and tells us if the app is already connected
chrome.runtime.sendMessage({ type: "register_meet_tab" }, (response) => {
  if (response?.connected) {
    updateOverlay({ status: "connected" });
    startCapture();
  }
});