// ─────────────────────────────────────────────
//  SOVA Service Worker — background.js
//  Owns the WebSocket connection to the local
//  SOVA app. Forwards video frames from the
//  Meet content script and routes results back.
// ─────────────────────────────────────────────

const WS_URL          = "ws://localhost:8765";
const RECONNECT_DELAY = 3000;
const DASHBOARD_URL   = chrome.runtime.getURL("dashboard.html");

let socket          = null;
let _dashboardTabId = null;
let _meetTabId      = null;


// ── WebSocket ─────────────────────────────────

function connect() {
  socket = new WebSocket(WS_URL);

  socket.addEventListener("open", () => {
    console.log("[SOVA BG] Connected to local app");
    broadcastStatus(true);
  });

  socket.addEventListener("message", (event) => {
    try {
      const msg = JSON.parse(event.data);
      // Forward results to the Meet tab's content script
      if (msg.type === "result" && _meetTabId !== null) {
        chrome.tabs.sendMessage(_meetTabId, msg).catch(() => {});
      }
      // Forward config/results to dashboard tab
      if ((msg.type === "result" || msg.type === "config") && _dashboardTabId !== null) {
        chrome.tabs.sendMessage(_dashboardTabId, msg).catch(() => {});
      }
    } catch (e) {
      console.warn("[SOVA BG] Bad message:", e);
    }
  });

  socket.addEventListener("close", () => {
    console.warn("[SOVA BG] Disconnected — retrying in 3s...");
    broadcastStatus(false);
    socket = null;
    setTimeout(connect, RECONNECT_DELAY);
  });

  socket.addEventListener("error", () => {
    socket?.close();
  });
}

function sendToApp(payload) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(payload));
  }
}

function broadcastStatus(connected) {
  chrome.runtime.sendMessage({ type: "sova_status", connected }).catch(() => {});
  if (_meetTabId !== null) {
    chrome.tabs.sendMessage(_meetTabId, {
      type: "sova_status", connected,
    }).catch(() => {});
  }
}


// ── Message routing ───────────────────────────

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {

  // content.js registering itself as the active Meet tab
  if (msg.type === "register_meet_tab") {
    _meetTabId = sender.tab?.id ?? null;
    console.log(`[SOVA BG] Meet tab registered: ${_meetTabId}`);
    sendResponse({ ok: true, connected: socket?.readyState === WebSocket.OPEN });
    return true;
  }

  // Video frame from content.js → forward to SOVA app
  if (msg.type === "frame") {
    sendToApp({ type: "frame", data: msg.data });
    sendResponse({ ok: true });
    return;
  }

  // Settings from dashboard → forward to app
  if (msg.type === "settings") {
    sendToApp(msg);
    sendResponse({ ok: true });
    return;
  }

  // get_config from dashboard → forward to app
  if (msg.type === "get_config") {
    sendToApp({ type: "get_config" });
    sendResponse({ ok: true });
    return;
  }

  // Popup or dashboard requesting connection status
  if (msg.type === "get_status") {
    sendResponse({ connected: socket?.readyState === WebSocket.OPEN });
    return true;
  }

  // Open dashboard tab
  if (msg.type === "open_dashboard") {
    openDashboard().then(() => sendResponse({ ok: true }));
    return true;
  }

  // Keepalive ping from content.js
  if (msg.type === "ping") {
    sendResponse({ pong: true });
    return;
  }
});


// ── Dashboard tab management ──────────────────

async function openDashboard() {
  if (_dashboardTabId !== null) {
    try {
      await chrome.tabs.update(_dashboardTabId, { active: true });
      const tab = await chrome.tabs.get(_dashboardTabId);
      await chrome.windows.update(tab.windowId, { focused: true });
      return;
    } catch {
      _dashboardTabId = null;
    }
  }
  const tab = await chrome.tabs.create({ url: DASHBOARD_URL });
  _dashboardTabId = tab.id;
}

chrome.tabs.onRemoved.addListener((tabId) => {
  if (tabId === _dashboardTabId) _dashboardTabId = null;
  if (tabId === _meetTabId)      _meetTabId      = null;
});


// ── Lifecycle ─────────────────────────────────

chrome.runtime.onInstalled.addListener(({ reason }) => {
  if (reason === "install") openDashboard();
});

// Keepalive — prevents service worker from going idle
chrome.alarms.create("keepalive", { periodInMinutes: 0.4 });
chrome.alarms.onAlarm.addListener(() => {});


// ── Boot ─────────────────────────────────────

connect();