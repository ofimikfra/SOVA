// ─────────────────────────────────────────────
//  SOVA Service Worker — background.js
//  Owns the WebSocket connection to the local
//  SOVA app. Content scripts relay through here
//  to avoid Chrome's Private Network Access block.
// ─────────────────────────────────────────────

const WS_URL            = "ws://localhost:8765";
const RECONNECT_DELAY   = 3000;
const DASHBOARD_URL     = chrome.runtime.getURL("dashboard.html");

let socket          = null;
let _dashboardTabId = null;
let _meetTabId      = null;
let _engineRunning  = false;
let _intentionalClose = false;


// ── WebSocket ─────────────────────────────────

function connect() {
  // Close any stale socket first. _intentionalClose prevents the close
  // handler from scheduling a second connect() call.
  if (socket) {
    _intentionalClose = true;
    socket.close();
    socket = null;
  }

  socket = new WebSocket(WS_URL);

  socket.addEventListener("open", () => {
    console.log("[SOVA BG] Connected to local app");
    broadcastStatus(true);
  });

  socket.addEventListener("message", (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === "result" && msg.description && !msg.summary) {
        msg.summary = msg.description;
      }
      // route result messages carefully — don't duplicate when the
      // dashboard and meet tab id are the same
      if (msg.type === "result" && _meetTabId !== null && _meetTabId !== _dashboardTabId) {
        chrome.tabs.sendMessage(_meetTabId, msg).catch(() => {});
      }
      if (msg.type === "result") {
        chrome.runtime.sendMessage(msg).catch(() => {});
      }
      if ((msg.type === "result" || msg.type === "config") && _dashboardTabId !== null) {
        chrome.tabs.sendMessage(_dashboardTabId, msg).catch(() => {});
      }
      if (msg.type === "engine_status") {
        _engineRunning = msg.running ?? false;
        chrome.runtime.sendMessage(msg).catch(() => {});
        if (_dashboardTabId !== null) {
          chrome.tabs.sendMessage(_dashboardTabId, msg).catch(() => {});
        }
      }
    } catch (e) {
      console.warn("[SOVA BG] Bad message:", e);
    }
  });

  socket.addEventListener("close", () => {
    if (_intentionalClose) { _intentionalClose = false; return; }
    console.warn("[SOVA BG] Disconnected — retrying in 3s...");
    broadcastStatus(false);
    socket = null;
    setTimeout(connect, RECONNECT_DELAY);
  });

  socket.addEventListener("error", () => { socket?.close(); });
}

function sendToApp(payload) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(payload));
  }
}

function broadcastStatus(connected) {
  chrome.runtime.sendMessage({ type: "sova_status", connected }).catch(() => {});
  if (_meetTabId !== null) {
    chrome.tabs.sendMessage(_meetTabId, { type: "sova_status", connected }).catch(() => {});
  }
  if (_dashboardTabId !== null) {
    chrome.tabs.sendMessage(_dashboardTabId, { type: "sova_status", connected }).catch(() => {});
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

  // dashboard.js registering itself so results are always routed to it,
  // even when the tab was opened by navigating directly to the URL
  if (msg.type === "register_dashboard_tab") {
    _dashboardTabId = sender.tab?.id ?? null;
    console.log(`[SOVA BG] Dashboard tab registered: ${_dashboardTabId}`);
    // if the same tab was previously registered as the Meet page, clear
    // the meet registration so we don't send the same message twice
    if (_dashboardTabId !== null && _dashboardTabId === _meetTabId) {
      console.log("[SOVA BG] dashboard and meet tab are identical; clearing meet tab registration to avoid duplicates");
      _meetTabId = null;
    }
    sendResponse({ ok: true, connected: socket?.readyState === WebSocket.OPEN });
    return true;
  }

  // settings from dashboard → forward to app
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

  // popup or dashboard requesting connection status
  if (msg.type === "get_status") {
    sendResponse({ connected: socket?.readyState === WebSocket.OPEN });
    return true;
  }

  if (msg.type === "get_engine_status") {
    sendResponse({ running: _engineRunning });
    return true;
  }

  if (msg.type === "start_engine" || msg.type === "stop_engine") {
    sendToApp(msg);
    sendResponse({ ok: true });
    return true;
  }

  // open dashboard tab
  if (msg.type === "open_dashboard") {
    openDashboard().then(() => sendResponse({ ok: true }));
    return true;
  }

  // keepalive ping from content.js
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

// Keepalive alarm — prevents service worker from going idle
chrome.alarms.create("keepalive", { periodInMinutes: 0.4 });
chrome.alarms.onAlarm.addListener(() => {});


// ── Boot ─────────────────────────────────────

connect();