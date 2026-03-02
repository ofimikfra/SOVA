// --- EXISTING STREAM LOGIC ---
chrome.runtime.onInstalled.addListener(() => {
  console.log("SOVA: Meet Expression Detector installed.");
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.action === "getStreamId") {
    chrome.tabCapture.getMediaStreamId(
      { targetTabId: sender.tab.id },
      (streamId) => {
        if (chrome.runtime.lastError) {
          sendResponse({ error: chrome.runtime.lastError.message });
        } else {
          sendResponse({ streamId });
        }
      }
    );
    return true;
  }
});

setInterval(async () => {
  try {

    const response = await fetch('http://127.0.0.1:5000/get_status');
    const data = await response.json();

    const [activeTab] = await chrome.tabs.query({ 
      active: true, 
      url: "*://meet.google.com/*" 
    });

    if (activeTab) {
      chrome.tabs.sendMessage(activeTab.id, {
        type: "UPDATE_DASHBOARD",
        payload: data
      });
    }
  } catch (error) {
    // We ignore errors here so the console doesn't fill up if Python is off
    // console.log("Python server not reachable...");
  }
}, 1000); // 1000ms = 1 second