chrome.runtime.onInstalled.addListener(() => {
  console.log("Meet Expression Detector installed.");
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.action !== "getStreamId") return;

  // sender.tab.id is gmeet tab that asked for capture
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
});