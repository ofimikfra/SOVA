const host = document.createElement('div');
host.id = 'sova-dashboard-host';
document.body.appendChild(host);

const shadow = host.attachShadow({ mode: 'open' });

// 3. Create the Dashboard structure
const dashboard = document.createElement('div');
dashboard.id = 'sova-dashboard';
dashboard.innerHTML = `
  <div class="header">
    <img src="${chrome.runtime.getURL('logo.png')}" class="mini-logo">
    <span>SOVA Dashboard</span>
  </div>
  <div class="stats-grid">
    <div class="stat-box">
        <label>Expression</label>
        <div id="expr-val" class="value">Neutral</div>
    </div>
    <div class="stat-box">
        <label>Gesture</label>
        <div id="gest-val" class="value">None</div>
    </div>
  </div>
  <div class="history-container">
    <label>Last Voice Output</label>
    <div id="tts-history" class="history-list"></div>
  </div>
`;
const style = document.createElement('style');
style.textContent = `
  #sova-dashboard {
    position: fixed; top: 20px; right: 20px; width: 280px;
    background: #1a6d7a; color: white; border-radius: 12px;
    padding: 15px; z-index: 999999; font-family: 'Segoe UI', Tahoma, sans-serif;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1);
  }
  .header { display: flex; align-items: center; gap: 10px; margin-bottom: 15px; font-weight: bold; }
  .mini-logo { width: 24px; height: 24px; }
  .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px; }
  .stat-box { background: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px; text-align: center; }
  label { font-size: 0.7rem; text-transform: uppercase; opacity: 0.8; display: block; margin-bottom: 4px; }
  .value { font-size: 1rem; font-weight: bold; }
  .history-list { font-size: 0.85rem; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 8px; min-height: 40px; }
`;

shadow.appendChild(style);
shadow.appendChild(dashboard);

chrome.runtime.onMessage.addListener((message) => {
  if (message.type === "UPDATE_DASHBOARD") {
    const data = message.payload;
    shadow.getElementById('expr-val').innerText = data.last_expression;
    shadow.getElementById('gest-val').innerText = data.last_gesture;

    const historyBox = shadow.getElementById('tts-history');
    if (data.tts_history && data.tts_history.length > 0) {
        historyBox.innerText = data.tts_history[0]; // Show the most recent one
    }
  }
});