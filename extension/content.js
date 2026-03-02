// --- 1. INITIALIZATION & SAFETY ---
console.log("SOVA: Dashboard script loading...");

// Prevent multiple injections if the extension reloads
if (!document.getElementById('sova-dashboard-host')) {
    const host = document.createElement('div');
    host.id = 'sova-dashboard-host';
    document.body.appendChild(host);

    const shadow = host.attachShadow({ mode: 'open' });

    // --- 2. THE DASHBOARD HTML ---
    const dashboard = document.createElement('div');
    dashboard.id = 'sova-dashboard';
    
    // Safety check for logo: If logo.png is missing, it won't crash the script
    const logoUrl = chrome.runtime.getURL('logo.png');
    
    dashboard.innerHTML = `
      <div class="header">
        <img src="${logoUrl}" class="mini-logo" onerror="this.style.display='none'">
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
        <div id="tts-history" class="history-list">Waiting...</div>
      </div>
      <div class="volume-container">
        <label>Voice Volume</label>
        <input type="range" id="volume-slider" min="0" max="1" step="0.1" value="0.8">
      </div>
    `;

    // --- 3. THE ISOLATED STYLES ---
    const style = document.createElement('style');
    style.textContent = `
      #sova-dashboard {
        position: fixed; top: 20px; right: 20px; width: 280px;
        background: #1a6d7a; color: white; border-radius: 12px;
        padding: 15px; z-index: 999999; font-family: 'Segoe UI', sans-serif;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1);
        transition: opacity 0.4s ease, transform 0.4s ease;
        opacity: 1; pointer-events: auto;
      }
      .dashboard-hidden {
        opacity: 0 !important;
        transform: translateY(-20px);
        pointer-events: none !important;
      }
      .header { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; font-weight: bold; }
      .mini-logo { width: 24px; height: 24px; object-fit: contain; }
      .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px; }
      .stat-box { background: rgba(0,0,0,0.25); padding: 10px; border-radius: 8px; text-align: center; }
      label { font-size: 0.65rem; text-transform: uppercase; opacity: 0.7; display: block; margin-bottom: 2px; }
      .value { font-size: 0.95rem; font-weight: bold; color: #00f2ff; }
      .history-list { font-size: 0.8rem; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 6px; min-height: 30px; border-left: 3px solid #00f2ff; }
    `;

    shadow.appendChild(style);
    shadow.appendChild(dashboard);

    // --- 4. DATA LISTENER ---
    chrome.runtime.onMessage.addListener((message) => {
      if (message.type === "UPDATE_DASHBOARD") {
        const data = message.payload;
        const dashboardEl = shadow.getElementById('sova-dashboard');

        // Toggle visibility based on the 'show_dashboard' boolean from Python
        if (data.show_dashboard === false) {
            dashboardEl.classList.add('dashboard-hidden');
        } else {
            dashboardEl.classList.remove('dashboard-hidden');
        }

        // Update Text Content
        shadow.getElementById('expr-val').innerText = data.last_expression || "Neutral";
        shadow.getElementById('gest-val').innerText = data.last_gesture || "None";
        
        if (data.tts_history && data.tts_history.length > 0) {
            shadow.getElementById('tts-history').innerText = data.tts_history[0];
        }
      }
    });
}