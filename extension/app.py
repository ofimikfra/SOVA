import sys
import os
import threading
from flask import Flask, jsonify
from flask_cors import CORS

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import main
app = Flask(__name__)
CORS(app)
sova_state = {
    "status": "Idle",
    "last_expression": "Neutral",
    "last_gesture": "No Gesture",
    "last_action": "Person Center",
    "tts_history": [],
    "is_running": False
}
@app.route('/start', methods=['GET'])
def start_sova():
    if not sova_state["is_running"]:
        sova_state["is_running"] = True
        sova_state["status"] = "Active"

        # Start the vision loop in a background thread
        # We pass a callback function so main.py can update the state here
        thread = threading.Thread(target=main.run_system, args=(update_state_callback,))
        thread.daemon = True
        thread.start()

        return jsonify({"status": "SOVA Started"}), 200
    return jsonify({"status": "Already Running"}), 200
@app.route('/get_status', methods=['GET'])
def get_status():
    """The Chrome Extension will call this every 1 second to update the dashboard."""
    return jsonify(sova_state), 200
def update_state_callback(expr, gest, act, tts_text=None):
    global sova_state
    sova_state["last_expression"] = expr
    sova_state["last_gesture"] = gest
    sova_state["last_action"] = act

    if tts_text:
        sova_state["tts_history"].insert(0, tts_text)
        sova_state["tts_history"] = sova_state["tts_history"][:5]


if __name__ == '__main__':
    print("SOVA Backend running on http://127.0.0.1:5000")
    app.run(port=5000, debug=False, threaded=True)