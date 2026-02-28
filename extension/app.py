import sys
import os
from flask import Flask, jsonify
from flask_cors import CORS
import threading

# This tells Python to look in the current folder for your files
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import main  # This should now link to your main_logic.py
app = Flask(__name__)
CORS(app) # This allows the Chrome Extension to talk to the server

@app.route('/start', methods=['GET'])
def start_sova():
    # Run the vision loop in a separate thread so the server doesn't freeze
    thread = threading.Thread(target=main.run_system)
    thread.start()
    return jsonify({"status": "SOVA Started"}), 200

if __name__ == '__main__':
    app.run(port=5000)