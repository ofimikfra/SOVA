# Sova
Requirement for CSIT321 Capstone Project

## What is Sova?

## Features
- Facial expression & gesture detection.
- WebSocket results include both `description` and a new `summary` field
  (used by the dashboard/popup UI).

## How to run
pip install -r requirements.txt
python src/main.py


# TEST - 1 - How to run!
python3 -m venv venv   
source venv/bin/activate

# flow 
Google Meet Page
    ↓
Chrome Extension (content script)
    ↓  (scrapes subtitles every X ms)
Background script
    ↓  (POST JSON)
Flask (extension/app.py)
    ↓
Subtitle processor
    ↓
Your existing processor.py buffers
    ↓
flushAll() every interval
    ↓
TTS speaks multimodal summary

# 1. Install Ollama
# https://ollama.com/download  (macOS / Windows / Linux)

# 2. Pull the model (one-time, ~2GB download)
ollama pull llama3.2:3b

# 3. Start Ollama before running SOVA
ollama serve

# 4. Run SOVA
python main.py
