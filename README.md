# Sova
Requirement for CSIT321 Capstone Project

## What is Sova?

## Features
- Facial expression & gesture detection.

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
