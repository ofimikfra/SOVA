
# SOVA
An AI-powered video conferencing accessibility tool capable of real-time interpretation of facial expressions and gestures, delivered as audio descriptions.

## Requirements

- Python 3.11+
- macOS, Windows, or Linux
- **For macOS audio capture**: macOS 13+ (ScreenCaptureKit, no setup) or [BlackHole](https://existential.audio/blackhole/) on macOS 12 and earlier
- **For Windows audio**: WASAPI loopback (built-in)
- **For Linux audio**: PulseAudio or PipeWire monitor device

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/ofimikfra/SOVA.git
cd sova
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Install [Ollama](https://ollama.com/download)
> SOVA still runs if Ollama isn't installed, but descriptions will fall back to pre-made templates.

**Download from the website: https://ollama.com/download**

**For Windows:**
```bash
irm https://ollama.com/install.ps1 | iex
```
**For macOS & Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 4. Install [BlackHole](https://existential.audio/blackhole/) (macOS 12 or earlier)
Download [BlackHole 2ch](https://existential.audio/blackhole/) and set it as your system audio output. SOVA will detect it automatically. If you are using macOS 13+, skip this step.

## Chrome Extension
1. Open `chrome://extensions` and enable Developer mode.
2. Click **Load unpacked** and select the `extension/` folder.
3. Join a Google Meet call; the SOVA overlay will appear automatically.
4. Click the SOVA toolbar icon (alternatively, `⌥ + S` on macOS or `alt + S` on Windows & Linux) to access quick settings.

## Usage

Run this in the terminal:
```bash
python app.py
```
Once the dashboard opens, click the ***Start SOVA*** button at the top (alternatively, `⌥ + S` on macOS or `alt + S` on Windows & Linux) to start the software.

