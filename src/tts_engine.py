import pyttsx3
import threading

class TTSEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 170)
        self.engine.setProperty('volume', 1.0)
        self.lock = threading.Lock()
        self.is_speaking = False

    def speak(self, text):
        if not text:
            return
        def _say():
            self.is_speaking = True
            with self.lock:
                self.engine.say(text)
                self.engine.runAndWait()
            self.is_speaking = False
        threading.Thread(target=_say, daemon=True).start()

_engine = TTSEngine()

def speak(text):
    _engine.speak(text)
