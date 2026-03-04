import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 170)
engine.setProperty('volume', 1.0)

def speak(text):
    if not text:
        return
    engine.say(text)
    engine.runAndWait()