import webview
import threading
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


class SovaApi:

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._running = False

    def get_status(self) -> dict:
        return {"running": self._running}

    def start(self) -> dict:
        if self._running:
            return {"ok": False, "reason": "Already running"}

        self._stop_event = threading.Event()
        self._running    = True

        def _run():
            try:
                from main import run_system
                run_system(
                    source      = "webcam",
                    headless    = True,          # no cv2.imshow()
                    stop_event  = self._stop_event,
                )
            except Exception as e:
                print(f"[APP] run_system error: {e}")
            finally:
                self._running = False
                print("[APP] SOVA engine stopped.")

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        print("[APP] SOVA engine started")
        return {"ok": True}

    def stop(self) -> dict:
        if self._stop_event:
            self._stop_event.set()   # signals the while loop to exit cleanly
        self._running = False
        print("[APP] SOVA engine stopping...")
        return {"ok": True}


def main():
    dashboard = os.path.join(ROOT, "extension", "dashboard.html")

    api = SovaApi()

    window = webview.create_window(
        title            = "SOVA",
        url              = f"file://{dashboard}",
        js_api           = api,
        width            = 760,
        height           = 860,
        min_size         = (480, 600),
        background_color = "#0f0f0f",
        easy_drag = False,
    )

    webview.start(debug=False)


if __name__ == "__main__":
    main()