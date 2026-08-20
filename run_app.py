import os
import sys
import time
import webbrowser
import threading
import uvicorn

# Fix Unicode trên terminal Windows
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def open_browser(url: str, delay: float = 1.8):
    """Mở trình duyệt sau delay giây."""
    def _open():
        time.sleep(delay)
        print(f"\n🌐 Đang mở trình duyệt: {url}")
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()

if __name__ == "__main__":
    print("=" * 65)
    print("🚀 LapTrinhAmThanh – AI Music Genre Classifier (Full Stack 2.0)")
    print("=" * 65)
    print(f"📁 Project Root : {BASE_DIR}")
    print(f"🔗 Web App      : http://127.0.0.1:8000")
    print(f"📋 Swagger Docs : http://127.0.0.1:8000/docs")
    print(f"📊 API Stats    : http://127.0.0.1:8000/api/stats")
    print("=" * 65)
    print("⏳ Đang khởi động server, vui lòng chờ...\n")

    open_browser("http://127.0.0.1:8000")

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
        access_log=True
    )
