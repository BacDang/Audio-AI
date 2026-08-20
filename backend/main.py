import os
import sys
import time
import shutil
import tempfile
from typing import Optional

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.ml_engine import ml_engine, GENRE_METADATA
from backend.audio_processor import audio_processor
from backend.database import HistoryRepository

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')
MUSIC_TEST_DIR = os.path.join(BASE_DIR, 'music_test')

app = FastAPI(
    title="LapTrinhAmThanh - Music Genre Classifier API",
    description="Hệ thống AI phân loại thể loại âm nhạc tự động với mô hình SVM & Librosa",
    version="2.0.0"
)

# Kích hoạt CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Phục vụ file nhạc mẫu (mount trước static frontend để tránh xung đột route)
if os.path.exists(MUSIC_TEST_DIR):
    app.mount("/demo-audio", StaticFiles(directory=MUSIC_TEST_DIR), name="demo-audio")

# Mount static frontend assets (css, js, images)
if os.path.exists(FRONTEND_DIR):
    app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND_DIR, "css")), name="css")
    app.mount("/js", StaticFiles(directory=os.path.join(FRONTEND_DIR, "js")), name="js")


# ─────────────────────────────────────────────────────────────
# HEALTH & INFO
# ─────────────────────────────────────────────────────────────
@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": ml_engine.is_loaded,
        "categories": ml_engine.categories,
        "total_categories": len(ml_engine.categories),
        "timestamp": time.time(),
        "version": "2.0.0"
    }


# ─────────────────────────────────────────────────────────────
# GENRE CATALOG
# ─────────────────────────────────────────────────────────────
@app.get("/api/genres")
def get_genres():
    """Lấy danh sách và thông tin chi tiết 10 thể loại nhạc."""
    return {
        "genres": [
            {"id": key, **meta}
            for key, meta in GENRE_METADATA.items()
        ]
    }


# ─────────────────────────────────────────────────────────────
# DEMO SAMPLES
# ─────────────────────────────────────────────────────────────
@app.get("/api/demo-samples")
def get_demo_samples():
    """Lấy danh sách các bài hát mẫu trong thư mục music_test."""
    samples = []
    if os.path.exists(MUSIC_TEST_DIR):
        for genre_folder in sorted(os.listdir(MUSIC_TEST_DIR)):
            folder_path = os.path.join(MUSIC_TEST_DIR, genre_folder)
            if os.path.isdir(folder_path):
                for f in sorted(os.listdir(folder_path)):
                    if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                        meta = GENRE_METADATA.get(genre_folder, {})
                        samples.append({
                            "genre": genre_folder,
                            "filename": f,
                            "audio_url": f"/demo-audio/{genre_folder}/{f}",
                            "display_name": f"{meta.get('icon','🎵')} {genre_folder.upper()} – {f}",
                            "icon": meta.get("icon", "🎵"),
                            "color": meta.get("color", "#6366f1"),
                        })
    return {"samples": samples, "total": len(samples)}


# ─────────────────────────────────────────────────────────────
# PREDICT – FILE UPLOAD
# ─────────────────────────────────────────────────────────────
@app.post("/api/predict")
async def predict_audio(file: UploadFile = File(...)):
    """Nhận file âm thanh upload, trích xuất đặc trưng và dự đoán thể loại."""
    start_time = time.time()
    valid_extensions = ('.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac')
    filename = file.filename or "uploaded_audio.wav"

    if not any(filename.lower().endswith(ext) for ext in valid_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng không hỗ trợ. Dùng: {', '.join(valid_extensions)}"
        )

    if not ml_engine.is_loaded:
        raise HTTPException(status_code=503, detail="Mô hình AI chưa sẵn sàng. Vui lòng thử lại sau.")

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(temp_path)
        features, metrics = audio_processor.extract_features_and_metrics(temp_path, duration=30.0)
        prediction_result = ml_engine.predict(features)
        processing_time_ms = (time.time() - start_time) * 1000

        history_record = HistoryRepository.add_entry(
            filename=filename,
            file_size_bytes=file_size,
            duration_seconds=metrics.get("duration_seconds", 0.0),
            predicted_genre=prediction_result["predicted_genre"],
            predicted_genre_vi=prediction_result["predicted_genre_vi"],
            confidence_percentage=prediction_result["confidence_percentage"],
            tempo_bpm=metrics.get("tempo_bpm", 0.0),
            rms_energy=metrics.get("rms_energy", 0.0),
            spectral_centroid_hz=metrics.get("spectral_centroid_hz", 0.0),
            processing_time_ms=processing_time_ms
        )

        return {
            "success": True,
            "filename": filename,
            "file_size_bytes": file_size,
            "processing_time_ms": round(processing_time_ms, 1),
            "prediction": prediction_result,
            "metrics": metrics,
            "history_id": history_record.get("id")
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích âm thanh: {str(e)}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────
# PREDICT – DEMO SAMPLE
# ─────────────────────────────────────────────────────────────
@app.post("/api/predict-demo")
def predict_demo_sample(genre: str = Query(...), filename: str = Query(...)):
    """Dự đoán bài hát mẫu có sẵn trong thư mục music_test."""
    start_time = time.time()

    # Validate path traversal
    if ".." in genre or ".." in filename:
        raise HTTPException(status_code=400, detail="Đường dẫn không hợp lệ.")

    audio_path = os.path.join(MUSIC_TEST_DIR, genre, filename)

    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail=f"Không tìm thấy file mẫu: {genre}/{filename}")

    if not ml_engine.is_loaded:
        raise HTTPException(status_code=503, detail="Mô hình AI chưa sẵn sàng.")

    try:
        file_size = os.path.getsize(audio_path)
        features, metrics = audio_processor.extract_features_and_metrics(audio_path, duration=30.0)
        prediction_result = ml_engine.predict(features)
        processing_time_ms = (time.time() - start_time) * 1000

        HistoryRepository.add_entry(
            filename=f"[Demo] {genre}/{filename}",
            file_size_bytes=file_size,
            duration_seconds=metrics.get("duration_seconds", 0.0),
            predicted_genre=prediction_result["predicted_genre"],
            predicted_genre_vi=prediction_result["predicted_genre_vi"],
            confidence_percentage=prediction_result["confidence_percentage"],
            tempo_bpm=metrics.get("tempo_bpm", 0.0),
            rms_energy=metrics.get("rms_energy", 0.0),
            spectral_centroid_hz=metrics.get("spectral_centroid_hz", 0.0),
            processing_time_ms=processing_time_ms
        )

        return {
            "success": True,
            "filename": filename,
            "genre": genre,
            "audio_url": f"/demo-audio/{genre}/{filename}",
            "file_size_bytes": file_size,
            "processing_time_ms": round(processing_time_ms, 1),
            "prediction": prediction_result,
            "metrics": metrics,
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích file mẫu: {str(e)}")


# ─────────────────────────────────────────────────────────────
# HISTORY & STATS
# ─────────────────────────────────────────────────────────────
@app.get("/api/history")
def get_history(limit: int = Query(default=50, ge=1, le=200)):
    return {"history": HistoryRepository.get_all(limit=limit)}


@app.delete("/api/history/{entry_id}")
def delete_history_entry(entry_id: int):
    success = HistoryRepository.delete_entry(entry_id)
    if not success:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi.")
    return {"success": True, "message": "Đã xóa bản ghi lịch sử."}


@app.get("/api/stats")
def get_statistics():
    return HistoryRepository.get_stats()


# ─────────────────────────────────────────────────────────────
# SERVE FRONTEND INDEX.HTML  (phải đặt CUỐI CÙNG)
# ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
@app.get("/{full_path:path}", response_class=HTMLResponse)
async def serve_spa(full_path: str = ""):
    """Phục vụ Single Page App cho mọi route không phải API."""
    # Không intercept API routes
    if full_path.startswith("api/") or full_path.startswith("demo-audio/") \
            or full_path.startswith("css/") or full_path.startswith("js/") \
            or full_path == "docs" or full_path == "openapi.json" or full_path == "redoc":
        raise HTTPException(status_code=404)

    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>Frontend chưa được build. Vui lòng kiểm tra thư mục frontend/</h1>", status_code=503)

    with open(index_path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
