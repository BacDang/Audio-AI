# 🎵 LapTrinhAmThanh - AI Music Genre Classification Web App

> Hệ thống AI phân loại và nhận diện thể loại âm nhạc tự động (Full Stack) sử dụng Machine Learning (SVM) kết hợp 60 đặc trưng âm học trích xuất từ Librosa.

---

## 🌟 Tính Năng Nổi Bật

- 🔮 **Nhận Diện Âm Nhạc Tự Động**: Tải lên file `.wav`, `.mp3`, `.ogg`, `.flac` để phân tích 10 thể loại nhạc (Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock).
- 🎛️ **Trích Xuất 60 Đặc Trưng Âm Học**: Tự động tính toán MFCCs, Tempo (BPM), RMS Energy, Spectral Centroid, Bandwidth, Rolloff, Harmonic & Percussive components.
- 📊 **Visualizer Sóng Âm Thời Gian Thực**: Trình phát nhạc Web Audio API với hiệu ứng visualizer dynamic canvas.
- 📚 **Bách Khoa Thể Loại Nhạc**: Tra cứu chi tiết thông tin, nhạc cụ, nghệ sĩ tiêu biểu và đặc trưng âm học của từng thể loại.
- 📈 **Lịch Sử & Thống Kê**: Tự động lưu trữ lịch sử phân tích vào SQLite, biểu đồ phân bố và thống kê độ tin cậy.
- ⚡ **RESTful API**: FastAPI endpoints đầy đủ tài liệu Swagger UI tại `/docs`.

---

## 🛠️ Công Nghệ Sử Dụng

- **Backend**: FastAPI, Uvicorn, SQLAlchemy, SQLite
- **AI & Audio**: Librosa, Scikit-learn, NumPy, Pandas, Joblib
- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism & Micro-animations), Vanilla JavaScript (Web Audio API)

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy

### 1. Cài đặt thư viện:
```bash
pip install -r requirements.txt
```

### 2. Khởi động ứng dụng Web:
```bash
python run_app.py
```
Ứng dụng sẽ tự động mở trình duyệt tại: `http://127.0.0.1:8000`

---

## 📡 API Endpoints

- `POST /api/predict`: Tải lên file âm thanh để nhận diện thể loại & đặc trưng.
- `POST /api/predict-demo`: Phân loại nhanh các file nhạc mẫu có sẵn.
- `GET /api/genres`: Lấy danh sách 10 thể loại và thông tin chi tiết.
- `GET /api/demo-samples`: Lấy danh sách các bài hát mẫu.
- `GET /api/history`: Xem lịch sử phân tích.
- `GET /api/stats`: Thống kê tổng hợp số liệu.
- `GET /docs`: Swagger UI tương tác API trực tiếp.
