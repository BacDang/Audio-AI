# 📋 PRD – Product Requirements Document
# Hệ Thống AI Phân Biệt Thể Loại Nhạc (LapTrinhAmThanh)

> **Phiên bản:** 1.0  
> **Ngày cập nhật:** 29/05/2026  
> **Tác giả:** Nhóm LapTrinhAmThanh  
> **Trạng thái:** Released

---

## 1. Tổng Quan Sản Phẩm

### 1.1 Mô Tả

**LapTrinhAmThanh** là hệ thống phân loại thể loại âm nhạc tự động sử dụng Machine Learning. Hệ thống nhận đầu vào là một file âm thanh (`.wav`, `.mp3`,...), phân tích các đặc trưng âm học và đưa ra dự đoán thể loại nhạc tương ứng trong số 10 thể loại được hỗ trợ.

Ứng dụng được xây dựng bằng Python, sử dụng mô hình **SVM (Support Vector Machine)** được huấn luyện trên bộ dữ liệu **GTZAN**, kết hợp thư viện **Librosa** để trích xuất đặc trưng âm thanh và **Gradio** để tạo giao diện web.

### 1.2 Mục Tiêu Sản Phẩm

| STT | Mục tiêu | Chỉ số đo lường |
|-----|----------|-----------------|
| 1 | Phân loại đúng thể loại nhạc từ file âm thanh | Accuracy ≥ 80% trên tập test |
| 2 | Thời gian xử lý chấp nhận được | ≤ 10 giây / bài |
| 3 | Giao diện dễ sử dụng, không cần kỹ thuật | Người dùng phổ thông có thể dùng ngay |
| 4 | Hệ thống chạy offline không cần internet | Chạy local hoàn toàn |

### 1.3 Phạm Vi Phiên Bản 1.0

**Bao gồm:**
- Trích xuất đặc trưng âm thanh từ file 30 giây
- Trích xuất đặc trưng theo đoạn 3 giây (10 đoạn/bài)
- Lưu đặc trưng vào CSV và MySQL
- Giao diện web Gradio để dự đoán
- Mô hình SVM đã huấn luyện sẵn (`.pkl`)

**Không bao gồm:**
- Mobile App
- API REST cho bên thứ ba
- Hỗ trợ streaming audio
- Training lại model từ giao diện

---

## 2. Người Dùng Mục Tiêu

### 2.1 Personas

**Persona 1 – Sinh viên / Nghiên cứu viên**
- Mục đích: Kiểm tra kết quả mô hình, học về audio ML
- Hành vi: Upload file nhạc, xem kết quả dự đoán, so sánh với nhãn thực

**Persona 2 – Giảng viên / Demo**
- Mục đích: Trình bày ứng dụng học máy trong lớp học
- Hành vi: Chạy `app.py`, mở browser, demo trực tiếp trước lớp

**Persona 3 – Developer**
- Mục đích: Tái sử dụng pipeline trích xuất đặc trưng cho dự án khác
- Hành vi: Import `feature_extraction.py`, tích hợp vào pipeline riêng

---

## 3. Yêu Cầu Chức Năng

### 3.1 Module Trích Xuất Đặc Trưng (`feature_extraction.py`)

| ID | Tính năng | Mô tả chi tiết | Ưu tiên |
|----|-----------|----------------|---------|
| F-01 | Load audio 30 giây | Dùng `librosa.load(duration=30.0)` để đọc đúng 30s đầu tiên | P0 |
| F-02 | Trích xuất Tempo | `librosa.beat.beat_track()` – nhịp điệu bài hát (BPM) | P0 |
| F-03 | Trích xuất Chroma STFT | `librosa.feature.chroma_stft()` – phân bố tần số theo 12 nốt nhạc | P0 |
| F-04 | Trích xuất RMS Energy | `librosa.feature.rms()` – năng lượng âm thanh | P0 |
| F-05 | Trích xuất Spectral Centroid | `librosa.feature.spectral_centroid()` – trọng tâm phổ tần số | P0 |
| F-06 | Trích xuất Spectral Bandwidth | `librosa.feature.spectral_bandwidth()` – độ rộng phổ tần số | P0 |
| F-07 | Trích xuất Rolloff | `librosa.feature.spectral_rolloff()` – điểm tần số rolloff 85% | P0 |
| F-08 | Trích xuất ZCR | `librosa.feature.zero_crossing_rate()` – tỷ lệ vượt qua 0 | P0 |
| F-09 | Trích xuất 20 MFCC | `librosa.feature.mfcc(n_mfcc=20)` – 20 hệ số cepstral (mean + var = 40 features) | P0 |
| F-10 | Trích xuất Harmony/Percussive | `librosa.effects.harmonic/percussive()` – tách thành phần hòa âm và tiết tấu | P0 |
| F-11 | Trích xuất Chroma CQT | `librosa.feature.chroma_cqt()` – phân bố tần số theo CQT | P0 |
| F-12 | Lưu CSV | `df.to_csv('example_features_with_path.csv')` với cột `filename`, `file_path`, `genre_label` | P0 |
| F-13 | Lưu MySQL | Ghi vào bảng `songs_features` trong database `music_db` qua SQLAlchemy | P1 |
| F-14 | Multiprocessing | Dùng `multiprocessing.Pool` với tối đa 16 core để xử lý song song | P1 |

**Tổng số đặc trưng được trích xuất: 59 features**
- 1 (`length`) + 2 (`chroma_stft`) + 2 (`rms`) + 2 (`spectral_centroid`) + 2 (`spectral_bandwidth`) + 2 (`rolloff`) + 2 (`zcr`) + 2 (`harmony`) + 2 (`percussive`) + 1 (`tempo`) + 40 (`mfcc1-20 mean+var`) + 2 (`chroma_cqt`) = **60 features**

---

### 3.2 Module Trích Xuất 3 Giây (`extract_features_3sec.py`)

| ID | Tính năng | Mô tả chi tiết | Ưu tiên |
|----|-----------|----------------|---------|
| F-15 | Chia bài thành 10 đoạn 3s | Mỗi bài 30s → 10 segment, offset = `idx * 3.0` | P0 |
| F-16 | Ép `length = 66149` | Chuẩn hóa độ dài mẫu cho đoạn 3s (sample count cố định) | P0 |
| F-17 | Đặt tên segment | Format: `{base_name}.{segment_idx}` (ví dụ: `blues.00000.0`) | P0 |
| F-18 | Xuất CSV 3s | Lưu `example_features_3_sec.csv` với ~10.000 hàng (1000 bài × 10 đoạn) | P0 |
| F-19 | Gán nhãn tự động | Lấy prefix trước dấu `.` đầu tiên của filename làm `genre_label` | P0 |

---

### 3.3 Module Ứng Dụng Web (`app.py`)

| ID | Tính năng | Mô tả chi tiết | Ưu tiên |
|----|-----------|----------------|---------|
| F-20 | Load model `.pkl` | Tự động load `music_scaler.pkl`, `music_svm_model.pkl`, `music_categories.pkl` khi khởi động | P0 |
| F-21 | Upload file nhạc | Gradio `gr.Audio(type="filepath")` – hỗ trợ kéo thả | P0 |
| F-22 | Trích xuất đặc trưng realtime | Chạy `predict_genre_of_file()` với file vừa upload | P0 |
| F-23 | Chuẩn hóa đặc trưng | `scaler.transform()` đảm bảo đúng thứ tự cột theo `feature_names_in_` | P0 |
| F-24 | Dự đoán thể loại | `model.predict()` → map qua `categories_list` → trả về tên thể loại viết hoa | P0 |
| F-25 | Hiển thị kết quả | Textbox với format: `🎵 Dự đoán thể loại: {GENRE}` | P0 |
| F-26 | Xử lý lỗi | Bắt exception, trả về thông báo lỗi thân thiện, log chi tiết ra console | P1 |
| F-27 | Chế độ share | Tham số `share=False` (local), đổi `True` để lấy link Gradio public | P2 |

---

## 4. Yêu Cầu Phi Chức Năng

### 4.1 Hiệu Năng

| Yêu cầu | Giá trị mục tiêu |
|---------|-----------------|
| Thời gian trích xuất đặc trưng 1 bài (30s) | ≤ 5 giây |
| Thời gian trích xuất toàn bộ dataset (1000 bài) | ≤ 10 phút (với 16 core) |
| Thời gian dự đoán trên giao diện web | ≤ 10 giây |
| RAM sử dụng khi chạy app.py | ≤ 2 GB |

### 4.2 Môi Trường

| Thành phần | Yêu cầu |
|-----------|---------|
| Python | ≥ 3.9 |
| OS | Windows / macOS / Linux |
| RAM | ≥ 4 GB |
| Database | MySQL 8.0+ (tùy chọn) |
| Browser | Chrome / Firefox / Edge (cho Gradio UI) |

### 4.3 Bảo Mật

- Không lưu file âm thanh của người dùng sau khi dự đoán
- Thông tin đăng nhập MySQL được hard-code trong `feature_extraction.py` (cần chuyển sang biến môi trường trong production)
- Ứng dụng chạy `share=False` theo mặc định (local only)

---

## 5. Luồng Xử Lý Chính

### 5.1 Luồng Huấn Luyện (Offline)

```
genres_original/
    ├── blues/     (100 file .wav)
    ├── classical/ (100 file .wav)
    ├── ...
    └── rock/      (100 file .wav)
          │
          ▼
feature_extraction.py          extract_features_3sec.py
(trích xuất 30s/bài)           (trích xuất 10×3s/bài)
          │                              │
          ▼                              ▼
example_features_with_path.csv    example_features_3_sec.csv
songs_features (MySQL)
          │
          ▼
ml_classification.ipynb
(Train SVM, GridSearchCV, evaluate)
          │
          ▼
music_svm_model.pkl + music_scaler.pkl + music_categories.pkl
```

### 5.2 Luồng Dự Đoán (Online – app.py)

```
User upload file nhạc
          │
          ▼
librosa.load(duration=30.0)
          │
          ▼
Trích xuất 60 đặc trưng
(tempo, chroma, rms, mfcc×20, ...)
          │
          ▼
pd.DataFrame → reindex theo scaler.feature_names_in_
          │
          ▼
scaler.transform()
          │
          ▼
model_svm.predict()
          │
          ▼
categories_list[pred_code].upper()
          │
          ▼
"🎵 Dự đoán thể loại: BLUES"
```

---

## 6. Thể Loại Nhạc Được Hỗ Trợ

| STT | Thể loại | Số bài (train) |
|-----|----------|---------------|
| 1 | Blues | 100 |
| 2 | Classical | 100 |
| 3 | Country | 100 |
| 4 | Disco | 100 |
| 5 | Hip-hop | 100 |
| 6 | Jazz | 100 |
| 7 | Metal | 100 |
| 8 | Pop | 100 |
| 9 | Reggae | 100 |
| 10 | Rock | 100 |
| **Tổng** | **10 thể loại** | **1.000 bài** |

> **Nguồn dữ liệu:** GTZAN Dataset – chuẩn nghiên cứu phổ biến trong Music Information Retrieval (MIR)

---

## 7. Các Ràng Buộc Kỹ Thuật

| Ràng buộc | Lý do |
|-----------|-------|
| File nhạc phải có thời lượng ≥ 30 giây | `librosa.load(duration=30.0)` lấy đúng 30s đầu |
| Model chỉ nhận đúng 60 features theo thứ tự | `scaler.feature_names_in_` kiểm tra thứ tự cột |
| Cần có 3 file `.pkl` trước khi chạy `app.py` | App exit ngay nếu load model thất bại |
| MySQL cần chạy trước khi chạy `feature_extraction.py` | Kết nối `127.0.0.1:3306/music_db` |
| Không có Chroma CQT trong `extract_features_3sec.py` | File 3s đủ 60 features kể cả chroma_cqt |

---

## 8. Rủi Ro & Giảm Thiểu

| Rủi ro | Mức độ | Giảm thiểu |
|--------|--------|-----------|
| File nhạc < 30 giây → thiếu data | Cao | Thêm validation trước khi predict |
| MySQL không kết nối được | Trung bình | Feature extraction vẫn lưu CSV, MySQL là tùy chọn |
| Model overfit trên GTZAN | Trung bình | Đánh giá trên `music_test/` riêng biệt |
| Hard-code password MySQL trong code | Cao | Chuyển sang `.env` file hoặc `os.environ` |

---

## 9. Định Nghĩa Hoàn Thành (Definition of Done)

- [ ] `feature_extraction.py` chạy thành công trên toàn bộ `genres_original/`
- [ ] `extract_features_3sec.py` tạo ra file CSV với đúng 10.000 hàng
- [ ] `ml_classification.ipynb` train xong và xuất 3 file `.pkl`
- [ ] `app.py` khởi động được và load model thành công
- [ ] Giao diện Gradio mở trên browser, dự đoán đúng ≥ 3/5 file test thủ công
- [ ] Tài liệu README, PRD, Wiki đầy đủ

---

*PRD này được tạo dựa trên phân tích mã nguồn thực tế của project LapTrinhAmThanh.*
