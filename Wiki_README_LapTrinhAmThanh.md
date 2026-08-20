# 📖 WIKI – Tài Liệu Hướng Dẫn Đầy Đủ
# LapTrinhAmThanh – AI Phân Biệt Thể Loại Nhạc

> **Phiên bản:** 1.0 | **Cập nhật:** 29/05/2026

---

## Mục Lục

1. [Giới thiệu dự án](#1-giới-thiệu-dự-án)
2. [Yêu cầu cài đặt](#2-yêu-cầu-cài-đặt)
3. [Cài đặt môi trường](#3-cài-đặt-môi-trường)
4. [Cấu trúc thư mục](#4-cấu-trúc-thư-mục)
5. [Hướng dẫn chạy từng module](#5-hướng-dẫn-chạy-từng-module)
6. [Giải thích chi tiết từng file](#6-giải-thích-chi-tiết-từng-file)
7. [Các đặc trưng âm thanh được trích xuất](#7-các-đặc-trưng-âm-thanh-được-trích-xuất)
8. [Cơ sở dữ liệu MySQL](#8-cơ-sở-dữ-liệu-mysql)
9. [Mô hình Machine Learning](#9-mô-hình-machine-learning)
10. [Giao diện Web Gradio](#10-giao-diện-web-gradio)
11. [Xử lý lỗi thường gặp](#11-xử-lý-lỗi-thường-gặp)
12. [Thư viện và dependencies](#12-thư-viện-và-dependencies)

---

## 1. Giới Thiệu Dự Án

**LapTrinhAmThanh** là hệ thống phân loại thể loại âm nhạc tự động sử dụng Machine Learning, được xây dựng bằng Python.

### Tính năng chính

- Phân tích file âm thanh và dự đoán 1 trong 10 thể loại nhạc
- Trích xuất 60 đặc trưng âm học từ thư viện **Librosa**
- Sử dụng mô hình **SVM (Support Vector Machine)** để phân loại
- Giao diện web trực quan bằng **Gradio** (kéo thả file, click dự đoán)
- Hỗ trợ lưu dữ liệu vào **MySQL** và **CSV**
- Xử lý song song (multiprocessing) để tăng tốc trích xuất

### 10 thể loại được hỗ trợ

`blues` | `classical` | `country` | `disco` | `hiphop` | `jazz` | `metal` | `pop` | `reggae` | `rock`

---

## 2. Yêu Cầu Cài Đặt

### Phần mềm bắt buộc

| Phần mềm | Phiên bản tối thiểu | Tải về |
|---------|-------------------|--------|
| Python | 3.9+ | https://python.org |
| pip | 21+ | (đi kèm Python) |

### Phần mềm tùy chọn

| Phần mềm | Mục đích | Ghi chú |
|---------|---------|---------|
| MySQL Server | Lưu features vào database | Chỉ cần cho `feature_extraction.py` |
| JupyterLab | Chạy `ml_classification.ipynb` | Đã có trong `requirements.txt` |

### Phần cứng khuyến nghị

| Thành phần | Tối thiểu | Khuyến nghị |
|-----------|-----------|------------|
| RAM | 4 GB | 8 GB+ |
| CPU | 4 cores | 8+ cores (cho multiprocessing) |
| Ổ cứng | 5 GB trống | 10 GB+ |

---

## 3. Cài Đặt Môi Trường

### Bước 1: Clone hoặc mở project

```bash
cd ~/Desktop/LTAT/LapTrinhAmThanh
```

### Bước 2: Tạo virtual environment

```bash
# Tạo .venv
python -m venv .venv

# Kích hoạt (Windows)
.venv\Scripts\activate

# Kích hoạt (macOS / Linux)
source .venv/bin/activate
```

### Bước 3: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Chi tiết `requirements.txt`:
```
numpy==2.4.6          # Xử lý mảng đa chiều, ma trận
pandas==3.0.3         # Đọc/ghi CSV, thao tác DataFrame
scipy                 # Toán học nâng cao (hỗ trợ librosa, sklearn)
matplotlib==3.10.0    # Vẽ biểu đồ cơ bản
seaborn==0.13.2       # Vẽ Confusion Matrix, Heatmap
scikit-learn==1.6.0   # SVM, StandardScaler, train/test split
xgboost==2.1.3        # XGBoost (có trong notebook)
librosa==0.11.0       # Phân tích và trích xuất đặc trưng âm thanh
gradio==6.15.2        # Giao diện web UI
joblib==1.5.3         # Lưu/tải model .pkl
jupyterlab==4.3.0     # Môi trường chạy notebook .ipynb
```

### Bước 4: (Tùy chọn) Thiết lập MySQL

```sql
-- Tạo database
CREATE DATABASE music_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Tạo user (nếu không dùng root)
CREATE USER 'music_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON music_db.* TO 'music_user'@'localhost';
FLUSH PRIVILEGES;
```

> ⚠️ **Quan trọng:** Nếu đổi thông tin đăng nhập, cập nhật trong `feature_extraction.py` tại:
> ```python
> username = "root"
> password = quote_plus("22092005@Dung")
> host = "127.0.0.1"
> port = "3306"
> database_name = "music_db"
> ```

---

## 4. Cấu Trúc Thư Mục

```
LapTrinhAmThanh/
│
├── .venv/                              # Virtual environment (không commit)
│
├── genres_original/                    # Dataset GTZAN – 1000 bài WAV (30s)
│   ├── blues/         → blues.00000.wav ... blues.00099.wav
│   ├── classical/     → classical.00000.wav ... classical.00099.wav
│   ├── country/       → ...
│   ├── disco/         → ...
│   ├── hiphop/        → ...
│   ├── jazz/          → ...
│   ├── metal/         → ...
│   ├── pop/           → ...
│   ├── reggae/        → ...
│   └── rock/          → rock.00000.wav ... rock.00099.wav
│
├── music_test/                         # Dataset test độc lập
│   ├── blues/
│   ├── classical/
│   ├── disco/
│   ├── hiphop/
│   ├── jazz/
│   ├── metal/
│   ├── pop/
│   ├── reggae/
│   └── rock/
│
├── ── SCRIPTS ──
├── feature_extraction.py               # Trích xuất 60 features / bài 30s → CSV + MySQL
├── extract_features_3sec.py            # Trích xuất 60 features / đoạn 3s → CSV
├── ml_classification.ipynb             # Huấn luyện và đánh giá mô hình SVM
├── app.py                              # Web app Gradio – dự đoán thể loại
│
├── ── DATA ──
├── example_features.csv                # Features mẫu (format cơ bản)
├── example_features_3_sec.csv          # ~10.000 hàng × 61 cột (3s segments)
├── example_features_with_path.csv      # ~1.000 hàng × 63 cột (30s + path)
│
├── ── MODELS ──
├── music_svm_model.pkl                 # Mô hình SVM đã huấn luyện
├── music_scaler.pkl                    # StandardScaler đã fit trên train data
├── music_categories.pkl                # ['blues', 'classical', ..., 'rock']
│
├── ── DOCUMENTATION ──
├── Bao_cao_Testcase_Phan_Cap.csv       # Test case phân cấp
├── Dump20260528.sql                    # MySQL dump
└── requirements.txt                    # Dependencies
```

---

## 5. Hướng Dẫn Chạy Từng Module

### 5.1 Trích Xuất Đặc Trưng 30 Giây

**Mục đích:** Xử lý toàn bộ `genres_original/`, tạo CSV và ghi vào MySQL.

```bash
# Đảm bảo đang ở thư mục gốc và đã kích hoạt .venv
python feature_extraction.py
```

**Đầu ra:**
- File: `example_features_with_path.csv` (~1.000 hàng × 63 cột)
- Database: bảng `songs_features` trong MySQL `music_db`

**Console output mẫu:**
```
🚀 Tìm thấy 1000 bài hát.
⚡ Đang dùng 16 nhân CPU...
✅ Đã trích xuất xong!
💾 Đã lưu CSV: example_features_with_path.csv
✅ Kết nối MySQL thành công!
🎉 Đã lưu dữ liệu vào MySQL!
🚀 HOÀN THÀNH!
```

---

### 5.2 Trích Xuất Đặc Trưng 3 Giây

**Mục đích:** Băm mỗi bài 30s thành 10 đoạn 3s, tạo dataset lớn hơn (10.000 samples).

```bash
python extract_features_3sec.py
```

**Đầu ra:**
- File: `example_features_3_sec.csv` (~10.000 hàng × 61 cột)

**Console output mẫu:**
```
🚀 Tìm thấy 1000 bài hát gốc.
🔥 Sẵn sàng băm nhỏ thành 10000 đoạn 3 giây sử dụng 8 nhân CPU...
✅ Đã trích xuất xong toàn bộ 10.000 đoạn! Đang gộp dữ liệu...
🎉 THÀNH CÔNG RỰC RỠ! Bản 3s đã được tạo lập.
💾 File lưu tại: /path/to/example_features_3_sec.csv
📊 Tổng quy mô dữ liệu thu được: 10000 hàng, 61 cột.
```

---

### 5.3 Huấn Luyện Mô Hình

**Mục đích:** Đọc CSV đặc trưng, train SVM, lưu 3 file `.pkl`.

```bash
# Khởi động JupyterLab
jupyter lab

# Mở file ml_classification.ipynb
# Chạy tuần tự các ô code từ trên xuống dưới
```

**Kết quả sau khi chạy notebook:**
- `music_svm_model.pkl` – mô hình SVM
- `music_scaler.pkl` – StandardScaler
- `music_categories.pkl` – danh sách thể loại

> ⚠️ **Bắt buộc phải chạy notebook này trước khi chạy `app.py`!**

---

### 5.4 Chạy Web App

**Mục đích:** Khởi động giao diện web để dự đoán thể loại nhạc.

```bash
python app.py
```

**Sau khi khởi động:**
```
Đang nạp cấu hình AI từ file .pkl... Vui lòng đợi...
Nạp AI thành công! Hệ thống sẵn sàng hoạt động.
Running on local URL:  http://127.0.0.1:7860
```

Mở trình duyệt, truy cập: **http://127.0.0.1:7860**

**Cách sử dụng:**
1. Kéo thả file nhạc vào ô "Tải file nhạc của bạn lên tại đây"
2. Nhấn nút **"🔮 Bắt đầu phân tích & Dự đoán"**
3. Chờ khoảng 5–10 giây
4. Kết quả hiện ở ô bên phải: `🎵 Dự đoán thể loại: BLUES`

**Muốn chia sẻ link cho người khác:**
```python
# Trong app.py, đổi dòng cuối thành:
demo.launch(share=True)
# Gradio sẽ tạo link dạng: https://xxxx.gradio.live
```

---

### 5.5 Thứ Tự Chạy Đúng (Lần Đầu Cài)

```
Bước 1: pip install -r requirements.txt
Bước 2: python feature_extraction.py     ← Tạo CSV + MySQL
Bước 3: python extract_features_3sec.py  ← Tạo CSV 3s (tùy chọn)
Bước 4: jupyter lab → ml_classification.ipynb → Run All  ← Tạo .pkl
Bước 5: python app.py                    ← Chạy web app
```

---

## 6. Giải Thích Chi Tiết Từng File

### 6.1 `feature_extraction.py`

**Hàm chính: `get_all_musical_features()`**

```python
def get_all_musical_features(
    audio_path,      # Đường dẫn file .wav
    song_name,       # Tên bài (dùng làm index)
    start=0,         # Offset giây bắt đầu (mặc định 0)
    chroma_method_list=['cqt']  # Phương pháp chroma thêm
)
```

Hàm này load file âm thanh và trích xuất 60 đặc trưng. Kết quả là một DataFrame 1 hàng.

**Xử lý song song:**

```python
# Sử dụng multiprocessing để xử lý nhiều file cùng lúc
num_processors = min(16, cpu_count())  # Tối đa 16 core
with Pool(processes=num_processors) as pool:
    df_list = pool.map(extract_single_song, wav_list)
```

> **Lưu ý:** Trên Windows, code multiprocessing phải nằm trong `if __name__ == '__main__':` – đã được đảm bảo trong file.

---

### 6.2 `extract_features_3sec.py`

**Khác biệt so với `feature_extraction.py`:**

| Tiêu chí | feature_extraction.py | extract_features_3sec.py |
|---------|----------------------|--------------------------|
| Thời lượng | 30 giây / bài | 3 giây / đoạn |
| Số mẫu đầu ra | ~1.000 hàng | ~10.000 hàng |
| `length` | Tự nhiên (~661.794) | Ép cứng = `66149` |
| Lưu MySQL | Có | Không |
| Offset | 0 | `idx * 3.0` (idx từ 0 đến 9) |

**Tại sao ép `length = 66149`?**

Vì dataset GTZAN 3-second gốc có cột `length = 66149` (3s × 22050 Hz ≈ 66.150 samples). Ép cứng giá trị này giúp đặc trưng nhất quán với mô hình đã train trên dataset 3s gốc.

---

### 6.3 `app.py`

**Hàm dự đoán: `predict_genre_of_file()`**

Hàm này thực hiện chính xác cùng pipeline trích xuất như `feature_extraction.py`, nhưng chỉ cho 1 file tại thời điểm runtime:

```python
# Bước quan trọng: reindex theo đúng thứ tự cột của scaler
if hasattr(scaler, 'feature_names_in_'):
    df_song = df_song.reindex(columns=scaler.feature_names_in_)
```

> ⚠️ Nếu bỏ dòng này, predict có thể sai hoặc crash do sai thứ tự cột!

**Graceful Error Handling:**

```python
try:
    prediction = predict_genre_of_file(...)
    return f"🎵 Dự đoán thể loại: {prediction}"
except Exception as e:
    error_msg = traceback.format_exc()
    print(f"Chi tiết lỗi hệ thống:\n{error_msg}")  # Log chi tiết ra console
    return f"Lỗi trích xuất đặc trưng âm thanh: {str(e)}"  # Thông báo ngắn gọn cho user
```

---

## 7. Các Đặc Trưng Âm Thanh Được Trích Xuất

### 7.1 Danh Sách Đầy Đủ 60 Đặc Trưng

| STT | Tên cột | Thư viện | Ý nghĩa |
|-----|---------|---------|---------|
| 1 | `length` | numpy | Số lượng sample của file âm thanh |
| 2 | `chroma_stft_mean` | librosa | Phân bố năng lượng trung bình theo 12 nốt nhạc (STFT) |
| 3 | `chroma_stft_var` | librosa | Độ biến thiên phân bố nốt nhạc |
| 4 | `rms_mean` | librosa | Biên độ âm thanh trung bình (cường độ) |
| 5 | `rms_var` | librosa | Độ biến thiên cường độ |
| 6 | `spectral_centroid_mean` | librosa | Tần số "trọng tâm" – nhạc bass vs treble |
| 7 | `spectral_centroid_var` | librosa | Độ biến thiên trọng tâm phổ |
| 8 | `spectral_bandwidth_mean` | librosa | Độ rộng dải tần số |
| 9 | `spectral_bandwidth_var` | librosa | Độ biến thiên độ rộng phổ |
| 10 | `rolloff_mean` | librosa | Điểm tần số chứa 85% năng lượng |
| 11 | `rolloff_var` | librosa | Độ biến thiên rolloff |
| 12 | `zero_crossing_rate_mean` | librosa | Tỷ lệ tín hiệu đổi dấu (noise level) |
| 13 | `zero_crossing_rate_var` | librosa | Độ biến thiên ZCR |
| 14 | `harmony_mean` | librosa | Trung bình thành phần hòa âm (sau HPSS) |
| 15 | `harmony_var` | librosa | Độ biến thiên hòa âm |
| 16 | `perceptr_mean` | librosa | Trung bình thành phần nhịp/tiết tấu (sau HPSS) |
| 17 | `perceptr_var` | librosa | Độ biến thiên tiết tấu |
| 18 | `tempo` | librosa | Ước tính nhịp điệu BPM |
| 19–38 | `mfcc1_mean` → `mfcc10_var` | librosa | 10 MFCC đầu (mean + var) – màu âm sắc |
| 39–58 | `mfcc11_mean` → `mfcc20_var` | librosa | 10 MFCC sau (mean + var) – chi tiết âm sắc |
| 59 | `chroma_cqt_mean` | librosa | Phân bố nốt nhạc theo CQT (chính xác hơn) |
| 60 | `chroma_cqt_var` | librosa | Độ biến thiên Chroma CQT |

### 7.2 Tại Sao MFCC Quan Trọng?

MFCC (Mel-Frequency Cepstral Coefficients) mô phỏng cách tai người cảm nhận âm thanh. 20 MFCC × 2 (mean + var) = 40 features, chiếm 67% tổng số đặc trưng. Đây là nhóm đặc trưng quan trọng nhất cho phân loại thể loại nhạc.

---

## 8. Cơ Sở Dữ Liệu MySQL

### 8.1 Kết Nối

```python
# Trong feature_extraction.py
from sqlalchemy import create_engine
from urllib.parse import quote_plus

username = "root"
password = quote_plus("22092005@Dung")  # URL-encode ký tự đặc biệt
host = "127.0.0.1"
port = "3306"
database_name = "music_db"

engine = create_engine(
    f"mysql+pymysql://{username}:{password}@{host}:{port}/{database_name}"
)
```

### 8.2 Ghi Dữ Liệu

```python
df_features_sql.to_sql(
    name='songs_features',
    con=engine,
    if_exists='replace',   # Xóa bảng cũ và tạo lại
    index=False
)
```

> ⚠️ `if_exists='replace'` sẽ **xóa toàn bộ dữ liệu cũ** mỗi lần chạy. Đổi thành `'append'` nếu muốn cộng dồn.

### 8.3 Restore từ Dump

```bash
# Restore database từ file Dump20260528.sql
mysql -u root -p music_db < Dump20260528.sql
```

---

## 9. Mô Hình Machine Learning

### 9.1 Thuật Toán SVM

**Support Vector Machine (SVM)** tìm hyperplane tối ưu để phân chia các thể loại nhạc trong không gian 60 chiều đặc trưng. Đặc biệt hiệu quả với dữ liệu tầm trung (1000–10000 mẫu) và đặc trưng số.

### 9.2 Pipeline Huấn Luyện (trong `ml_classification.ipynb`)

```python
# 1. Load data
df = pd.read_csv('example_features_with_path.csv')  # hoặc _3_sec.csv

# 2. Tách features và labels
X = df.drop(['filename', 'genre_label', 'file_path'], axis=1)
y = df['genre_label']

# 3. Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# 5. Chuẩn hóa
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train SVM
from sklearn.svm import SVC
model = SVC(kernel='rbf', C=10, gamma='scale')
model.fit(X_train_scaled, y_train)

# 7. Lưu model
import joblib
joblib.dump(model, 'music_svm_model.pkl')
joblib.dump(scaler, 'music_scaler.pkl')
joblib.dump(list(le.classes_), 'music_categories.pkl')
```

### 9.3 File `.pkl`

| File | Nội dung | Dùng trong |
|------|---------|-----------|
| `music_svm_model.pkl` | Object `SVC` đã fit | `app.py` – `model.predict()` |
| `music_scaler.pkl` | Object `StandardScaler` đã fit | `app.py` – `scaler.transform()` |
| `music_categories.pkl` | `['blues', 'classical', ..., 'rock']` | `app.py` – map index → tên |

---

## 10. Giao Diện Web Gradio

### 10.1 Khởi Động

```bash
python app.py
# → http://127.0.0.1:7860
```

### 10.2 Luồng Sử Dụng

```
1. Upload file nhạc (WAV, MP3, FLAC, ...)
          ↓
2. Gradio lưu file vào thư mục temp → trả filepath cho app
          ↓
3. app.py gọi predict_genre_of_file(filepath)
          ↓
4. Trích xuất 60 đặc trưng (mất 3–8 giây)
          ↓
5. Chuẩn hóa → SVM predict → map category
          ↓
6. Hiển thị: "🎵 Dự đoán thể loại: JAZZ"
```

### 10.3 Tùy Chỉnh

**Đổi theme giao diện:**
```python
# Các theme có sẵn trong Gradio
gr.Blocks(theme=gr.themes.Soft())    # Hiện tại
gr.Blocks(theme=gr.themes.Default()) # Mặc định
gr.Blocks(theme=gr.themes.Monochrome()) # Đen trắng
gr.Blocks(theme=gr.themes.Glass())   # Trong suốt
```

**Cho phép share link công khai:**
```python
demo.launch(share=True)
# Sẽ in ra: Running on public URL: https://xxxx.gradio.live
```

---

## 11. Xử Lý Lỗi Thường Gặp

### Lỗi 1: Không load được model

```
LỖI KHÔNG NẠP ĐƯỢC AI: [Errno 2] No such file or directory: 'music_svm_model.pkl'
```

**Nguyên nhân:** Chưa chạy `ml_classification.ipynb`  
**Giải pháp:** Chạy toàn bộ notebook để tạo file `.pkl`

---

### Lỗi 2: MySQL connection failed

```
❌ Lỗi MySQL: (OperationalError) Can't connect to MySQL server on '127.0.0.1'
```

**Nguyên nhân:** MySQL chưa khởi động hoặc sai thông tin đăng nhập  
**Giải pháp:** Kiểm tra MySQL service, hoặc bỏ qua (feature extraction vẫn lưu CSV thành công)

---

### Lỗi 3: Không tìm thấy dataset

```
❌ Không tìm thấy dataset!
```

**Nguyên nhân:** Chưa đặt dataset vào thư mục `genres_original/`  
**Giải pháp:** Tải GTZAN dataset và giải nén đúng cấu trúc thư mục

---

### Lỗi 4: Lỗi âm thanh khi predict

```
Lỗi trích xuất đặc trưng âm thanh: Audio is too short
```

**Nguyên nhân:** File nhạc ngắn hơn 30 giây  
**Giải pháp:** Sử dụng file nhạc có thời lượng ≥ 30 giây

---

### Lỗi 5: Sai thứ tự cột features

```
ValueError: X has 58 features, but StandardScaler is expecting 60 features
```

**Nguyên nhân:** Model được train với tập đặc trưng khác số cột hiện tại  
**Giải pháp:** Train lại model và lưu `.pkl` mới từ đúng pipeline trích xuất

---

### Lỗi 6: Lỗi encoding trên Windows

```
UnicodeEncodeError: 'charmap' codec can't encode character
```

**Nguyên nhân:** Terminal Windows không hỗ trợ UTF-8 mặc định  
**Giải pháp:** Đã được xử lý trong code với dòng:
```python
sys.stdout.reconfigure(encoding='utf-8')
```

---

## 12. Thư Viện và Dependencies

| Thư viện | Phiên bản | Vai trò |
|---------|---------|---------|
| `numpy` | 2.4.6 | Tính mean, var, xử lý mảng |
| `pandas` | 3.0.3 | DataFrame, đọc/ghi CSV |
| `scipy` | latest | Hỗ trợ librosa, sklearn |
| `matplotlib` | 3.10.0 | Vẽ biểu đồ trong notebook |
| `seaborn` | 0.13.2 | Confusion Matrix, Heatmap |
| `scikit-learn` | 1.6.0 | SVM, StandardScaler, metrics |
| `xgboost` | 2.1.3 | Thử nghiệm XGBoost trong notebook |
| `librosa` | 0.11.0 | Phân tích âm thanh, extract features |
| `gradio` | 6.15.2 | Giao diện web |
| `joblib` | 1.5.3 | Lưu/load file .pkl |
| `jupyterlab` | 4.3.0 | Chạy file .ipynb |
| `sqlalchemy` | (auto) | Kết nối MySQL từ pandas |
| `pymysql` | (auto) | MySQL driver cho SQLAlchemy |

---

## Ghi Chú Bảo Mật

> ⚠️ **Quan trọng cho môi trường Production:**
> 
> File `feature_extraction.py` hiện đang hard-code thông tin đăng nhập MySQL:
> ```python
> username = "root"
> password = quote_plus("22092005@Dung")
> ```
> 
> Khuyến nghị chuyển sang dùng biến môi trường:
> ```python
> import os
> username = os.environ.get("DB_USER", "root")
> password = quote_plus(os.environ.get("DB_PASS", ""))
> ```

---

*Wiki này được tạo dựa trên phân tích mã nguồn thực tế của project LapTrinhAmThanh. Cập nhật khi có thay đổi code.*
