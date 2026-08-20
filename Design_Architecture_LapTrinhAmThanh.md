# 🎨 Design & Architecture Document
# Hệ Thống AI Phân Biệt Thể Loại Nhạc – LapTrinhAmThanh

> **Phiên bản:** 1.0  
> **Ngày cập nhật:** 29/05/2026  
> **Trạng thái:** Released

---

## 1. Kiến Trúc Tổng Thể

```
┌─────────────────────────────────────────────────────────────────────┐
│                     HỆ THỐNG LAPTRINHÂMTHANH                        │
│                                                                     │
│  ┌─────────────────────┐      ┌──────────────────────────────────┐  │
│  │   PIPELINE OFFLINE  │      │       PIPELINE ONLINE (app.py)   │  │
│  │  (Huấn luyện mô hình│      │       (Dự đoán thời gian thực)  │  │
│  └─────────────────────┘      └──────────────────────────────────┘  │
│           │                                    │                    │
│           ▼                                    ▼                    │
│  ┌─────────────────┐                ┌─────────────────────┐        │
│  │ genres_original │                │   Gradio Web UI     │        │
│  │  (1000 .wav)    │                │   localhost:7860    │        │
│  └────────┬────────┘                └──────────┬──────────┘        │
│           │                                    │                    │
│           ▼                                    ▼                    │
│  ┌─────────────────┐                ┌─────────────────────┐        │
│  │feature_extraction│               │  predict_genre_of_  │        │
│  │      .py        │                │     file()          │        │
│  │extract_features_│                │  (60 features)      │        │
│  │  3sec.py        │                └──────────┬──────────┘        │
│  └────────┬────────┘                           │                    │
│           │                                    │                    │
│           ▼                                    ▼                    │
│  ┌────────────────────┐            ┌─────────────────────┐        │
│  │  example_features  │            │   music_scaler.pkl  │        │
│  │     _with_path.csv │            │   music_svm_model   │        │
│  │  example_features  │            │       .pkl          │        │
│  │   _3_sec.csv       │            │  music_categories   │        │
│  └────────┬───────────┘            │       .pkl          │        │
│           │                        └──────────┬──────────┘        │
│           ▼                                    │                    │
│  ┌─────────────────────┐                       │                    │
│  │ ml_classification   │                       │                    │
│  │     .ipynb          │──────────────────────►│                    │
│  │  (Train SVM Model)  │    xuất .pkl files    │                    │
│  └─────────────────────┘                       │                    │
│           │                                    ▼                    │
│           ▼                        ┌─────────────────────┐        │
│  ┌─────────────────────┐           │  🎵 Kết quả dự đoán │        │
│  │  MySQL: music_db    │           │  "BLUES / JAZZ / ..." │       │
│  │  songs_features     │           └─────────────────────┘        │
│  └─────────────────────┘                                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Thiết Kế Cơ Sở Dữ Liệu

### 2.1 Database: `music_db`

**Kết nối:**
```
Host    : 127.0.0.1
Port    : 3306
Database: music_db
User    : root
Engine  : MySQL 8.0+
```

### 2.2 Bảng `songs_features`

Bảng này được tạo tự động bởi `feature_extraction.py` thông qua `pandas.to_sql()` với `if_exists='replace'`.

| STT | Tên cột | Kiểu dữ liệu | Mô tả |
|-----|---------|-------------|-------|
| 1 | `filename` | VARCHAR(100) | Tên file gốc. VD: `blues.00000` |
| 2 | `length` | INT | Số lượng sample (≈ 661.794 với 30s@22050Hz) |
| 3 | `chroma_stft_mean` | FLOAT | Mean của Chroma STFT |
| 4 | `chroma_stft_var` | FLOAT | Variance của Chroma STFT |
| 5 | `rms_mean` | FLOAT | Mean của RMS Energy |
| 6 | `rms_var` | FLOAT | Variance của RMS Energy |
| 7 | `spectral_centroid_mean` | FLOAT | Mean của Spectral Centroid |
| 8 | `spectral_centroid_var` | FLOAT | Variance của Spectral Centroid |
| 9 | `spectral_bandwidth_mean` | FLOAT | Mean của Spectral Bandwidth |
| 10 | `spectral_bandwidth_var` | FLOAT | Variance của Spectral Bandwidth |
| 11 | `rolloff_mean` | FLOAT | Mean của Spectral Rolloff |
| 12 | `rolloff_var` | FLOAT | Variance của Spectral Rolloff |
| 13 | `zero_crossing_rate_mean` | FLOAT | Mean của Zero Crossing Rate |
| 14 | `zero_crossing_rate_var` | FLOAT | Variance của Zero Crossing Rate |
| 15 | `harmony_mean` | FLOAT | Mean của Harmonic component |
| 16 | `harmony_var` | FLOAT | Variance của Harmonic component |
| 17 | `perceptr_mean` | FLOAT | Mean của Percussive component |
| 18 | `perceptr_var` | FLOAT | Variance của Percussive component |
| 19 | `tempo` | FLOAT | Tempo ước tính (BPM) |
| 20–59 | `mfcc1_mean` → `mfcc20_var` | FLOAT | 20 MFCC × (mean + var) = 40 cột |
| 60 | `chroma_cqt_mean` | FLOAT | Mean của Chroma CQT |
| 61 | `chroma_cqt_var` | FLOAT | Variance của Chroma CQT |
| 62 | `genre_label` | VARCHAR(20) | Nhãn thể loại: `blues`, `rock`,... |
| 63 | `file_path` | VARCHAR(200) | Đường dẫn tương đối. VD: `genres_original/blues/blues.00000.wav` |

**Tổng: 63 cột** (60 features + filename + genre_label + file_path)

### 2.3 ERD (Entity Relationship)

```
┌──────────────────────────────────────────────────┐
│                  songs_features                  │
├──────────────────────────────────────────────────┤
│ PK  filename            VARCHAR(100)             │
│     length              INT                      │
│     chroma_stft_mean    FLOAT                    │
│     chroma_stft_var     FLOAT                    │
│     rms_mean            FLOAT                    │
│     rms_var             FLOAT                    │
│     spectral_centroid_mean  FLOAT                │
│     spectral_centroid_var   FLOAT                │
│     spectral_bandwidth_mean FLOAT                │
│     spectral_bandwidth_var  FLOAT                │
│     rolloff_mean        FLOAT                    │
│     rolloff_var         FLOAT                    │
│     zero_crossing_rate_mean FLOAT                │
│     zero_crossing_rate_var  FLOAT                │
│     harmony_mean        FLOAT                    │
│     harmony_var         FLOAT                    │
│     perceptr_mean       FLOAT                    │
│     perceptr_var        FLOAT                    │
│     tempo               FLOAT                    │
│     mfcc1_mean ... mfcc20_var  FLOAT (40 cols)   │
│     chroma_cqt_mean     FLOAT                    │
│     chroma_cqt_var      FLOAT                    │
│     genre_label         VARCHAR(20)              │
│     file_path           VARCHAR(200)             │
└──────────────────────────────────────────────────┘
```

> **Lưu ý:** Bảng không có khóa ngoại vì là bảng phẳng (flat table) dùng cho ML training. Không có quan hệ với bảng khác.

---

## 3. Thiết Kế Module

### 3.1 `feature_extraction.py`

```
feature_extraction.py
│
├── get_all_musical_features(audio_path, song_name, start, chroma_method_list)
│   ├── INPUT : đường dẫn file .wav, tên bài, offset, list method chroma
│   ├── PROCESS: load audio → extract 60 features → build dict → DataFrame
│   └── OUTPUT: pd.DataFrame 1 hàng × 60 cột, index = song_name
│
├── extract_single_song(wav_path)
│   ├── INPUT : đường dẫn file .wav
│   ├── PROCESS: gọi get_all_musical_features() → bắt exception
│   └── OUTPUT: DataFrame hoặc None nếu lỗi
│
└── __main__
    ├── Duyệt genres_original/ → gom danh sách .wav
    ├── Pool(processes=16).map(extract_single_song, wav_list)
    ├── Gộp DataFrame, gán genre_label, file_path
    ├── Lưu CSV: example_features_with_path.csv
    └── Lưu MySQL: songs_features (if_exists='replace')
```

### 3.2 `extract_features_3sec.py`

```
extract_features_3sec.py
│
├── get_single_3s_feature(audio_path, base_song_name, segment_idx)
│   ├── INPUT : đường dẫn, tên gốc, index đoạn (0-9)
│   ├── PROCESS: load 3s tại offset=idx*3 → extract → force length=66149
│   └── OUTPUT: DataFrame 1 hàng với index "{name}.{idx}"
│
├── extract_single_song_to_10_segments(wav_path)
│   ├── INPUT : đường dẫn file .wav
│   ├── PROCESS: vòng for 0→9, gọi get_single_3s_feature, concat
│   └── OUTPUT: DataFrame 10 hàng × 60 cột
│
└── __main__
    ├── Gom toàn bộ .wav từ genres_original/
    ├── Pool(processes=cpu_count()).map(extract_single_song_to_10_segments)
    ├── Gộp → ~10.000 hàng
    ├── Gán genre_label từ prefix filename
    └── Lưu: example_features_3_sec.csv
```

### 3.3 `app.py`

```
app.py
│
├── [Khởi động] Load 3 file .pkl
│   ├── music_scaler.pkl     → StandardScaler đã fit
│   ├── music_svm_model.pkl  → SVM model đã train
│   └── music_categories.pkl → list ['blues', 'classical', ..., 'rock']
│
├── predict_genre_of_file(song_path, model, scaler, categories)
│   ├── INPUT : đường dẫn file nhạc tạm thời từ Gradio
│   ├── PROCESS:
│   │   ├── librosa.load(duration=30.0)
│   │   ├── Extract 60 features (giống feature_extraction.py)
│   │   ├── pd.DataFrame → reindex theo scaler.feature_names_in_
│   │   ├── scaler.transform()
│   │   └── model.predict()[0] → map categories
│   └── OUTPUT: string tên thể loại viết hoa (VD: "BLUES")
│
├── gradio_predict(audio_file)
│   ├── INPUT : filepath từ Gradio component
│   ├── PROCESS: validate → gọi predict_genre_of_file → format output
│   └── OUTPUT: "🎵 Dự đoán thể loại: BLUES" hoặc thông báo lỗi
│
└── Gradio Blocks UI
    ├── Theme: gr.themes.Soft()
    ├── Title: "AI Phân Biệt Thể Loại Nhạc"
    ├── Layout: gr.Row() → 2 Column
    │   ├── Left:  gr.Audio(type="filepath") + Button
    │   └── Right: gr.Textbox (output)
    └── demo.launch(share=False)
```

---

## 4. Thiết Kế Giao Diện (UI Design)

### 4.1 Wireframe Giao Diện Web

```
┌─────────────────────────────────────────────────────────┐
│           🎵 HỆ THỐNG AI PHÂN BIỆT THỂ LOẠI NHẠC        │
│   (ỨNG DỤNG ĐỘC LẬP)                                   │
│                                                         │
│   Ứng dụng sử dụng SVM + Librosa để phân tích           │
│   dải tần số, nhịp điệu và MFCC để nhận diện...        │
├─────────────────────────┬───────────────────────────────┤
│                         │                               │
│  📁 Tải file nhạc lên  │  📊 Kết quả từ AI             │
│  ┌───────────────────┐  │  ┌─────────────────────────┐  │
│  │                   │  │  │                         │  │
│  │  Kéo thả file     │  │  │  🎵 Dự đoán thể loại:  │  │
│  │  nhạc vào đây     │  │  │       BLUES             │  │
│  │  hoặc click chọn  │  │  │                         │  │
│  │                   │  │  └─────────────────────────┘  │
│  └───────────────────┘  │                               │
│                         │                               │
│  [ 🔮 Bắt đầu phân     │                               │
│    tích & Dự đoán ]    │                               │
│                         │                               │
└─────────────────────────┴───────────────────────────────┘
```

### 4.2 Component Map

| Component | Gradio Class | Tham số quan trọng |
|-----------|-------------|-------------------|
| Header | `gr.Markdown` | Markdown text với HTML |
| Container | `gr.Blocks(theme=gr.themes.Soft())` | Theme Soft |
| Layout | `gr.Row()` | 2 cột ngang |
| File Input | `gr.Audio` | `type="filepath"`, label |
| Predict Button | `gr.Button` | `variant="primary"` |
| Output | `gr.Textbox` | `text_align="center"` |

### 4.3 Trạng Thái UI

| Trạng thái | Output Hiển Thị |
|-----------|----------------|
| Chưa upload file | `"Vui lòng kéo thả hoặc chọn một file nhạc trước khi bấm dự đoán!"` |
| Đang xử lý | Loading spinner (Gradio tự động) |
| Thành công | `"🎵 Dự đoán thể loại: {GENRE}"` |
| Lỗi xử lý | `"Lỗi trích xuất đặc trưng âm thanh: {error_message}"` |

---

## 5. Thiết Kế Đặc Trưng Âm Thanh (Feature Engineering Design)

### 5.1 Bảng Đặc Trưng

| Nhóm | Đặc trưng | Số cột | Ý nghĩa âm nhạc |
|------|-----------|--------|----------------|
| **Temporal** | `length` | 1 | Độ dài file (samples) |
| **Rhythm** | `tempo` | 1 | Nhịp điệu (BPM) – phân biệt nhạc nhanh/chậm |
| **Harmonic** | `chroma_stft_mean/var` | 2 | Phân bố năng lượng theo 12 nốt nhạc (C,D,E,...) |
| **Harmonic** | `chroma_cqt_mean/var` | 2 | Chroma theo CQT – chính xác hơn với nhạc cụ thực |
| **Energy** | `rms_mean/var` | 2 | Biên độ âm thanh tổng thể |
| **Spectral** | `spectral_centroid_mean/var` | 2 | "Độ sáng" của âm thanh – treble vs bass |
| **Spectral** | `spectral_bandwidth_mean/var` | 2 | Độ rộng dải tần – nhạc phức tạp vs đơn giản |
| **Spectral** | `rolloff_mean/var` | 2 | Điểm 85% năng lượng phổ |
| **Percussive** | `zero_crossing_rate_mean/var` | 2 | Tỷ lệ noise – nhạc rock/metal vs classical |
| **Separation** | `harmony_mean/var` | 2 | Thành phần giai điệu thuần |
| **Separation** | `perceptr_mean/var` | 2 | Thành phần nhịp trống/bass |
| **Timbre** | `mfcc1_mean` → `mfcc20_var` | 40 | Màu sắc âm sắc – quan trọng nhất trong phân loại |
| **Tổng** | | **60** | |

### 5.2 Pipeline Chuẩn Hóa

```
Raw Features (60 cols)
        │
        ▼
StandardScaler.fit_transform()  ← fit trên training data
        │                          saved vào music_scaler.pkl
        ▼
Normalized Features (mean=0, std=1)
        │
        ▼
SVM.fit()  ← train
        │     saved vào music_svm_model.pkl
        ▼
Predict time:
        │
Raw Features → reindex → StandardScaler.transform() → SVM.predict()
```

---

## 6. Cấu Trúc File Dự Án

```
LapTrinhAmThanh/
│
├── genres_original/          # Dataset train (GTZAN)
│   ├── blues/                # 100 file blues.00000.wav → blues.00099.wav
│   ├── classical/
│   ├── country/
│   ├── disco/
│   ├── hiphop/
│   ├── jazz/
│   ├── metal/
│   ├── pop/
│   ├── reggae/
│   └── rock/
│
├── music_test/               # Dataset test độc lập
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
├── feature_extraction.py     # Script trích xuất 60 features từ file 30s
├── extract_features_3sec.py  # Script trích xuất 60 features từ đoạn 3s
├── ml_classification.ipynb   # Notebook huấn luyện SVM, đánh giá mô hình
├── app.py                    # Web app Gradio – giao diện dự đoán
│
├── example_features.csv           # Features mẫu (format đơn giản)
├── example_features_3_sec.csv     # Features 10.000 đoạn 3s
├── example_features_with_path.csv # Features 1.000 bài 30s + file_path
│
├── music_svm_model.pkl       # Mô hình SVM đã train
├── music_scaler.pkl          # StandardScaler đã fit
├── music_categories.pkl      # List thể loại: ['blues', ..., 'rock']
│
├── Bao_cao_Testcase_Phan_Cap.csv  # Báo cáo test case phân cấp
├── Dump20260528.sql               # MySQL dump của music_db
├── requirements.txt               # Danh sách thư viện Python
└── .venv/                         # Virtual environment Python
```

---

## 7. Sơ Đồ Luồng Dữ Liệu (Data Flow)

```
[File .wav]
    │
    │  librosa.load(duration=30.0 hoặc 3.0)
    ▼
[Audio Signal: y, sr]
    │
    ├──► beat_track()        → tempo
    ├──► chroma_stft()       → chroma_stft [12 × T]
    ├──► rms()               → rmse [1 × T]
    ├──► spectral_centroid() → spec_cent [1 × T]
    ├──► spectral_bandwidth()→ spec_bw [1 × T]
    ├──► spectral_rolloff()  → rolloff [1 × T]
    ├──► zero_crossing_rate()→ zcr [1 × T]
    ├──► mfcc(n_mfcc=20)     → mfcc [20 × T]
    ├──► harmonic()          → harmony [N]
    ├──► percussive()        → percussive [N]
    └──► chroma_cqt()        → chroma_cqt [12 × T]
             │
             │  np.mean() + np.var() cho từng feature
             ▼
    [Feature Vector: 60 giá trị số]
             │
             │  pd.DataFrame → StandardScaler.transform()
             ▼
    [Normalized Vector: 60 giá trị chuẩn hóa]
             │
             │  SVM.predict()
             ▼
    [Predicted Class Index: 0-9]
             │
             │  categories_list[index].upper()
             ▼
    ["BLUES" / "CLASSICAL" / ... / "ROCK"]
```

---

*Tài liệu Design & Architecture này được tạo dựa trên phân tích mã nguồn thực tế của project LapTrinhAmThanh.*
