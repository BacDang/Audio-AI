import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Thư mục gốc chứa model files .pkl
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'music_svm_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'music_scaler.pkl')
CATEGORIES_PATH = os.path.join(BASE_DIR, 'music_categories.pkl')

# Dữ liệu kiến thức chi tiết về 10 thể loại nhạc
GENRE_METADATA: Dict[str, Dict[str, Any]] = {
    "blues": {
        "name_vi": "Nhạc Blues",
        "description": "Thể loại âm nhạc khởi nguồn từ cộng đồng người Mỹ gốc Phi, nổi bật với cấu trúc 12-bar blues, kỹ thuật blue notes (nốt trầm u uất), tiếng guitar slide và harmonica đầy cảm xúc.",
        "key_features": "Tempo chậm đến trung bình, hòa âm mượt mà (high harmonic ratio), năng lượng trầm ấm.",
        "icon": "🎸",
        "color": "#3b82f6",
        "gradient": "linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%)",
        "sample_artists": "B.B. King, Muddy Waters, Robert Johnson, Stevie Ray Vaughan"
    },
    "classical": {
        "name_vi": "Nhạc Cổ Điển",
        "description": "Âm nhạc nghệ thuật hàn lâm phương Tây, đặc trưng bởi phối khí giao hưởng phong phú, dải động lực (dynamic range) cực rộng từ êm dịu đến bùng nổ, không có trống gõ hiện đại.",
        "key_features": "Tỷ lệ Harmonic áp đảo Percussive, năng lượng RMS êm dịu, dải tần số biến chuyển tinh tế.",
        "icon": "🎻",
        "color": "#8b5cf6",
        "gradient": "linear-gradient(135deg, #4c1d95 0%, #8b5cf6 100%)",
        "sample_artists": "Beethoven, Mozart, Bach, Chopin, Tchaikovsky"
    },
    "country": {
        "name_vi": "Nhạc Đồng Quê (Country)",
        "description": "Âm nhạc truyền thống nông thôn Mỹ, tập trung vào ca từ tự sự mộc mạc, kết hợp nhạc cụ dây acoustic như Guitar, Banjo, Fiddle và Pedal Steel Guitar.",
        "key_features": "Tempo đều đặn, âm sắc mộc (acoustic acoustic brightness), độ trong trẻo cao.",
        "icon": "🤠",
        "color": "#f59e0b",
        "gradient": "linear-gradient(135deg, #78350f 0%, #f59e0b 100%)",
        "sample_artists": "Johnny Cash, Dolly Parton, Willie Nelson, Taylor Swift (Early)"
    },
    "disco": {
        "name_vi": "Nhạc Disco",
        "description": "Dòng nhạc khiêu vũ thịnh hành thập niên 1970, đặc trưng bởi nhịp gõ 'four-on-the-floor' dồn dập (120 BPM), tiếng bassline nảy (syncopated bass) và đàn dây sôi động.",
        "key_features": "Tempo rất ổn định quanh 115-130 BPM, thành phần gõ (percussive) mạnh mẽ, Spectral Centroid cao.",
        "icon": "🪩",
        "color": "#ec4899",
        "gradient": "linear-gradient(135deg, #831843 0%, #ec4899 100%)",
        "sample_artists": "Bee Gees, ABBA, Donna Summer, Earth Wind & Fire"
    },
    "hiphop": {
        "name_vi": "Nhạc Hiphop / Rap",
        "description": "Văn hóa âm nhạc đô thị với nhịp beat 808 nặng, sampling, vòng lặp giai điệu (loops) và kỹ thuật đọc rap gieo vần nhịp nhàng trên nền trống nảy lửa.",
        "key_features": "Năng lượng âm trầm (sub-bass) rất cao, Zero Crossing Rate thấp ở phần nền, nhịp boom-bap / trap rõ ràng.",
        "icon": "🎤",
        "color": "#10b981",
        "gradient": "linear-gradient(135deg, #064e3b 0%, #10b981 100%)",
        "sample_artists": "Eminem, Tupac, Kendrick Lamar, Drake, Jay-Z"
    },
    "jazz": {
        "name_vi": "Nhạc Jazz",
        "description": "Nghệ thuật ứng biến tự do đỉnh cao, đặc trưng bởi hợp âm phức tạp 7th/9th/11th, nhịp điệu swing lả lướt, tiếng kèn Saxophone, Trumpet và Piano điêu luyện.",
        "key_features": "Phổ tần số biến thiên phức tạp, tỷ lệ hài hòa cao, nhịp điệu đảo phách (syncopation).",
        "icon": "🎷",
        "color": "#06b6d4",
        "gradient": "linear-gradient(135deg, #164e63 0%, #06b6d4 100%)",
        "sample_artists": "Miles Davis, John Coltrane, Louis Armstrong, Duke Ellington"
    },
    "metal": {
        "name_vi": "Nhạc Heavy Metal",
        "description": "Dòng nhạc mạnh bạo với tiếng guitar điện méo tiếng cao độ (heavy distortion), tiếng trống double-bass dồn dập tốc độ cao và giọng hát gào thét nội lực.",
        "key_features": "Năng lượng RMS cực cao, Spectral Centroid và Rolloff cực đại, độ ồn và tỷ lệ Percussive dày đặc.",
        "icon": "⚡",
        "color": "#ef4444",
        "gradient": "linear-gradient(135deg, #7f1d1d 0%, #ef4444 100%)",
        "sample_artists": "Metallica, Iron Maiden, Black Sabbath, Megadeth, Slayer"
    },
    "pop": {
        "name_vi": "Nhạc Pop",
        "description": "Âm nhạc đại chúng hiện đại với cấu trúc Verse-Chorus bắt tai, giai điệu dễ nhớ (hooks), sản xuất phòng thu trau chuốt và giọng hát làm trung tâm.",
        "key_features": "Độ cân bằng hài hòa giữa Harmonic & Percussive, Tempo phổ biến 100-128 BPM, năng lượng vừa phải.",
        "icon": "✨",
        "color": "#a855f7",
        "gradient": "linear-gradient(135deg, #581c87 0%, #a855f7 100%)",
        "sample_artists": "Michael Jackson, Madonna, Bruno Mars, Dua Lipa, Ariana Grande"
    },
    "reggae": {
        "name_vi": "Nhạc Reggae",
        "description": "Âm nhạc truyền thống Jamaica với nhịp đảo phách 'skank' đặc trưng ở phách 2 và 4, tiếng bass sâu lắng và thông điệp hòa bình, thư thái.",
        "key_features": "Nhịp điệu nhát chém guitar ở nhịp offbeat, tempo vừa phải (70-90 BPM), dải trầm sâu.",
        "icon": "🌴",
        "color": "#84cc16",
        "gradient": "linear-gradient(135deg, #365314 0%, #84cc16 100%)",
        "sample_artists": "Bob Marley, Peter Tosh, Jimmy Cliff, UB40"
    },
    "rock": {
        "name_vi": "Nhạc Rock",
        "description": "Thể loại nhạc guitar điện mạnh mẽ phát triển từ Rock & Roll, kết hợp dàn trống chắc nịch, bass dầy và tinh thần nổi loạn, phóng khoáng.",
        "key_features": "Năng lượng âm thanh cao, dải tần trung bùng nổ, sự kết hợp cân bằng giữa nhịp điệu và hòa âm guitar.",
        "icon": "🥁",
        "color": "#f97316",
        "gradient": "linear-gradient(135deg, #7c2d12 0%, #f97316 100%)",
        "sample_artists": "Queen, Led Zeppelin, The Beatles, AC/DC, Nirvana"
    }
}

class MLEngine:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.categories: List[str] = []
        self.is_loaded = False
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(CATEGORIES_PATH)):
                print(f"[MLEngine] Cảnh báo: Không tìm thấy đủ các file .pkl tại {BASE_DIR}")
                return
            
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            raw_cats = joblib.load(CATEGORIES_PATH)
            self.categories = [str(c).replace('\\', '/').split('/')[-1].strip().lower() for c in raw_cats]
            self.is_loaded = True
            print(f"[MLEngine] Nạp AI thành công! {len(self.categories)} thể loại: {self.categories}")
        except Exception as e:
            print(f"[MLEngine] Lỗi nạp mô hình AI: {e}")
            self.is_loaded = False

    def predict(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dự đoán thể loại từ từ điển 60 đặc trưng âm thanh
        Trả về thể loại dự đoán, điểm tin cậy (confidence), và phân bố xác suất cho tất cả thể loại.
        """
        if not self.is_loaded:
            raise RuntimeError("Mô hình AI chưa được nạp thành công.")

        # Chuẩn bị DataFrame đúng thứ tự cột
        df_song = pd.DataFrame([features_dict])
        if hasattr(self.scaler, 'feature_names_in_'):
            df_song = df_song.reindex(columns=self.scaler.feature_names_in_, fill_value=0.0)

        # Chuẩn hóa đặc trưng
        df_song_scaled = self.scaler.transform(df_song)

        # Tính xác suất hoặc decision function
        probabilities: Dict[str, float] = {}
        if hasattr(self.model, 'predict_proba'):
            try:
                probs = self.model.predict_proba(df_song_scaled)[0]
                for idx, cat in enumerate(self.categories):
                    probabilities[cat] = float(probs[idx])
            except Exception:
                probs = None

        # Nếu không có predict_proba, tính từ decision_function qua softmax
        if not probabilities and hasattr(self.model, 'decision_function'):
            dec_values = self.model.decision_function(df_song_scaled)[0]
            exp_vals = np.exp(dec_values - np.max(dec_values))
            softmax_probs = exp_vals / np.sum(exp_vals)
            for idx, cat in enumerate(self.categories):
                probabilities[cat] = float(softmax_probs[idx])

        # Dự đoán nhãn
        pred_code = int(self.model.predict(df_song_scaled)[0])
        predicted_genre = self.categories[pred_code].lower()
        confidence = probabilities.get(predicted_genre, 1.0)

        # Sắp xếp phân bố xác suất từ cao xuống thấp
        sorted_probs = sorted(
            [{"genre": k, "probability": round(v * 100, 2), "meta": GENRE_METADATA.get(k, {})} for k, v in probabilities.items()],
            key=lambda x: x["probability"],
            reverse=True
        )

        return {
            "predicted_genre": predicted_genre,
            "predicted_genre_vi": GENRE_METADATA.get(predicted_genre, {}).get("name_vi", predicted_genre.upper()),
            "confidence_percentage": round(confidence * 100, 2),
            "confidence_level": "Cao" if confidence >= 0.7 else ("Trung bình" if confidence >= 0.4 else "Thấp"),
            "probabilities": sorted_probs,
            "genre_meta": GENRE_METADATA.get(predicted_genre, {}),
            "total_classes": len(self.categories)
        }

# Khởi tạo singleton instance
ml_engine = MLEngine()
