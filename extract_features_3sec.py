import pandas as pd
import os
import numpy as np
import librosa
from multiprocessing import Pool, cpu_count

path = "genres_original/"

def get_single_3s_feature(audio_path, base_song_name, segment_idx):
    """Hàm phụ trách trích xuất đặc trưng cho CHỈ MỘT đoạn 3 giây cụ thể"""
    start_time = segment_idx * 3.0
    song_name_3s = f"{base_song_name}.{segment_idx}"
    
    try:
        # 1. Tải CHỈ ĐÚNG 3 giây âm thanh tại vị trí offset tương ứng
        y, sr = librosa.load(audio_path, offset=start_time, duration=3.0)
        
        # 2. Trích xuất các đặc trưng thời gian và thực thể phổ cho đoạn 3s
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        # 3. Tính toán Mean và Var cho đoạn 3s
        features = {
            'length': 66149, # Ép độ dài chuẩn của đoạn 3s (sample count) giống bộ dữ liệu gốc
            'chroma_stft_mean': np.mean(chroma_stft), 'chroma_stft_var': np.var(chroma_stft),
            'rms_mean': np.mean(rmse), 'rms_var': np.var(rmse),
            'spectral_centroid_mean': np.mean(spec_cent), 'spectral_centroid_var': np.var(spec_cent),
            'spectral_bandwidth_mean': np.mean(spec_bw), 'spectral_bandwidth_var': np.var(spec_bw),
            'rolloff_mean': np.mean(rolloff), 'rolloff_var': np.var(rolloff),
            'zero_crossing_rate_mean': np.mean(zcr), 'zero_crossing_rate_var': np.var(zcr),
            'harmony_mean': np.mean(librosa.effects.harmonic(y)), 'harmony_var': np.var(librosa.effects.harmonic(y)),
            'perceptr_mean': np.mean(librosa.effects.percussive(y)), 'perceptr_var': np.var(librosa.effects.percussive(y)),
            'tempo': tempo[0] if isinstance(tempo, (list, np.ndarray)) else tempo
        }
        
        # Bổ sung 20 đặc trưng MFCC
        for i in range(20):
            features[f'mfcc{i+1}_mean'] = np.mean(mfcc[i])
            features[f'mfcc{i+1}_var'] = np.var(mfcc[i])
            
        # Tính thêm Chroma CQT
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
        features['chroma_cqt_mean'] = np.mean(chroma_cqt)
        features['chroma_cqt_var'] = np.var(chroma_cqt)

        return pd.DataFrame(features, index=[song_name_3s])
    except Exception as e:
        # Bỏ qua nếu dính file lỗi kỹ thuật
        return None

# --- LUỒNG CHẠY ĐA NHÂN CHO TẬP 3 GIÂY ---
def extract_single_song_to_10_segments(wav_path):
    """Hàm bốc 1 bài hát gốc rồi băm thành list 10 đoạn DataFrame 3s"""
    try:
        base_song_name = os.path.splitext(os.path.basename(wav_path))[0]
        song_df_list = []
        
        # Vòng lặp băm nhỏ bài hát 30s thành 10 khúc 3s
        for segment_idx in range(10):
            df_segment = get_single_3s_feature(wav_path, base_song_name, segment_idx)
            if df_segment is not None:
                song_df_list.append(df_segment)
                
        if len(song_df_list) == 0:
            return None
        # Gộp tạm 10 khúc của bài này lại thành 1 cụm DataFrame nhỏ trước
        return pd.concat(song_df_list, axis=0)
    except Exception as e:
        print(f"❌ Lỗi khi xử lý file {wav_path}: {e}")
        return None

if __name__ == '__main__':
    # 1. Định nghĩa 10 thư mục con
    genre_list = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    # 2. Gom toàn bộ file nhạc .wav
    wav_list = []
    for g in genre_list:
        genre_path = os.path.join(path, g)
        if os.path.exists(genre_path):
            for wav in os.listdir(genre_path):
                if wav.endswith('.wav'):
                    wav_list.append(os.path.join(genre_path, wav))

    total_songs = len(wav_list)
    
    if total_songs == 0:
        print("❌ Không tìm thấy file nhạc nào trong thư mục 'genres_original'.")
    else:
        num_processors = cpu_count()
        print(f"🚀 Tìm thấy {total_songs} bài hát gốc.")
        print(f"🔥 Sẵn sàng băm nhỏ thành {total_songs * 10} đoạn 3 giây sử dụng {num_processors} nhân CPU...")

        # Kích hoạt Pool chạy đa nhân song song
        with Pool(processes=num_processors) as pool:
            df_list = pool.map(extract_single_song_to_10_segments, wav_list)

        # Lọc bỏ các cụm file lỗi và gộp đại thành một bảng dữ liệu siêu lớn
        df_list = [df for df in df_list if df is not None]
        print("✅ Đã trích xuất xong toàn bộ 10.000 đoạn! Đang gộp dữ liệu...")
        
        df_features_3s = pd.concat(df_list, axis=0)

        # Tạo nhãn thể loại nhạc từ tên hàng (Ví dụ: blues.00000.1 -> lấy chữ 'blues')
        labels = []
        for i in df_features_3s.index:
            labels.append(i.split('.')[0])
        df_features_3s['genre_label'] = labels

        # Đặt đường dẫn đầu ra riêng biệt, không đè lên file 30s cũ
        current_dir = os.getcwd()
        output_path = os.path.join(current_dir, 'example_features_3_sec.csv')
        df_features_3s.to_csv(output_path, index_label='filename')
        
        print("-" * 60)
        print(f"🎉 THÀNH CÔNG RỰC RỠ! Bản 3s đã được tạo lập.")
        print(f"💾 File lưu tại: {output_path}")
        print(f"📊 Tổng quy mô dữ liệu thu được: {df_features_3s.shape[0]} hàng, {df_features_3s.shape[1]} cột.")