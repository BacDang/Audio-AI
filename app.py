import os
import librosa
import numpy as np
import pandas as pd
import gradio as gr
import traceback
import joblib
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 🔌 BƯỚC 1: TỰ ĐỘNG NẠP MÔ HÌNH AI ĐÃ LƯU (KHÔNG CẦN TRAIN LẠI)
print("Đang nạp cấu hình AI từ file .pkl... Vui lòng đợi...")

try:
    scaler = joblib.load('music_scaler.pkl')
    model_svm = joblib.load('music_svm_model.pkl')
    categories_list = joblib.load('music_categories.pkl')
    print("Nạp AI thành công! Hệ thống sẵn sàng hoạt động.")
except Exception as e:
    print(f"LỖI KHÔNG NẠP ĐƯỢC AI: {e}")
    print("Vui lòng chạy ô code lưu file .pkl bên Jupyter Notebook trước!")
    exit()

#HÀM TRÍCH XUẤT ĐẶC TRƯNG VÀ DỰ ĐOÁN FILE PHÁT SINH
def predict_genre_of_file(song_path, model, scaler, categories):
    # 1. Tải file âm thanh trọn vẹn 30 giây
    y, sr = librosa.load(song_path, duration=30.0) 
    
    # 2. Trích xuất toàn bộ các đặc trưng nâng cao bằng thư viện Librosa
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    
    harmony_wave = librosa.effects.harmonic(y)
    percussive_wave = librosa.effects.percussive(y)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    features = {
        'length': len(y),
        'chroma_stft_mean': np.mean(chroma_stft), 'chroma_stft_var': np.var(chroma_stft),
        'rms_mean': np.mean(rmse), 'rms_var': np.var(rmse),
        'spectral_centroid_mean': np.mean(spec_cent), 'spectral_centroid_var': np.var(spec_cent),
        'spectral_bandwidth_mean': np.mean(spec_bw), 'spectral_bandwidth_var': np.var(spec_bw),
        'rolloff_mean': np.mean(rolloff), 'rolloff_var': np.var(rolloff),
        'zero_crossing_rate_mean': np.mean(zcr), 'zero_crossing_rate_var': np.var(zcr),
        'harmony_mean': np.mean(harmony_wave), 'harmony_var': np.var(harmony_wave),
        'perceptr_mean': np.mean(percussive_wave), 'perceptr_var': np.var(percussive_wave),
        'tempo': tempo[0] if isinstance(tempo, (list, np.ndarray)) else tempo,
        'chroma_cqt_mean': np.mean(chroma_cqt), 'chroma_cqt_var': np.var(chroma_cqt)
    }
    for i in range(20):
        features[f'mfcc{i+1}_mean'] = np.mean(mfcc[i])
        features[f'mfcc{i+1}_var'] = np.var(mfcc[i])
        
    # 3. Ép DataFrame bài hát mới nhận ĐÚNG THỨ TỰ CỘT mà bộ chuẩn hóa yêu cầu
    df_song = pd.DataFrame([features])
    if hasattr(scaler, 'feature_names_in_'):
        df_song = df_song.reindex(columns=scaler.feature_names_in_)
    
    # Chuẩn hóa dữ liệu và dự đoán
    df_song_scaled = scaler.transform(df_song)
    pred_code = model.predict(df_song_scaled)[0]
    return categories[pred_code].upper()

# 🚀 BƯỚC 3: HÀM ĐIỀU PHỐI GIAO DIỆN GRADIO
def gradio_predict(audio_file):
    if audio_file is None:
        return "Vui lòng kéo thả hoặc chọn một file nhạc trước khi bấm dự đoán!"
    
    try:
        print(f"Đang xử lý file nhạc phát sinh: {audio_file}")
        prediction = predict_genre_of_file(
            song_path=audio_file,
            model=model_svm,
            scaler=scaler,
            categories=categories_list
        )
        return f"🎵 Dự đoán thể loại: {prediction}"
        
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"Chi tiết lỗi hệ thống:\n{error_msg}")
        return f"Lỗi trích xuất đặc trưng âm thanh: {str(e)}"

# 🎨 BƯỚC 4: THIẾT KẾ VÀ KHỞI CHẠY GIAO DIỆN WEB
with gr.Blocks(theme=gr.themes.Soft(), title="AI Phân Biệt Thể Loại Nhạc") as demo:
    
    gr.Markdown(
        """
        # 🎵 HỆ THỐNG AI PHÂN BIỆT THỂ LOẠI NHẠC (ỨNG DỤNG ĐỘC LẬP)
        Ứng dụng sử dụng mô hình học máy **SVM** kết hợp với thư viện xử lý âm thanh **Librosa** để phân tích dải tần số âm thanh, nhịp điệu (Tempo) và MFCC nhằm nhận diện thể loại nhạc.
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_audio = gr.Audio(
                label="Tải file nhạc của bạn lên tại đây", 
                type="filepath"
            )
            btn_predict = gr.Button("🔮 Bắt đầu phân tích & Dự đoán", variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(
                label="Kết quả từ AI", 
                placeholder="Kết quả dự đoán sẽ hiển thị ở đây...",
                text_align="center"
            )
            
    btn_predict.click(
        fn=gradio_predict, 
        inputs=input_audio, 
        outputs=output_text
    )

# Chạy ứng dụng web độc lập
if __name__ == "__main__":
    # share=False để chạy mạng nội bộ, đổi thành True nếu muốn lấy link gửi cho bạn bè dùng thử
    demo.launch(share=False)
