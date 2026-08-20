import os
import sys
import numpy as np
import librosa
from typing import Dict, Any, Tuple, List

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

class AudioProcessor:
    def __init__(self):
        pass

    @staticmethod
    def extract_features_and_metrics(audio_path: str, duration: float = 30.0) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Trích xuất 60 đặc trưng âm học chuẩn GTZAN cho Model AI
        đồng thời tính toán các chỉ số trực quan (Acoustic Metrics) và Waveform cho Frontend.
        """
        # 1. Tải tín hiệu âm thanh
        y, sr = librosa.load(audio_path, duration=duration)
        total_duration = float(librosa.get_duration(y=y, sr=sr))

        # 2. Trích xuất các đặc trưng cơ bản
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(tempo[0] if isinstance(tempo, (list, np.ndarray)) else tempo)

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

        # Phân tách hòa âm (harmonic) và tiết tấu (percussive)
        harmony_wave = librosa.effects.harmonic(y)
        percussive_wave = librosa.effects.percussive(y)
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)

        # 3. Tạo dictionary 60 đặc trưng (khớp 100% với scaler/model)
        features: Dict[str, Any] = {
            'length': len(y),
            'chroma_stft_mean': float(np.mean(chroma_stft)),
            'chroma_stft_var': float(np.var(chroma_stft)),
            'rms_mean': float(np.mean(rmse)),
            'rms_var': float(np.var(rmse)),
            'spectral_centroid_mean': float(np.mean(spec_cent)),
            'spectral_centroid_var': float(np.var(spec_cent)),
            'spectral_bandwidth_mean': float(np.mean(spec_bw)),
            'spectral_bandwidth_var': float(np.var(spec_bw)),
            'rolloff_mean': float(np.mean(rolloff)),
            'rolloff_var': float(np.var(rolloff)),
            'zero_crossing_rate_mean': float(np.mean(zcr)),
            'zero_crossing_rate_var': float(np.var(zcr)),
            'harmony_mean': float(np.mean(harmony_wave)),
            'harmony_var': float(np.var(harmony_wave)),
            'perceptr_mean': float(np.mean(percussive_wave)),
            'perceptr_var': float(np.var(percussive_wave)),
            'tempo': tempo_val,
            'chroma_cqt_mean': float(np.mean(chroma_cqt)),
            'chroma_cqt_var': float(np.var(chroma_cqt))
        }

        # Thêm 40 giá trị MFCC (mean & var)
        for i in range(20):
            features[f'mfcc{i+1}_mean'] = float(np.mean(mfcc[i]))
            features[f'mfcc{i+1}_var'] = float(np.var(mfcc[i]))

        # 4. Trích xuất chỉ số thân thiện cho giao diện người dùng
        # Downsample waveform (100 điểm) để vẽ biểu đồ sóng âm tức thời trên web
        step = max(1, len(y) // 100)
        waveform_sample = [round(float(abs(val)), 3) for val in y[::step][:100]]

        # Tính tỷ lệ hài hòa / tiết tấu
        harm_energy = float(np.sum(harmony_wave ** 2))
        perc_energy = float(np.sum(percussive_wave ** 2))
        total_hp = harm_energy + perc_energy + 1e-9
        harmonic_ratio = round((harm_energy / total_hp) * 100, 1)
        percussive_ratio = round((perc_energy / total_hp) * 100, 1)

        metrics: Dict[str, Any] = {
            "duration_seconds": round(total_duration, 2),
            "sample_rate_hz": sr,
            "tempo_bpm": round(tempo_val, 1),
            "rms_energy": round(float(np.mean(rmse)), 4),
            "spectral_centroid_hz": round(float(np.mean(spec_cent)), 1),
            "spectral_bandwidth_hz": round(float(np.mean(spec_bw)), 1),
            "spectral_rolloff_hz": round(float(np.mean(rolloff)), 1),
            "zero_crossing_rate": round(float(np.mean(zcr)), 4),
            "harmonic_ratio": harmonic_ratio,
            "percussive_ratio": percussive_ratio,
            "waveform_preview": waveform_sample,
            "mfcc_top_summary": [round(float(np.mean(mfcc[i])), 2) for i in range(6)]
        }

        return features, metrics

audio_processor = AudioProcessor()
