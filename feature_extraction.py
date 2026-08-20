import pandas as pd
import os
import numpy as np
import librosa
import sys

from multiprocessing import Pool, cpu_count
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# ==========================================================
# FIX UNICODE TERMINAL WINDOWS
# ==========================================================
sys.stdout.reconfigure(encoding='utf-8')

# ==========================================================
# ĐƯỜNG DẪN DATASET
# ==========================================================
path = "genres_original/"

# ==========================================================
# HÀM TRÍCH XUẤT ĐẶC TRƯNG ÂM THANH
# ==========================================================
def get_all_musical_features(
    audio_path,
    song_name,
    start=0,
    chroma_method_list=['cqt']
):

    # LOAD AUDIO
    y, sr = librosa.load(
        audio_path,
        offset=start,
        duration=30.0
    )

    # ======================================================
    # FEATURE EXTRACTION
    # ======================================================

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr) #tính nhịp độ bài hát

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr) #tính chroma stft

    rmse = librosa.feature.rms(y=y) #tính root mean square energy (RMSE) - năng lượng của tín hiệu âm thanh

    spec_cent = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    ) #tính spectral centroid - vị trí trung bình của phổ âm thanh

    spec_bw = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    ) #tính spectral bandwidth - độ rộng phổ âm thanh

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    ) #tính spectral rolloff - tần số mà phổ âm thanh giảm xuống dưới một ngưỡng nhất định 

    zcr = librosa.feature.zero_crossing_rate(y) #tính zero crossing rate - tốc độ thay đổi dấu của tín hiệu âm thanh

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20
    )#tính Mel-frequency cepstral coefficients (MFCCs) - đặc trưng phổ âm thanh được sử dụng rộng rãi trong nhận dạng âm thanh

    harmony = librosa.effects.harmonic(y)#tách phần hài hòa của tín hiệu âm thanh

    percussive = librosa.effects.percussive(y) #tách phần gõ của tín hiệu âm thanh

    # ======================================================
    # FEATURE DICTIONARY
    # ======================================================

    features = {

        'length': len(y), #độ dài của tín hiệu âm thanh (số mẫu)

        'chroma_stft_mean': np.mean(chroma_stft), #tính giá trị trung bình của chroma stft`
        'chroma_stft_var': np.var(chroma_stft), #tính giá trị phương sai của chroma stft

        'rms_mean': np.mean(rmse),#tính giá trị trung bình của RMSE
        'rms_var': np.var(rmse),#tính giá trị phương sai của RMSE

        'spectral_centroid_mean': np.mean(spec_cent),
        'spectral_centroid_var': np.var(spec_cent),

        'spectral_bandwidth_mean': np.mean(spec_bw),
        'spectral_bandwidth_var': np.var(spec_bw),

        'rolloff_mean': np.mean(rolloff),
        'rolloff_var': np.var(rolloff),

        'zero_crossing_rate_mean': np.mean(zcr),
        'zero_crossing_rate_var': np.var(zcr),

        'harmony_mean': np.mean(harmony),
        'harmony_var': np.var(harmony),

        'perceptr_mean': np.mean(percussive),
        'perceptr_var': np.var(percussive),

        'tempo': tempo[0]
        if isinstance(tempo, (list, np.ndarray))
        else tempo
    }#tính giá trị trung bình và phương sai của các đặc trưng đã trích xuất, đồng thời lưu vào một dictionary có tên là features

    # ======================================================
    # MFCC FEATURES
    # ======================================================

    for i in range(20):

        features[f'mfcc{i+1}_mean'] = np.mean(mfcc[i])#tính giá trị trung bình của hệ số MFCC thứ i và lưu vào dictionary features với tên khóa tương ứng

        features[f'mfcc{i+1}_var'] = np.var(mfcc[i])#tính giá trị trung bình và phương sai của từng hệ số MFCC và lưu vào dictionary features với tên khóa tương ứng

    # ======================================================
    # CHROMA CQT
    # ======================================================

    if 'cqt' in chroma_method_list:

        chroma_cqt = librosa.feature.chroma_cqt(
            y=y,
            sr=sr
        )

        features['chroma_cqt_mean'] = np.mean(chroma_cqt)

        features['chroma_cqt_var'] = np.var(chroma_cqt)

    # ======================================================
    # RETURN DATAFRAME
    # ======================================================

    return pd.DataFrame(
        features,
        index=[song_name]
    )

# ==========================================================
# HÀM XỬ LÝ 1 FILE NHẠC
# ==========================================================
def extract_single_song(wav_path):

    try:
        song_name = (
            os.path.basename(wav_path)
            .replace('.wav', '')
        )

        df_ = get_all_musical_features(
            wav_path,
            song_name,
            start=0,
            chroma_method_list=['cqt']
        )

        return df_

    except Exception as e:

        print(f"Lỗi khi xử lý file {wav_path}: {e}")

        return None

# ==========================================================
# MAIN
# ==========================================================
if __name__ == '__main__':

    # ======================================================
    # DANH SÁCH THỂ LOẠI
    # ======================================================

    genre_list = [
        'blues',
        'classical',
        'country',
        'disco',
        'hiphop',
        'jazz',
        'metal',
        'pop',
        'reggae',
        'rock'
    ]

    # ======================================================
    # GOM FILE WAV
    # ======================================================

    wav_list = []

    for g in genre_list:

        genre_path = os.path.join(path, g)

        if os.path.exists(genre_path):

            for wav in os.listdir(genre_path):

                if wav.endswith('.wav'):

                    wav_list.append(
                        os.path.join(genre_path, wav)
                    )

    total_songs = len(wav_list)

    # ======================================================
    # CHECK DATASET
    # ======================================================

    if total_songs == 0:

        print(
            "Không tìm thấy dataset!"
        )

    else:

        num_processors = min(16, cpu_count())

        print(
            f"Tìm thấy {total_songs} bài hát."
        )

        print(
            f"Đang dùng {num_processors} nhân CPU..."
        )

        # ==================================================
        # MULTIPROCESSING
        # ==================================================

        with Pool(processes=num_processors) as pool:

            df_list = pool.map(
                extract_single_song,
                wav_list
            )

        # ==================================================
        # LỌC FILE LỖI
        # ==================================================

        df_list = [
            df for df in df_list
            if df is not None
        ]

        print(
            "Đã trích xuất xong!"
        )

        # ==================================================
        # GỘP DATAFRAME
        # ==================================================

        df_features = pd.concat(
            df_list,
            axis=0
        )

        # ==================================================
        # LABEL
        # ==================================================

        labels = []

        for i in df_features.index:

            labels.append(
                i.split('.')[0]
            )

        df_features['genre_label'] = labels

        # ==================================================
        # FILE PATH
        # ==================================================

        file_paths = []

        for i in df_features.index:

            genre = i.split('.')[0]

            file_paths.append(
                f"genres_original/{genre}/{i}.wav"
            )

        df_features['file_path'] = file_paths

        # ==================================================
        # SAVE CSV
        # ==================================================

        output_csv_path = (
            'example_features_with_path.csv'
        )

        df_features.to_csv(
            output_csv_path,
            index_label='filename'
        )

        print(
            f"Đã lưu CSV: {output_csv_path}"
        )

        # ==================================================
        # MYSQL
        # ==================================================

        try:

            username = os.getenv("DB_USER", "root")
            raw_password = os.getenv("DB_PASSWORD", "501130")
            password = quote_plus(raw_password)
            host = os.getenv("DB_HOST", "127.0.0.1")
            port = os.getenv("DB_PORT", "3306")
            database_name = os.getenv("DB_NAME", "music_db")

            connection_string = (f"mysql+pymysql://{username}:{password}@{host}:{port}/{database_name}")

            print(
                "Đang kết nối MySQL..."
            )
            engine = create_engine(
                connection_string
            )

            # TEST KẾT NỐI
            with engine.connect() as conn:

                print(
                    "Kết nối MySQL thành công!"
                )

            # RESET INDEX
            df_features_sql = (
                df_features
                .reset_index()
                .rename(
                    columns={
                        'index': 'filename'
                    }
                )
            )

            print(
                "Đang ghi dữ liệu vào bảng songs_features..."
            )

            # SAVE MYSQL
            df_features_sql.to_sql(
                name='songs_features',
                con=engine,
                if_exists='replace',
                index=False
            )

            print(
                "Đã lưu dữ liệu vào MySQL!"
            )

        except Exception as e:

            print(
                f"Lỗi MySQL: {e}"
            )

        print(
            "HOÀN THÀNH!"
        )
