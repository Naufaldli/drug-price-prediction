import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
import os
from tcn import TCN # Pastikan TCN diimpor jika menggunakan model TCN

# --- Konfigurasi (sesuaikan dengan skrip train Anda) ---
MODELS_BASE_DIR = 'model/models'
DATASET_DIR = 'model/dataset'
WINDOW_SIZE = 10
DRUG_NAME = 'Amlodipin' # Ganti dengan nama obat yang relevan
MODEL_TYPE = 'LSTM' # Ganti 'TCN' jika menggunakan model TCN

# Jalur ke model dan scaler yang sudah tersimpan
model_path = os.path.join(MODELS_BASE_DIR, DRUG_NAME, f'{MODEL_TYPE.lower()}_model.h5')
scaler_path = os.path.join(MODELS_BASE_DIR, DRUG_NAME, f'scaler_{MODEL_TYPE.lower()}.gz')
data_path = os.path.join(DATASET_DIR, f'{DRUG_NAME}.csv')

try:
    # 1. Muat kembali model dan scaler
    if MODEL_TYPE == 'LSTM':
        model = load_model(model_path)
    elif MODEL_TYPE == 'TCN':
        # Untuk TCN, Anda mungkin perlu custom_objects jika ada lapisan kustom
        model = load_model(model_path, custom_objects={'TCN': TCN})
    
    scaler = joblib.load(scaler_path)

    # 2. Muat dan siapkan data (mirip dengan proses di train_*.py)
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip().str.lower()
    df.rename(columns={'tanggal': 'date', 'harga': 'price'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.sort_index()

    data_scaled = scaler.transform(df[['price']].values)

    X, y = [], []
    for i in range(len(data_scaled) - WINDOW_SIZE):
        X.append(data_scaled[i:(i + WINDOW_SIZE)])
        y.append(data_scaled[i + WINDOW_SIZE])
    X = np.array(X)
    y = np.array(y)

    # Membagi data menjadi train dan test (sesuai proporsi yang Anda gunakan di training, misalnya 80:20)
    # Gunakan split yang sama persis seperti saat training!
    train_size = int(len(y) * 0.8) # Contoh 80% data latih
    X_test, y_test = X[train_size:], y[train_size:]

    # 3. Lakukan prediksi pada X_test
    # Pastikan bentuk X_test sesuai dengan input model
    # LSTM/TCN biasanya menerima input (batch_size, window_size, features)
    if MODEL_TYPE == 'LSTM':
        # LSTM input shape (samples, timesteps, features)
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    elif MODEL_TYPE == 'TCN':
        # TCN input shape (samples, timesteps, features)
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    predictions_scaled = model.predict(X_test_reshaped, verbose=0)

    # 4. Inverse transform y_test dan predictions_scaled
    # y_test dan predictions_scaled mungkin perlu di-reshape kembali ke 2D sebelum inverse_transform
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    predictions_original = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

    print("\n--- Data untuk Perhitungan Manual ---")
    print(f"Jumlah data aktual (y_test_original): {len(y_test_original)}")
    print(f"Contoh 5 nilai aktual: {y_test_original}")
    print(f"Jumlah data prediksi (predictions_original): {len(predictions_original)}")
    print(f"Contoh 5 nilai prediksi: {predictions_original}")

    # Sekarang Anda memiliki y_test_original dan predictions_original
    # yang dapat Anda gunakan untuk perhitungan MAE, RMSE, MAPE secara manual
    # seperti yang dijelaskan sebelumnya.
except Exception as e:
    print(f"Terjadi kesalahan saat mendapatkan data: {e}")
    print("Pastikan Anda telah melatih dan menyimpan model serta scaler dengan benar, dan jalur filenya sesuai.")