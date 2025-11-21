import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tcn import TCN # Pastikan library tcn sudah terinstal: pip install keras-tcn
import warnings

# Suppress warnings from TensorFlow/Keras
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# --- Konfigurasi ---
DATASET_DIR = 'model/datasett'
MODELS_BASE_DIR = 'model/modelss'
WINDOW_SIZE = 10
EPOCHS = 50
BATCH_SIZE = 16
PATIENCE = 5 # For EarlyStopping

# --- Fungsi untuk membuat dataset ---
def create_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# --- Fungsi untuk melatih, menyimpan, dan memplot model TCN ---
def train_and_plot_tcn_model(drug_name, df_path):
    print(f"\n--- Memulai pelatihan TCN untuk {drug_name} ---")

    df = pd.read_csv(df_path)
    df.columns = df.columns.str.strip().str.lower()
    df.rename(columns={'tanggal': 'date', 'harga': 'price'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.sort_index()

    # Pastikan data cukup untuk window_size
    if len(df) < WINDOW_SIZE + 1:
        print(f"Dataset untuk {drug_name} terlalu kecil ({len(df)} entri) untuk window_size {WINDOW_SIZE}. Melewatkan pelatihan.")
        return

    # Normalisasi harga
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[['price']])

    # Buat sequence window
    X, y = create_dataset(data_scaled, WINDOW_SIZE)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape((-1, 1))

    # Split data training/test (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Bangun model TCN
    model = Sequential()
    # Menambahkan layer TCN dengan konfigurasi yang umum
    model.add(TCN(nb_filters=64, kernel_size=3, dilations=[1, 2, 4],
                  input_shape=(WINDOW_SIZE, 1), activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    print(f"Summary model TCN untuk {drug_name}:")
    model.summary()

    # Training dengan EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stop], verbose=1) # Set verbose=1 to see training progress

    # Simpan model dan scaler di folder khusus obat
    drug_model_dir = os.path.join(MODELS_BASE_DIR, drug_name)
    os.makedirs(drug_model_dir, exist_ok=True)

    model.save(os.path.join(drug_model_dir, 'tcn_model.h5'))
    joblib.dump(scaler, os.path.join(drug_model_dir, 'scaler_tcn.gz'))

    print(f"✅ TCN model dan scaler untuk {drug_name} disimpan di: {drug_model_dir}")

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'TCN Model Loss for {drug_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Main loop untuk setiap obat ---
if __name__ == "__main__":
    os.makedirs(DATASET_DIR, exist_ok=True) # Pastikan direktori dataset ada
    for filename in os.listdir(DATASET_DIR):
        if filename.endswith('.csv'):
            drug_name = os.path.splitext(filename)[0] # Ambil nama obat tanpa ekstensi
            file_path = os.path.join(DATASET_DIR, filename)
            train_and_plot_tcn_model(drug_name, file_path)