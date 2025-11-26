import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)]) # Input: urutan data selama window_size
        y.append(data[i + window_size])     # Output: data setelah urutan tersebut
    return np.array(X), np.array(y)

def load_and_prepare_data(filepath, window_size=10, tcn_mode=False, features=None, scaler_obj=None):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()
    df.rename(columns={'tanggal': 'date', 'harga': 'price'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.sort_index()

    if features is None:
        features_to_use = ['price']
    else:
        features_to_use = features
    
    data = df[features_to_use].values

    if scaler_obj is None:
        raise ValueError("Scaler object must be provided for data scaling.")
    
    data_scaled = scaler_obj.transform(data) # <--- Perubahan Kunci di sini

    X, y = create_sequences(data_scaled, window_size)

    # Pastikan output shape cocok dengan model
    if tcn_mode:
        X = X.reshape(X.shape[0], X.shape[1], len(features_to_use))
        y = y.reshape(-1, len(features_to_use)) 
    else: # LSTM
        X = X.reshape(X.shape[0], X.shape[1], len(features_to_use))
        y = y.reshape(-1, len(features_to_use)) 
    
    return df, X, y