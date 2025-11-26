import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error # Tambahkan import ini

# Import fungsi create_sequences dari preprocess.py
from utils.preprocess import create_sequences

# Fungsi untuk prediksi menggunakan LSTM
def predict_future_lstm(df_price, model, scaler, days_ahead=30, window_size=10):
    try:
        print(f"\n--- DEBUGGING INSIDE predict_future_lstm ---")
        print(f"  Input df_price head:\n{df_price.head()}")
        print(f"  Input df_price dtypes:\n{df_price.dtypes}")
        print(f"  Input df_price shape: {df_price.shape}")
        print(f"  Scaler object type: {type(scaler)}")
        print(f"  Model object type: {type(model)}")

        if not hasattr(scaler, 'transform') or not hasattr(scaler, 'inverse_transform'):
            raise AttributeError("Scaler object does not have 'transform' or 'inverse_transform' method.")

        if 'price' not in df_price.columns:
            raise ValueError("Kolom 'price' tidak ditemukan di DataFrame input untuk prediksi.")

        scaled_history_data = scaler.transform(df_price[['price']])
        
        # --- Prediksi In-Sample (untuk data historis) ---
        # Buat sequence untuk in-sample predictions dari seluruh data historis yang diskalakan
        # Perhatikan: y_actual_in_sample_scaled adalah target untuk X_in_sample
        if len(scaled_history_data) <= window_size:
            # Jika data terlalu pendek untuk membuat sequence, inisialisasi kosong
            in_sample_predictions_inverse = np.array([])
            y_actual_in_sample_inverse = np.array([])
            print("  Not enough historical data to create in-sample sequences for prediction.")
        else:
            X_in_sample, y_actual_in_sample_scaled = create_sequences(scaled_history_data, window_size)
            
            # Reshape X_in_sample sesuai input model (samples, timesteps, features)
            X_in_sample = X_in_sample.reshape((X_in_sample.shape[0], X_in_sample.shape[1], 1))

            # Lakukan prediksi in-sample
            in_sample_predictions_scaled = model.predict(X_in_sample, verbose=0)
            
            # Inverse transform untuk mendapatkan nilai asli
            in_sample_predictions_inverse = scaler.inverse_transform(in_sample_predictions_scaled).flatten()
            y_actual_in_sample_inverse = scaler.inverse_transform(y_actual_in_sample_scaled).flatten()

        # Hitung metrik untuk in-sample predictions
        # Hanya hitung jika ada data yang valid untuk perbandingan
        if len(y_actual_in_sample_inverse) > 0:
            mae = mean_absolute_error(y_actual_in_sample_inverse, in_sample_predictions_inverse)
            rmse = np.sqrt(mean_squared_error(y_actual_in_sample_inverse, in_sample_predictions_inverse))
            
            # Pastikan tidak ada pembagian dengan nol untuk MAPE
            non_zero_actuals = y_actual_in_sample_inverse[y_actual_in_sample_inverse != 0]
            corresponding_predictions = in_sample_predictions_inverse[y_actual_in_sample_inverse != 0]
            if len(non_zero_actuals) > 0:
                mape = np.mean(np.abs((non_zero_actuals - corresponding_predictions) / non_zero_actuals)) * 100
            else:
                mape = np.nan # Atau nilai lain yang sesuai jika tidak ada nilai aktual non-nol
        else:
            mae, rmse, mape = np.nan, np.nan, np.nan # Jika tidak ada data in-sample, metrik adalah NaN

        print(f"--- DEBUGGING FROM predict_future_lstm (In-Sample Metrics) ---")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"-------------------------------------------------------------")

        # --- Prediksi Masa Depan (Existing Logic) ---
        predicted_prices = []
        # Pastikan current_input memiliki bentuk yang benar (window_size, 1)
        # Ambil WINDOW_SIZE data terakhir dari scaled_history_data untuk input awal prediksi masa depan
        current_input = scaled_history_data[-window_size:].reshape(window_size, 1)

        for _ in range(days_ahead):
            current_input_reshaped = current_input.reshape(1, window_size, 1)
            predicted_scaled_price = model.predict(current_input_reshaped, verbose=0)[0, 0]
            predicted_prices.append(predicted_scaled_price)
            # Update current_input dengan prediksi terbaru
            current_input = np.append(current_input[1:], [[predicted_scaled_price]], axis=0)
        
        predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1)).flatten()

        last_date_in_history = df_price.index[-1]
        future_dates = [last_date_in_history + timedelta(days=i+1) for i in range(days_ahead)]
        future_labels = [d.strftime('%Y-%m-%d') for d in future_dates]
        
        print(f"\n--- DEBUGGING FROM predict_future_lstm (Final Output) ---")
        print(f"  Length of predicted_prices (future): {len(predicted_prices)}")
        print(f"  First 5 predicted prices (future): {predicted_prices[:5]}")
        print(f"-------------------------------------------------------")

        # Mengembalikan prediksi masa depan, label masa depan, prediksi in-sample, dan metrik
        return predicted_prices, future_labels, in_sample_predictions_inverse, mae, rmse, mape

    except Exception as e:
        print(f"ERROR in predict_future_lstm: {e}")
        raise # Rethrow exception to be caught by Flask route


# Fungsi untuk prediksi menggunakan TCN (terapkan perubahan yang sama seperti LSTM)
def predict_future_tcn(df_price, model, scaler, days_ahead=30, window_size=10):
    try:
        print(f"\n--- DEBUGGING INSIDE predict_future_tcn ---")
        print(f"  Input df_price head:\n{df_price.head()}")
        print(f"  Input df_price dtypes:\n{df_price.dtypes}")
        print(f"  Input df_price shape: {df_price.shape}")
        print(f"  Scaler object type: {type(scaler)}")
        print(f"  Model object type: {type(model)}")

        if not hasattr(scaler, 'transform') or not hasattr(scaler, 'inverse_transform'):
            raise AttributeError("Scaler object does not have 'transform' or 'inverse_transform' method.")
        
        if 'price' not in df_price.columns:
            raise ValueError("Kolom 'price' tidak ditemukan di DataFrame input untuk prediksi.")

        scaled_history_data = scaler.transform(df_price[['price']])
        
        # --- Prediksi In-Sample (untuk data historis) ---
        if len(scaled_history_data) <= window_size:
            in_sample_predictions_inverse = np.array([])
            y_actual_in_sample_inverse = np.array([])
            print("  Not enough historical data to create in-sample sequences for prediction (TCN).")
        else:
            X_in_sample, y_actual_in_sample_scaled = create_sequences(scaled_history_data, window_size)
            X_in_sample = X_in_sample.reshape((X_in_sample.shape[0], X_in_sample.shape[1], 1))

            in_sample_predictions_scaled = model.predict(X_in_sample, verbose=0)
            
            in_sample_predictions_inverse = scaler.inverse_transform(in_sample_predictions_scaled).flatten()
            y_actual_in_sample_inverse = scaler.inverse_transform(y_actual_in_sample_scaled).flatten()

        # Hitung metrik untuk in-sample predictions
        if len(y_actual_in_sample_inverse) > 0:
            mae = mean_absolute_error(y_actual_in_sample_inverse, in_sample_predictions_inverse)
            rmse = np.sqrt(mean_squared_error(y_actual_in_sample_inverse, in_sample_predictions_inverse))
            
            non_zero_actuals = y_actual_in_sample_inverse[y_actual_in_sample_inverse != 0]
            corresponding_predictions = in_sample_predictions_inverse[y_actual_in_sample_inverse != 0]
            if len(non_zero_actuals) > 0:
                mape = np.mean(np.abs((non_zero_actuals - corresponding_predictions) / non_zero_actuals)) * 100
            else:
                mape = np.nan
        else:
            mae, rmse, mape = np.nan, np.nan, np.nan

        print(f"--- DEBUGGING FROM predict_future_tcn (In-Sample Metrics) ---")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"-------------------------------------------------------------")

        # --- Prediksi Masa Depan (Existing Logic) ---
        predicted_prices = []
        current_input = scaled_history_data[-window_size:].reshape(window_size, 1)

        for _ in range(days_ahead):
            current_input_reshaped = current_input.reshape(1, window_size, 1)
            predicted_scaled_price = model.predict(current_input_reshaped, verbose=0)[0, 0]
            predicted_prices.append(predicted_scaled_price)
            current_input = np.append(current_input[1:], [[predicted_scaled_price]], axis=0)
        
        predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1)).flatten()

        future_dates = [df_price.index[-1] + timedelta(days=i+1) for i in range(days_ahead)]
        future_labels = [d.strftime('%Y-%m-%d') for d in future_dates]
        
        print(f"\n--- DEBUGGING FROM predict_future_tcn (Final Output) ---")
        print(f"  Length of predicted_prices (future): {len(predicted_prices)}")
        print(f"  First 5 predicted prices (future): {predicted_prices[:5]}")
        print(f"-------------------------------------------------------")

        # Mengembalikan prediksi masa depan, label masa depan, prediksi in-sample, dan metrik
        return predicted_prices, future_labels, in_sample_predictions_inverse, mae, rmse, mape

    except Exception as e:
        print(f"ERROR in predict_future_tcn: {e}")
        raise