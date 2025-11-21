import os
import math
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, flash, redirect, url_for, session as login_session, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
# Import semua metrik yang dibutuhkan di sini
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Pastikan TCN sudah terinstal dan import dengan benar.
try:
    from tcn import TCN
except ImportError:
    print("WARNING: TCN library not found. TCN models might fail to load if they contain TCN layers.")
    # Define a dummy TCN class for graceful error handling if library is missing
    class TCN:
        def __init__(self, *args, **kwargs):
            pass # Dummy TCN for graceful error handling if library is missing

# Import fungsi dari modul Anda
from utils.preprocess import load_and_prepare_data, create_sequences # Pastikan create_sequences juga diimpor
from predict_future_multi import predict_future_lstm as future_lstm_func, predict_future_tcn as future_tcn_func

# ==================== APLIKASI FLASK ====================
app = Flask(__name__)
# Kunci rahasia tetap dibutuhkan untuk fitur flash messages dan session (jika digunakan)
app.secret_key = 'super_secret_key_for_simple_app' 

# ==================== KONFIGURASI PATHS & FOLDERS ====================
BASE_MODELS_DIR = 'model/modelss/'
DATASET_DIR = 'model/datasett'       # Direktori tempat file CSV obat berada
# OUTPUT_DIR dan EXPORT_DIR tidak lagi digunakan karena tidak ada fitur download/grafik atau penyimpanan file di server

# Direktori untuk menyimpan file CSV sementara yang diunggah.
TEMP_FOLDER = 'temp' 

# Konfigurasi aplikasi Flask
app.config['TEMP_FOLDER'] = TEMP_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Batasan ukuran file (16 MB)

# Buat direktori jika belum ada (hanya yang diperlukan)
os.makedirs(BASE_MODELS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

# ==================== FUNGSI PEMBANTU ====================

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    non_zero_true_indices = y_true != 0
    
    if not np.any(non_zero_true_indices):
        return np.inf 
    
    mape = np.mean(np.abs((y_true[non_zero_true_indices] - y_pred[non_zero_true_indices]) / y_true[non_zero_true_indices])) * 100
    
    return mape

def load_model_and_scaler_for_drug(drug_name, model_type):
    """
    Memuat model dan scaler yang spesifik untuk nama obat dan tipe model tertentu.
    """
    model_path = os.path.join(BASE_MODELS_DIR, drug_name, f'{model_type.lower()}_model.h5')
    scaler_path = os.path.join(BASE_MODELS_DIR, drug_name, f'scaler_{model_type.lower()}.gz')

    model = None
    scaler = None

    if not os.path.exists(model_path):
        print(f"DEBUG: Model tidak ditemukan untuk '{drug_name}' ({model_type}) di '{model_path}'.")
        return None, None
    if not os.path.exists(scaler_path):
        print(f"DEBUG: Scaler tidak ditemukan untuk '{drug_name}' ({model_type}) di '{scaler_path}'.")
        return None, None

    try:
        if model_type.lower() == 'tcn':
            model = load_model(model_path, custom_objects={'TCN': TCN})
        else: # LSTM
            model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        print(f"DEBUG: Model dan scaler untuk '{drug_name}' ({model_type}) berhasil dimuat dari: {model_path}")
        return model, scaler
    except Exception as e:
        print(f"ERROR: Gagal memuat model/scaler untuk '{drug_name}' ({model_type}): {e}")
        return None, None

# ==================== ROUTES FLASK ====================

@app.route('/')
def home():
    try:
        # Ambil daftar obat dari nama file CSV di direktori dataset
        obat_list_from_dataset = [os.path.splitext(f)[0].replace('_', ' ').title() for f in os.listdir(DATASET_DIR) if f.endswith('.csv')]
        obat_list = sorted(list(set(obat_list_from_dataset)))
        
        return render_template('home.html', obat_list=obat_list, max_date=None, harga_prediksi=None)
    except Exception as e:
        flash(f"Terjadi kesalahan saat memuat beranda: {e}", "danger")
        print(f"DEBUG: Error in home route: {e}")
        return render_template('home.html', obat_list=[], max_date=None, harga_prediksi=None)

@app.route('/predict', methods=['POST'])
def predict():
    temp_filepath = None
    try:
        model_type = request.form.get('model')
        file = request.files.get('file')
        nama_obat_selected = request.form.get('nama_obat') 

        nama_obat_display = "" 

        if not model_type:
            flash("Tipe model tidak valid. Pilih 'LSTM' atau 'TCN'.", "danger")
            return redirect(url_for('home'))

        if file and file.filename != '':
            filename = secure_filename(file.filename)
            drug_name = os.path.splitext(filename)[0]
            nama_obat_display = drug_name.replace('_', ' ').title() 
            temp_filepath = os.path.join(app.config['TEMP_FOLDER'], filename)
            file.save(temp_filepath)
            csv_path = temp_filepath
        elif nama_obat_selected:
            drug_name = nama_obat_selected.lower().replace(' ', '_')
            nama_obat_display = nama_obat_selected 
            filename = f"{drug_name}.csv"
            csv_path = os.path.join(DATASET_DIR, filename)
            if not os.path.exists(csv_path):
                flash(f"File dataset untuk obat '{nama_obat_selected}' tidak ditemukan di server.", "danger")
                return redirect(url_for('home'))
        else:
            flash("Input tidak valid. Harap pilih obat dari daftar atau unggah file CSV.", "danger")
            return redirect(url_for('home'))

        model, scaler = load_model_and_scaler_for_drug(drug_name, model_type)

        if model is None or scaler is None:
            flash(f"Model atau scaler untuk '{nama_obat_display}' ({model_type}) tidak ditemukan atau rusak. Harap pastikan model telah dilatih untuk obat ini.", "danger")
            return redirect(url_for('home'))

        window_size = 10 # Pastikan ini adalah window_size yang benar untuk model Anda

        # Load data mentah dari CSV untuk mendapatkan df_all
        df_all = pd.read_csv(csv_path)
        df_all.columns = df_all.columns.str.strip().str.lower()
        df_all.rename(columns={'tanggal': 'date', 'harga': 'price'}, inplace=True)
        df_all['date'] = pd.to_datetime(df_all['date'])
        df_all.set_index('date', inplace=True)
        df_all = df_all.sort_index()

        if df_all.empty or 'price' not in df_all.columns:
            flash("Data CSV tidak valid atau kosong setelah preprocessing.", "danger")
            return redirect(url_for('home'))
        
        # Panggil fungsi prediksi yang sesuai dan TANGKAP SEMUA NILAI YANG DIKEMBALIKAN, TERMASUK METRIK
        if model_type == 'LSTM':
            # predicted_future_prices tidak digunakan di route ini, tapi tetap ditangkap
            _, _, in_sample_predictions_raw, mae, rmse, mape = future_lstm_func(df_all[['price']], model, scaler, days_ahead=1, window_size=window_size)
        elif model_type == 'TCN':
            # predicted_future_prices tidak digunakan di route ini, tapi tetap ditangkap
            _, _, in_sample_predictions_raw, mae, rmse, mape = future_tcn_func(df_all[['price']], model, scaler, days_ahead=1, window_size=window_size)
        else:
            flash("Jenis model tidak dikenal.", "danger")
            return redirect(url_for('home'))

        # --- Persiapan Data untuk Grafik di result.html (Dua Garis) ---

        # 1. Labels untuk seluruh grafik (seluruh periode historis)
        all_labels_for_chart = df_all.index.strftime('%Y-%m-%d').tolist()

        # 2. Data Aktual Historis (seluruhnya)
        actual_historical_data_for_chart = df_all['price'].tolist()

        # 3. Prediksi In-Sample untuk seluruh data historis
        # in_sample_predictions_raw sudah didapatkan dari future_lstm_func/future_tcn_func
        in_sample_predictions_padded = [None] * window_size + in_sample_predictions_raw.tolist() # Pastikan ini adalah list

        in_sample_predictions_final = in_sample_predictions_padded[:len(actual_historical_data_for_chart)]
        if len(in_sample_predictions_final) < len(actual_historical_data_for_chart):
            in_sample_predictions_final += [None] * (len(actual_historical_data_for_chart) - len(in_sample_predictions_final))


        flash("Prediksi berhasil dibuat!", "success")

        login_session['last_predicted_drug_raw'] = drug_name

        print("\n--- DEBUGGING DATA UNTUK RESULT.HTML (FINAL PREP) ---")
        print(f"  all_labels_for_chart (len {len(all_labels_for_chart)}): {all_labels_for_chart[:5]} ... {all_labels_for_chart[-5:]}")
        print(f"  actual_historical_data_for_chart (len {len(actual_historical_data_for_chart)}): {actual_historical_data_for_chart[:5]} ... {actual_historical_data_for_chart[-5:]}")
        print(f"  in_sample_predictions_final (len {len(in_sample_predictions_final)}): {in_sample_predictions_final[:5]} ... {in_sample_predictions_final[-5:]}")
        print(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%") # Debug metrik
        print("--------------------------------------------------\n")

        return render_template('result.html',
                               nama_obat=nama_obat_display,
                               model=model_type,
                               mae=round(mae, 4), rmse=round(rmse, 4), mape=round(mape, 4), # Teruskan metrik yang baru dihitung
                               labels=all_labels_for_chart,
                               actual_data=actual_historical_data_for_chart,
                               in_sample_predictions=in_sample_predictions_final,
                               zip=zip
                               )

    except Exception as e:
        flash(f"Terjadi kesalahan prediksi: {str(e)}", "danger")
        print(f"ERROR: Error in predict route: {e}")
        return redirect(url_for('home'))
    finally:
        if temp_filepath and os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            print(f"DEBUG: File sementara '{temp_filepath}' berhasil dihapus.")

@app.route('/comparison')
def comparison():
    print("--- DEBUG COMPARISON ROUTE ---")
    try:
        # Ambil daftar obat dari nama file CSV di direktori dataset
        obat_csv_files_raw = [os.path.splitext(f)[0] for f in os.listdir(DATASET_DIR) if f.endswith('.csv')]
        
        if not obat_csv_files_raw:
            flash("Tidak ada dataset obat yang tersedia untuk perbandingan. Harap unggah atau tambahkan dataset.", "info")
            return redirect(url_for('home'))
            
        drug_name_for_comparison_raw = None
        
        if 'last_predicted_drug_raw' in login_session:
            temp_drug_name = login_session['last_predicted_drug_raw']
            if temp_drug_name in obat_csv_files_raw:
                drug_name_for_comparison_raw = temp_drug_name
                print(f"DEBUG: Menggunakan obat terakhir yang diprediksi dari session: {drug_name_for_comparison_raw}")
            
        if drug_name_for_comparison_raw is None:
            obat_csv_files_raw.sort() 
            drug_name_for_comparison_raw = obat_csv_files_raw[0]
            print(f"DEBUG: Kembali menggunakan obat pertama yang tersedia: {drug_name_for_comparison_raw}")
            
        drug_name_for_comparison_display = drug_name_for_comparison_raw.replace('_', ' ').title()
        filepath = os.path.join(DATASET_DIR, f"{drug_name_for_comparison_raw}.csv")

        if not os.path.exists(filepath):
            flash(f"File dataset untuk '{drug_name_for_comparison_display}' tidak ditemukan di server.", "danger")
            return redirect(url_for('home'))

        print(f"DEBUG: File CSV ditemukan di: {filepath}. Melanjutkan pemrosesan data untuk perbandingan.")
        
        window_size = 10 
        
        # Load data mentah dari CSV untuk mendapatkan df_all_common
        df_all_common = pd.read_csv(filepath)
        df_all_common.columns = df_all_common.columns.str.strip().str.lower()
        df_all_common.rename(columns={'tanggal': 'date', 'harga': 'price'}, inplace=True)
        df_all_common['date'] = pd.to_datetime(df_all_common['date'])
        df_all_common.set_index('date', inplace=True)
        df_all_common = df_all_common.sort_index()

        if df_all_common.empty or 'price' not in df_all_common.columns:
            flash(f"Data di '{drug_name_for_comparison_display}' kosong atau tidak valid setelah preprocessing.", "danger")
            return redirect(url_for('home'))


        # --- Muat Model dan Scaler untuk LSTM ---
        model_lstm, scaler_lstm = load_model_and_scaler_for_drug(drug_name_for_comparison_raw, 'LSTM')
        if model_lstm is None or scaler_lstm is None:
            flash(f"Model atau scaler LSTM untuk '{drug_name_for_comparison_display}' tidak ditemukan atau rusak.", "danger")
            return redirect(url_for('home'))

        # --- Muat Model dan Scaler untuk TCN ---
        model_tcn, scaler_tcn = load_model_and_scaler_for_drug(drug_name_for_comparison_raw, 'TCN')
        if model_tcn is None or scaler_tcn is None:
            flash(f"Model atau scaler TCN untuk '{drug_name_for_comparison_display}' tidak ditemukan atau rusak.", "danger")
            return redirect(url_for('home'))

        # --- Prediksi In-Sample Penuh untuk LSTM dan Hitung Metriknya ---
        # days_ahead=1 karena kita hanya butuh in-sample, bukan prediksi masa depan di sini
        _, _, lstm_in_sample_predictions_raw, mae_lstm, rmse_lstm, mape_lstm = future_lstm_func(df_all_common[['price']], model_lstm, scaler_lstm, days_ahead=1, window_size=window_size)
        
        # --- Prediksi In-Sample Penuh untuk TCN dan Hitung Metriknya ---
        _, _, tcn_in_sample_predictions_raw, mae_tcn, rmse_tcn, mape_tcn = future_tcn_func(df_all_common[['price']], model_tcn, scaler_tcn, days_ahead=1, window_size=window_size)

        # Data aktual historis penuh
        actual_historical_full_chart_data = df_all_common['price'].tolist()
        all_labels_for_chart = df_all_common.index.strftime('%Y-%m-%d').tolist()

        # Pad prediksi in-sample LSTM agar panjangnya sama dengan all_labels_for_chart
        lstm_in_sample_predictions_final = [None] * window_size + lstm_in_sample_predictions_raw.tolist() # Pastikan ini list
        if len(lstm_in_sample_predictions_final) < len(all_labels_for_chart):
            lstm_in_sample_predictions_final += [None] * (len(all_labels_for_chart) - len(lstm_in_sample_predictions_final))
        elif len(lstm_in_sample_predictions_final) > len(all_labels_for_chart):
            lstm_in_sample_predictions_final = lstm_in_sample_predictions_final[:len(all_labels_for_chart)]

        # Pad prediksi in-sample TCN agar panjangnya sama dengan all_labels_for_chart
        tcn_in_sample_predictions_final = [None] * window_size + tcn_in_sample_predictions_raw.tolist() # Pastikan ini list
        if len(tcn_in_sample_predictions_final) < len(all_labels_for_chart):
            tcn_in_sample_predictions_final += [None] * (len(all_labels_for_chart) - len(tcn_in_sample_predictions_final))
        elif len(tcn_in_sample_predictions_final) > len(all_labels_for_chart):
            tcn_in_sample_predictions_final = tcn_in_sample_predictions_final[:len(all_labels_for_chart)]


        lstm_metrics_data = (round(mae_lstm,2), round(rmse_lstm,2), round(mape_lstm,2))
        tcn_metrics_data = (round(mae_tcn,2), round(rmse_tcn,2), round(mape_tcn,2))


        print("\n--- DEBUGGING DATA UNTUK COMPARISON.HTML (FINAL PREP) ---")
        print(f"  nama_obat_display: {drug_name_for_comparison_display}")
        print(f"  all_labels_for_chart (len {len(all_labels_for_chart)}): {all_labels_for_chart[:5]} ... {all_labels_for_chart[-5:]}")
        print(f"  actual_historical_full_chart_data (len {len(actual_historical_full_chart_data)}): {actual_historical_full_chart_data[:5]} ... {actual_historical_full_chart_data[-5:]}")
        print(f"  lstm_in_sample_predictions_final (len {len(lstm_in_sample_predictions_final)}): {lstm_in_sample_predictions_final[:5]} ... {lstm_in_sample_predictions_final[-5:]}")
        print(f"  tcn_in_sample_predictions_final (len {len(tcn_in_sample_predictions_final)}): {tcn_in_sample_predictions_final[:5]} ... {tcn_in_sample_predictions_final[-5:]}")
        print(f"  LSTM Metrics: MAE={mae_lstm:.2f}, RMSE={rmse_lstm:.2f}, MAPE={mape_lstm:.2f}%")
        print(f"  TCN Metrics: MAE={mae_tcn:.2f}, RMSE={rmse_tcn:.2f}, MAPE={mape_tcn:.2f}%")
        print("--------------------------------------------------\n")

        return render_template('comparison.html',
                               nama_obat=drug_name_for_comparison_display,
                               all_labels=all_labels_for_chart, 
                               actual_historical_data_for_chart=actual_historical_full_chart_data, 
                               lstm_predicted_series=lstm_in_sample_predictions_final, # Sekarang ini adalah prediksi in-sample penuh
                               tcn_predicted_series=tcn_in_sample_predictions_final,   # Sekarang ini adalah prediksi in-sample penuh
                               lstm_metrics=lstm_metrics_data,
                               tcn_metrics=tcn_metrics_data)

    except Exception as e:
        print(f"--- DEBUG ERROR in /comparison route: {type(e).__name__}: {str(e)} ---")
        flash(f"Terjadi kesalahan pada perbandingan model: {str(e)}", 'danger')
        return redirect(url_for('home'))


@app.route('/future')
def future():
    obat_list = sorted([os.path.splitext(f)[0].replace('_', ' ').title() for f in os.listdir(DATASET_DIR) if f.endswith('.csv')])
    return render_template('future.html', obat_list=obat_list, max_date=None)

@app.route('/future/predict', methods=['POST'])
def future_predict():
    temp_path = None 
    try:
        model_type = request.form.get('model')
        n_days = int(request.form.get('n_days'))
        
        file = request.files.get('file')
        nama_obat_selected = request.form.get('nama_obat')

        if not model_type:
            flash("Tipe model tidak valid. Pilih 'LSTM' atau 'TCN'.", "danger")
            return redirect(url_for('future'))

        if file and file.filename != '':
            filename = secure_filename(file.filename)
            drug_name = os.path.splitext(filename)[0]
            temp_path = os.path.join(app.config['TEMP_FOLDER'], filename)
            file.save(temp_path)
            csv_source_path = temp_path
        elif nama_obat_selected:
            drug_name = nama_obat_selected.lower().replace(' ', '_')
            filename = f"{drug_name}.csv"
            csv_source_path = os.path.join(DATASET_DIR, filename)
            if not os.path.exists(csv_source_path):
                flash(f"File dataset untuk obat '{nama_obat_selected}' tidak ditemukan di server: {csv_source_path}.", "danger")
                return redirect(url_for('future'))
        else:
            flash('Tidak ada file yang diunggah atau obat dipilih. Harap unggah file CSV atau pilih obat dari daftar.', 'danger')
            return redirect(url_for('future'))

        model_future, scaler_future = load_model_and_scaler_for_drug(drug_name, model_type)

        if model_future is None or scaler_future is None:
            flash(f"Model atau scaler {model_type} untuk '{drug_name}' tidak ditemukan atau rusak. Harap pastikan model telah dilatih untuk obat ini.", "danger")
            return redirect(url_for('future'))

        print(f"\n--- DEBUGGING CSV READING (future_predict) ---")
        print(f"  Attempting to read file: {csv_source_path}")
        
        df = pd.read_csv(csv_source_path)
        print(f"  Original DataFrame head:\n{df.head()}")
        print(f"  Original DataFrame columns: {df.columns.tolist()}")

        df.columns = df.columns.str.strip().str.lower()
        print(f"  Columns after strip().lower(): {df.columns.tolist()}")

        # Robust column detection and renaming
        date_col_found = False
        price_col_found = False
        for col in ['tanggal', 'date']:
            if col in df.columns:
                df.rename(columns={col: 'date'}, inplace=True)
                date_col_found = True
                break
        for col in ['harga', 'price']:
            if col in df.columns:
                df.rename(columns={col: 'price'}, inplace=True)
                price_col_found = True
                break
        
        if not date_col_found or not price_col_found:
            raise ValueError("Kolom 'tanggal'/'date' dan 'harga'/'price' tidak ditemukan di CSV setelah normalisasi nama kolom. Pastikan kolom di CSV adalah 'Tanggal'/'Date' dan 'Harga'/'Price' (case-insensitive).")

        print(f"  Columns after rename: {df.columns.tolist()}")
        
        if 'price' not in df.columns:
            raise ValueError("Kolom 'price' tidak ditemukan setelah rename. Cek nama kolom asli di CSV.")

        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        print(f"  Number of NaN values in 'price' column: {df['price'].isnull().sum()}")

        df.dropna(subset=['price'], inplace=True)
        print(f"  DataFrame shape after dropping NaN prices: {df.shape}")

        if df.empty or df['price'].isnull().all():
            raise ValueError("DataFrame kosong atau semua nilai di kolom 'price' adalah NaN setelah pembersihan data. Tidak ada data untuk prediksi.")

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df.sort_index()
        nama_obat = drug_name.replace('_', ' ').title()
        
        print(f"  Final DataFrame head for prediction:\n{df.head()}")
        print(f"  Final DataFrame tail for prediction:\n{df.tail()}")
        print(f"--------------------------------------------------\n")

        window_size = 10 # Pastikan ini sesuai dengan model Anda

        # Panggil fungsi prediksi dari predict_future_multi.py
        # Sekarang mengembalikan: (predicted_future_prices, future_labels, in_sample_predictions, mae, rmse, mape)
        if model_type == 'LSTM':
            predicted_future_prices, future_labels, in_sample_predictions, mae_future, rmse_future, mape_future = future_lstm_func(df[['price']], model_future, scaler_future, days_ahead=n_days, window_size=window_size)
        elif model_type == 'TCN':
            predicted_future_prices, future_labels, in_sample_predictions, mae_future, rmse_future, mape_future = future_tcn_func(df[['price']], model_future, scaler_future, days_ahead=n_days, window_size=window_size)
        else:
            raise ValueError(f"Tipe model '{model_type}' tidak dikenal. Pilih 'LSTM' atau 'TCN'.")

        print(f"\n--- DEBUGGING AFTER PREDICTION FUNCTION CALL ---")
        print(f"  predicted_future_prices received (len {len(predicted_future_prices)}): {predicted_future_prices[:5]}")
        print(f"  future_labels received (len {len(future_labels)}): {future_labels[:5]}")
        print(f"  in_sample_predictions received (len {len(in_sample_predictions)}): {in_sample_predictions[:5]}")
        print(f"  MAE (future_predict): {mae_future:.2f}, RMSE (future_predict): {rmse_future:.2f}, MAPE (future_predict): {mape_future:.2f}%")
        print(f"--------------------------------------------------\n")

        if predicted_future_prices is None or future_labels is None or in_sample_predictions is None:
            raise ValueError("Fungsi prediksi (LSTM/TCN) mengembalikan None. Ada kesalahan internal dalam fungsi prediksi. Cek log predict_future_multi.py.")

        # --- Persiapan Data untuk Grafik di future_result.html ---
        # 1. Labels untuk seluruh grafik (historis + masa depan)
        all_chart_labels = df.index.strftime('%Y-%m-%d').tolist() + future_labels
        
        # 2. Data Aktual Historis (seluruhnya)
        actual_historical_data = df['price'].tolist()
        
        # 3. Prediksi In-Sample (untuk periode historis)
        # Ini perlu dipad dengan null di awal karena in_sample_predictions dimulai setelah window_size
        padded_in_sample_predictions = [None] * window_size + in_sample_predictions.tolist()
        
        # 4. Prediksi Masa Depan (untuk periode masa depan)
        # Ini perlu dipad dengan null di awal untuk menyesuaikan panjang historis
        padded_future_predictions = [None] * len(actual_historical_data) + predicted_future_prices.tolist()

        # Pastikan semua array memiliki panjang yang sama dengan all_chart_labels untuk plotting
        # Ini penting agar Chart.js bisa memplot garis dengan benar
        def pad_to_length(arr, target_len):
            if len(arr) < target_len:
                return arr + [None] * (target_len - len(arr))
            return arr[:target_len]

        actual_historical_data_padded = pad_to_length(actual_historical_data, len(all_chart_labels))
        padded_in_sample_predictions_final = pad_to_length(padded_in_sample_predictions, len(all_chart_labels))
        padded_future_predictions_final = pad_to_length(padded_future_predictions, len(all_chart_labels))

        print("\n--- DEBUGGING DATA FOR CHART (FINAL PREP) ---")
        print(f"  all_chart_labels (len {len(all_chart_labels)}): {all_chart_labels[:5]} ... {all_chart_labels[-5:]}")
        print(f"  actual_historical_data_padded (len {len(actual_historical_data_padded)}): {actual_historical_data_padded[:5]} ... {actual_historical_data_padded[-5:]}")
        print(f"  padded_in_sample_predictions_final (len {len(padded_in_sample_predictions_final)}): {padded_in_sample_predictions_final[:5]} ... {padded_in_sample_predictions_final[-5:]}")
        print(f"  padded_future_predictions_final (len {len(padded_future_predictions_final)}): {padded_future_predictions_final[:5]} ... {padded_future_predictions_final[-5:]}")
        print("--------------------------------------------------\n")

        return render_template('future_result.html',
                               nama_obat=nama_obat,
                               model=model_type,
                               labels=future_labels, # Label untuk tabel prediksi masa depan
                               predicted=predicted_future_prices.tolist(), # Data untuk tabel prediksi masa depan
                               
                               # Data untuk grafik komprehensif 3 garis
                               all_labels_for_future_chart=all_chart_labels,
                               actual_historical_data_for_chart=actual_historical_data_padded,
                               in_sample_predictions_for_chart=padded_in_sample_predictions_final,
                               future_predictions_for_chart=padded_future_predictions_final,
                               
                               history_len=len(df), # Panjang seluruh data historis
                               zip=zip
                               )
                               
    except ValueError as ve:
        flash(f"Kesalahan data: {str(ve)}", "danger")
        print(f"ERROR (ValueError) in future_predict route: {ve}")
        return redirect(url_for('future'))
    except Exception as e:
        flash(f"Terjadi kesalahan saat melakukan prediksi masa depan: {str(e)}", "danger")
        print(f"ERROR in future_predict route: {e}")
        return redirect(url_for('future'))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"DEBUG: File sementara '{temp_path}' berhasil dihapus.")

if __name__ == '__main__':
    app.run(debug=True)