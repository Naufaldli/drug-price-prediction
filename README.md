# 🧪 Drug Price Forecasting using LSTM and TCN  
**Predicting pharmaceutical price trends using deep learning models (LSTM & TCN) with a Flask-based web interface.**

---

## 📌 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Technical Architecture](#technical-architecture)
- [Dataset Description](#dataset-description)
- [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)
- [Model Architecture](#model-architecture)
- [Experiment Setup](#experiment-setup)
- [Evaluation & Results](#evaluation--results)
- [How to Run This Project](#how-to-run-this-project)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [License & Acknowledgement](#license--acknowledgement)

---

# 🧭 Overview

This project implements two deep learning models — **LSTM** and **TCN** — to predict drug price trends based on historical pharmaceutical data.  
The goal is to provide an intelligent forecasting system that assists hospitals in planning budgets and inventory.

A simple **Flask web application** is included for interactive forecasting and visualization.

---

# 🌟 Key Features

### 🔹 Dual Deep Learning Models  
- LSTM (Long Short-Term Memory)  
- TCN (Temporal Convolutional Network)

### 🔹 Real Hospital Dataset  
- Historical drug price data from RSI PKU Muhammadiyah Maluku Utara  
- 10 types of commonly used drugs  
- Data from 2023–2025  

### 🔹 Complete ML Pipeline  
- Preprocessing  
- Scaling & windowing  
- Training  
- Evaluation  
- Forecast generation  
- Web app integration  

### 🔹 Comparative Performance Analysis  
- MAE, RMSE, MAPE  
- Model winner for each drug  

### 🔹 Flask Web Application  
- Interactive drug selection  
- Prediction visualization  
- Simple and accessible UI  

---

# 🏗 Technical Architecture

```
Raw Data → Preprocessing → Model Training (LSTM & TCN)
             ↓                      ↓
        Scaled Sequences      Saved Models (.pkl)
             ↓                      ↓
         Forecasting Engine ← Flask Web App
```

### **Components**
- **Preprocessing layer** — cleaning, scaling, windowing  
- **Model layer** — LSTM & TCN  
- **Evaluation layer** — performance metrics  
- **Forecasting engine** — 10-day prediction  
- **Web UI** — Flask for visualization  

---

# 📊 Dataset Description

- File: `harga_obat.csv`  
- Contains price history for 10 drugs  
- Fields typically include:
  - date  
  - drug_name  
  - price  

Dataset characteristics:
- Daily or periodic time series  
- Some drugs show smooth price trends  
- Others exhibit sudden fluctuations  

---

# 🧹 Preprocessing & Feature Engineering

### Steps
1. Load CSV  
2. Handle missing values  
3. Sort by date  
4. Scale values using **MinMaxScaler**  
5. Generate sliding windows:
   - Example: 30 timesteps → predict t+1  

### Output  
- X_train, X_test  
- y_train, y_test  
- Saved scaler for inverse transform  

---

# 🧠 Model Architecture

## 1. LSTM

```
Input (30 timesteps)
      ↓
LSTM Layer (32–64 units)
      ↓
Dropout (0.2–0.3)
      ↓
Dense (1)
```

Strengths:
- Great for long-term dependencies  
- Works well on smooth price trends  

---

## 2. TCN

```
Input
      ↓
Dilated Conv1D (kernel 3–5)
      ↓
Residual Block
      ↓
GlobalAveragePooling
      ↓
Dense (1)
```

Strengths:
- Fast  
- Stable  
- Excellent at capturing sudden price shifts  

---

# 🧪 Experiment Setup

### Environment
- Python 3.10+  
- TensorFlow / Keras  
- Scikit-learn  
- Numpy, Pandas  
- Flask  

### Training Settings
| Parameter | Value |
|----------|--------|
| Train/Test Split | 80/20 |
| Loss | MAE, MSE |
| Optimizer | Adam |
| Epochs | 50–100 |
| Batch size | 16–32 |

Both models were trained **per drug**, using identical train/test splits and preprocessing.

---

# 📊 Evaluation & Results

## MAE Comparison

| No | Drug Name     | LSTM MAE | TCN MAE | Best Model |
|----|----------------|----------|---------|------------|
| 1  | Amlodipin      | **15.65** | 23.35   | **LSTM**   |
| 2  | Amoxicillin    | **14.74** | 17.89   | **LSTM**   |
| 3  | Aspirine       | **42.25** | 52.15   | **LSTM**   |
| 4  | Cetirizine     | 131.42    | **17.67** | **TCN** |
| 5  | Ibuprofen      | 46.20     | **16.54** | **TCN** |
| 6  | Metformin      | **138.50** | 661.22 | **LSTM** |
| 7  | Omeprazole     | **237.66** | 1295.28 | **LSTM** |
| 8  | Paracetamol    | 91.55     | **18.47** | **TCN** |
| 9  | Ranitidine     | 5.74      | **2.77** | **TCN** |
| 10 | Simvastatin    | 20.63     | **3.61** | **TCN** |

### Summary
- **LSTM wins on 6 drugs**  
- **TCN wins on 4 drugs** (mostly highly fluctuating ones)

### Insight
- LSTM → best for stable trends  
- TCN → best for unpredictable price shifts  

---

# 🚀 How to Run This Project

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/drug-price-prediction-lstm-tcn.git
cd drug-price-prediction-lstm-tcn
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Preprocess Data

```bash
python preprocessing.py
```

## 4. Train Models

```bash
python LSTM.py
python TCN.py
```

## 5. Generate Forecast

```bash
python forecasting.py
```

## 6. Run Flask Web App

```bash
python app.py
```

Open browser:

```
http://127.0.0.1:5000
```

---

# 🗂 Project Structure

```
Drug Forecast/
│
├── data/
│   └── harga_obat.csv
│
├── models/
│   ├── LSTM_model.pkl
│   ├── TCN_model.pkl
│   └── ...
│
├── static/
│   ├── styles/style.css
│   └── images/ikon.png
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── app.py
├── LSTM.py
├── TCN.py
├── preprocessing.py
├── forecasting.py
├── requirements.txt
└── README.md
```

---

# 🚀 Future Improvements

### ✔ Advanced Models  
Transformer, N-BEATS, TFT, BiLSTM  

### ✔ Hyperparameter Optimization  
Optuna / Keras Tuner  

### ✔ Cloud Deployment  
AWS / GCP / Azure  

### ✔ Full MLOps Pipeline  
MLflow, DVC, Kubeflow  

### ✔ Dockerization  
Portable & scalable deployment  

### ✔ Improved Dashboard  
Interactive charts, forecasts, anomalies  

### ✔ Multi-Hospital Expansion  
Global models with local fine-tuning  

---

# 📜 License & Acknowledgement

## License  
Released under the **MIT License**.

## Acknowledgement  
This project was developed as part of the undergraduate thesis:  
**“Perbandingan Hasil Prediksi Harga Obat Menggunakan Algoritma LSTM dan TCN”**

Special thanks to:
- Program Studi Teknik Informatika, Universitas Muhammadiyah Maluku Utara  
- Dosen pembimbing dan penguji  
- RSI PKU Muhammadiyah Maluku Utara  
- Everyone who supported this research  

## Author  
**Naufal Adli, S.Kom**  
GitHub: https://github.com/Naufaldli
Email: naufaladli2019@gmail.com
