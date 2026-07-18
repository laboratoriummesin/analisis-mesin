"""
Training model untuk data sensor mesin — versi lengkap dengan ML & Deep Learning.
Jalankan: python train_model.py

Mencakup:
1. Klasifikasi kondisi: RandomForest & MLPClassifier (neural network ringan)
2. Deteksi anomali: Isolation Forest & Autoencoder (deep learning)
3. Clustering pola operasi: K-Means
4. Explainability: SHAP (menjelaskan prediksi RandomForest)
5. Forecasting: ARIMA (statistik) & LSTM (deep learning)
"""

import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

from api_client import ambil_data_sensor, kirim_hasil_analisis, kirim_forecast

# Kunci semua sumber keacakan supaya hasil training konsisten tiap dijalankan ulang
import random
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

MODEL_KLASIFIKASI_PATH = "model_klasifikasi_kondisi.pkl"
MODEL_MLP_PATH = "model_mlp_klasifikasi.pkl"
MODEL_ANOMALI_PATH = "model_anomali.pkl"
MODEL_AUTOENCODER_PATH = "model_autoencoder.keras"
MODEL_LSTM_PATH = "model_lstm_forecast.keras"


# =========================================================================
# 1. RandomForest — klasifikasi kondisi
# =========================================================================
def latih_klasifikasi_rf(df):
    print("\n=== RandomForest: Klasifikasi Kondisi ===")
    X = df[["suhu", "kecepatan_getaran"]]
    y = df["kondisi"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))

    joblib.dump(model, MODEL_KLASIFIKASI_PATH)
    return model


# =========================================================================
# 2. MLPClassifier — neural network ringan, pembanding RandomForest
# =========================================================================
def latih_klasifikasi_mlp(df):
    print("\n=== MLPClassifier: Klasifikasi Kondisi (Neural Network) ===")
    X = df[["suhu", "kecepatan_getaran"]]
    y = df["kondisi"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    model = MLPClassifier(
        hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42
    )
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))

    joblib.dump({"model": model, "scaler": scaler}, MODEL_MLP_PATH)
    return model, scaler


# =========================================================================
# 3. Isolation Forest — deteksi anomali (statistik/ML klasik)
# =========================================================================
def latih_isolation_forest(df):
    print("\n=== Isolation Forest: Deteksi Anomali ===")
    X = df[["suhu", "kecepatan_getaran"]]

    model = IsolationForest(contamination=0.02, random_state=42)
    model.fit(X)

    skor = model.decision_function(X)
    label = model.predict(X)

    df_hasil = df.copy()
    df_hasil["skor_anomali_if"] = skor
    df_hasil["anomali_if"] = label == -1

    print(f"Ditemukan {df_hasil['anomali_if'].sum()} anomali (Isolation Forest)")
    joblib.dump(model, MODEL_ANOMALI_PATH)
    return df_hasil


# =========================================================================
# 4. Autoencoder — deteksi anomali (deep learning)
# =========================================================================
def latih_autoencoder(df):
    print("\n=== Autoencoder: Deteksi Anomali (Deep Learning) ===")
    import tensorflow as tf
    from tensorflow import keras

    X = df[["suhu", "kecepatan_getaran"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Arsitektur sederhana: 2 fitur -> 4 -> 2 (bottleneck) -> 4 -> 2 fitur
    autoencoder = keras.Sequential([
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(4, activation="relu"),
        keras.layers.Dense(2, activation="relu"),  # bottleneck
        keras.layers.Dense(4, activation="relu"),
        keras.layers.Dense(2, activation="linear"),
    ])
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(X_scaled, X_scaled, epochs=30, batch_size=32, verbose=0)

    rekonstruksi = autoencoder.predict(X_scaled, verbose=0)
    error_rekonstruksi = np.mean(np.square(X_scaled - rekonstruksi), axis=1)

    # Ambang batas anomali: rata-rata error + 3 standar deviasi
    ambang = error_rekonstruksi.mean() + 3 * error_rekonstruksi.std()
    label_anomali = error_rekonstruksi > ambang

    df_hasil = df.copy()
    df_hasil["skor_anomali_ae"] = error_rekonstruksi
    df_hasil["anomali_ae"] = label_anomali

    print(f"Ditemukan {label_anomali.sum()} anomali (Autoencoder), ambang batas = {ambang:.4f}")
    autoencoder.save(MODEL_AUTOENCODER_PATH)
    return df_hasil


# =========================================================================
# 5. K-Means — clustering pola operasi
# =========================================================================
def latih_clustering(df, jumlah_cluster=3):
    print(f"\n=== K-Means: Clustering Pola Operasi ({jumlah_cluster} cluster) ===")
    X = df[["suhu", "kecepatan_getaran"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=jumlah_cluster, random_state=42, n_init=10)
    label_cluster = model.fit_predict(X_scaled)

    df_hasil = df.copy()
    df_hasil["cluster"] = label_cluster

    print("Jumlah data per cluster:")
    print(df_hasil["cluster"].value_counts())
    return df_hasil, model


# =========================================================================
# 6. SHAP — explainability untuk RandomForest
# =========================================================================
def hitung_shap(df, model_rf):
    print("\n=== SHAP: Menjelaskan Prediksi RandomForest ===")
    X = df[["suhu", "kecepatan_getaran"]]

    # Ambil sampel biar cepat (SHAP lumayan berat kalau data besar)
    sampel = X.sample(min(len(X), 500), random_state=42)

    explainer = shap.TreeExplainer(model_rf)
    shap_values = explainer.shap_values(sampel)

    # shap_values bisa berbentuk list (per kelas) tergantung versi shap;
    # kita rata-ratakan besarnya (absolute) di semua kelas untuk dapat importance global
    if isinstance(shap_values, list):
        rata_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
    else:
        rata_abs = np.abs(shap_values).mean(axis=(0, 2)) if shap_values.ndim == 3 else np.abs(shap_values).mean(axis=0)

    hasil = {"suhu": float(rata_abs[0]), "kecepatan_getaran": float(rata_abs[1])}
    print(f"Rata-rata pengaruh (SHAP): {hasil}")
    return hasil


# =========================================================================
# 7. ARIMA — forecasting statistik
# =========================================================================
def buat_forecast_arima(df, jam_ke_depan=24):
    print(f"\n=== ARIMA: Forecasting {jam_ke_depan} jam ke depan ===")
    df_ts = df.set_index("created_at").sort_index()

    hasil_forecast = []
    waktu_terakhir = df_ts.index.max()

    for kolom in ["suhu", "kecepatan_getaran"]:
        seri = df_ts[kolom].resample("1h").mean().interpolate()
        if len(seri) < 10:
            print(f"Data {kolom} terlalu sedikit untuk ARIMA, dilewati.")
            continue

        try:
            model = ARIMA(seri, order=(2, 1, 2))
            hasil_fit = model.fit()
            prediksi = hasil_fit.forecast(steps=jam_ke_depan)
        except Exception as e:
            print(f"ARIMA gagal untuk {kolom}: {e}")
            continue

        for i, nilai in enumerate(prediksi):
            waktu_target = waktu_terakhir + pd.Timedelta(hours=i + 1)
            hasil_forecast.append({
                "target_waktu": waktu_target.strftime("%Y-%m-%d %H:%M:%S"),
                "nilai_suhu_prediksi": float(nilai) if kolom == "suhu" else None,
                "nilai_getaran_prediksi": float(nilai) if kolom == "kecepatan_getaran" else None,
                "sumber": "arima_forecast_v1",
            })

    return hasil_forecast


# =========================================================================
# 8. LSTM — forecasting deep learning
# =========================================================================
def buat_forecast_lstm(df, langkah_ke_depan=24, jendela=10):
    print(f"\n=== LSTM: Forecasting {langkah_ke_depan} langkah ke depan (Deep Learning) ===")
    from tensorflow import keras

    df_ts = df.set_index("created_at").sort_index()
    hasil_forecast = []
    waktu_terakhir = df_ts.index.max()

    for kolom in ["suhu", "kecepatan_getaran"]:
        seri = df_ts[kolom].resample("1h").mean().interpolate().dropna()
        if len(seri) < jendela + 10:
            print(f"Data {kolom} terlalu sedikit untuk LSTM, dilewati.")
            continue

        nilai = seri.values.reshape(-1, 1)
        scaler = StandardScaler()
        nilai_scaled = scaler.fit_transform(nilai).flatten()

        # Bentuk data jadi urutan (sequence) untuk LSTM
        X, y = [], []
        for i in range(len(nilai_scaled) - jendela):
            X.append(nilai_scaled[i:i + jendela])
            y.append(nilai_scaled[i + jendela])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = keras.Sequential([
            keras.layers.Input(shape=(jendela, 1)),
            keras.layers.LSTM(16, activation="tanh"),
            keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=20, batch_size=16, verbose=0)

        # Forecast berulang (rolling forecast): prediksi 1 langkah, masukkan lagi ke input
        urutan_sekarang = nilai_scaled[-jendela:].tolist()
        prediksi_scaled = []
        for _ in range(langkah_ke_depan):
            X_input = np.array(urutan_sekarang[-jendela:]).reshape((1, jendela, 1))
            pred = model.predict(X_input, verbose=0)[0][0]
            prediksi_scaled.append(pred)
            urutan_sekarang.append(pred)

        prediksi_asli = scaler.inverse_transform(
            np.array(prediksi_scaled).reshape(-1, 1)
        ).flatten()

        for i, nilai_pred in enumerate(prediksi_asli):
            waktu_target = waktu_terakhir + pd.Timedelta(hours=i + 1)
            hasil_forecast.append({
                "target_waktu": waktu_target.strftime("%Y-%m-%d %H:%M:%S"),
                "nilai_suhu_prediksi": float(nilai_pred) if kolom == "suhu" else None,
                "nilai_getaran_prediksi": float(nilai_pred) if kolom == "kecepatan_getaran" else None,
                "sumber": "lstm_forecast_v1",
            })

        model.save(f"{MODEL_LSTM_PATH.replace('.keras', '')}_{kolom}.keras")

    return hasil_forecast


# =========================================================================
# MAIN
# =========================================================================
def main():
    print("Mengambil data dari API...")
    df = ambil_data_sensor(limit=5000)
    print(f"Total data untuk training: {len(df)} baris")

    if len(df) < 50:
        print("Data terlalu sedikit untuk training yang bermakna. Berhenti.")
        return

    baris_terakhir = df.iloc[[-1]]
    id_terakhir = int(baris_terakhir.iloc[0]["id"])

    # ---------- 1 & 2. Klasifikasi ----------
    model_rf = latih_klasifikasi_rf(df)
    model_mlp, scaler_mlp = latih_klasifikasi_mlp(df)

    pred_rf = model_rf.predict(baris_terakhir[["suhu", "kecepatan_getaran"]])[0]
    kirim_hasil_analisis(
        data_id=id_terakhir, prediksi_kondisi=pred_rf,
        sumber="random_forest_klasifikasi_v1", keterangan="Prediksi kondisi terbaru (RandomForest)",
    )

    X_terakhir_scaled = scaler_mlp.transform(baris_terakhir[["suhu", "kecepatan_getaran"]])
    pred_mlp = model_mlp.predict(X_terakhir_scaled)[0]
    kirim_hasil_analisis(
        data_id=id_terakhir, prediksi_kondisi=pred_mlp,
        sumber="mlp_klasifikasi_v1", keterangan="Prediksi kondisi terbaru (Neural Network/MLP)",
    )

    # ---------- 3 & 4. Deteksi anomali ----------
    df_if = latih_isolation_forest(df)
    for _, row in df_if[df_if["anomali_if"]].iterrows():
        kirim_hasil_analisis(
            data_id=int(row["id"]), skor_anomali=float(row["skor_anomali_if"]),
            sumber="isolation_forest_v1", keterangan="Anomali terdeteksi (Isolation Forest)",
        )

    df_ae = latih_autoencoder(df)
    for _, row in df_ae[df_ae["anomali_ae"]].iterrows():
        kirim_hasil_analisis(
            data_id=int(row["id"]), skor_anomali=float(row["skor_anomali_ae"]),
            sumber="autoencoder_v1", keterangan="Anomali terdeteksi (Autoencoder/Deep Learning)",
        )

    # ---------- 5. Clustering ----------
    df_cluster, _ = latih_clustering(df, jumlah_cluster=3)
    # Kirim label cluster untuk 100 data terbaru saja (biar tidak terlalu banyak request)
    for _, row in df_cluster.tail(100).iterrows():
        kirim_hasil_analisis(
            data_id=int(row["id"]), sumber="kmeans_cluster_v1",
            keterangan=f"Cluster {int(row['cluster'])}",
        )

    # ---------- 6. SHAP ----------
    hasil_shap = hitung_shap(df, model_rf)
    import json
    kirim_hasil_analisis(
        data_id=id_terakhir, sumber="shap_importance_v1",
        keterangan=json.dumps(hasil_shap),
    )

    # ---------- 7. ARIMA Forecasting ----------
    forecast_arima = buat_forecast_arima(df, jam_ke_depan=24)
    if forecast_arima:
        kirim_forecast(forecast_arima)
        print(f"Terkirim {len(forecast_arima)} titik forecast ARIMA")

    # ---------- 8. LSTM Forecasting ----------
    forecast_lstm = buat_forecast_lstm(df, langkah_ke_depan=24)
    if forecast_lstm:
        kirim_forecast(forecast_lstm)
        print(f"Terkirim {len(forecast_lstm)} titik forecast LSTM")

    print("\n=== Selesai semua training & pengiriman hasil ===")


if __name__ == "__main__":
    main()
