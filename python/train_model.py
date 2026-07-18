import os
import random
import json

import numpy as np
import pandas as pd
import joblib
import shap
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

from api_client import ambil_data_sensor, kirim_hasil_analisis, kirim_forecast

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

DAFTAR_MESIN = [1, 2, 3, 4, 5]


def latih_klasifikasi_rf(df, mesin_id):
    print(f"\n=== [Mesin {mesin_id}] RandomForest: Klasifikasi Kondisi ===")
    X = df[["suhu", "kecepatan_getaran"]]
    y = df["kondisi"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))

    joblib.dump(model, f"model_klasifikasi_kondisi_mesin{mesin_id}.pkl")
    return model


def latih_klasifikasi_mlp(df, mesin_id):
    print(f"\n=== [Mesin {mesin_id}] MLPClassifier: Klasifikasi Kondisi (Neural Network) ===")
    X = df[["suhu", "kecepatan_getaran"]]
    y = df["kondisi"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))

    joblib.dump({"model": model, "scaler": scaler}, f"model_mlp_klasifikasi_mesin{mesin_id}.pkl")
    return model, scaler


def latih_isolation_forest(df, mesin_id):
    print(f"\n=== [Mesin {mesin_id}] Isolation Forest: Deteksi Anomali ===")
    X = df[["suhu", "kecepatan_getaran"]]

    model = IsolationForest(contamination=0.02, random_state=42)
    model.fit(X)

    skor = model.decision_function(X)
    label = model.predict(X)

    df_hasil = df.copy()
    df_hasil["skor_anomali_if"] = skor
    df_hasil["anomali_if"] = label == -1

    print(f"Ditemukan {df_hasil['anomali_if'].sum()} anomali (Isolation Forest)")
    joblib.dump(model, f"model_anomali_mesin{mesin_id}.pkl")
    return df_hasil


def latih_autoencoder(df, mesin_id):
    print(f"\n=== [Mesin {mesin_id}] Autoencoder: Deteksi Anomali (Deep Learning) ===")
    from tensorflow import keras

    X = df[["suhu", "kecepatan_getaran"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    autoencoder = keras.Sequential([
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(4, activation="relu"),
        keras.layers.Dense(2, activation="relu"),
        keras.layers.Dense(4, activation="relu"),
        keras.layers.Dense(2, activation="linear"),
    ])
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(X_scaled, X_scaled, epochs=30, batch_size=32, verbose=0)

    rekonstruksi = autoencoder.predict(X_scaled, verbose=0)
    error_rekonstruksi = np.mean(np.square(X_scaled - rekonstruksi), axis=1)

    ambang = error_rekonstruksi.mean() + 3 * error_rekonstruksi.std()
    label_anomali = error_rekonstruksi > ambang

    df_hasil = df.copy()
    df_hasil["skor_anomali_ae"] = error_rekonstruksi
    df_hasil["anomali_ae"] = label_anomali

    print(f"Ditemukan {label_anomali.sum()} anomali (Autoencoder), ambang batas = {ambang:.4f}")
    autoencoder.save(f"model_autoencoder_mesin{mesin_id}.keras")
    return df_hasil


def latih_clustering(df, mesin_id, jumlah_cluster=3):
    print(f"\n=== [Mesin {mesin_id}] K-Means: Clustering Pola Operasi ({jumlah_cluster} cluster) ===")
    X = df[["suhu", "kecepatan_getaran"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=jumlah_cluster, random_state=42, n_init=10)
    label_cluster = model.fit_predict(X_scaled)

    df_hasil = df.copy()
    df_hasil["cluster"] = label_cluster
    print(df_hasil["cluster"].value_counts())
    return df_hasil, model


def hitung_shap(df, model_rf):
    print("=== SHAP: Menjelaskan Prediksi RandomForest ===")
    X = df[["suhu", "kecepatan_getaran"]]
    sampel = X.sample(min(len(X), 500), random_state=42)

    explainer = shap.TreeExplainer(model_rf)
    shap_values = explainer.shap_values(sampel)

    if isinstance(shap_values, list):
        rata_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
    else:
        rata_abs = np.abs(shap_values).mean(axis=(0, 2)) if shap_values.ndim == 3 else np.abs(shap_values).mean(axis=0)

    hasil = {"suhu": float(rata_abs[0]), "kecepatan_getaran": float(rata_abs[1])}
    print(f"Rata-rata pengaruh (SHAP): {hasil}")
    return hasil


def buat_forecast_arima(df, mesin_id, jam_ke_depan=24):
    print(f"\n=== [Mesin {mesin_id}] ARIMA: Forecasting {jam_ke_depan} jam ke depan ===")
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


def buat_forecast_lstm(df, mesin_id, langkah_ke_depan=24, jendela=10):
    print(f"\n=== [Mesin {mesin_id}] LSTM: Forecasting {langkah_ke_depan} langkah ke depan (Deep Learning) ===")
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

        model.save(f"model_lstm_forecast_{kolom}_mesin{mesin_id}.keras")

    return hasil_forecast


def proses_satu_mesin(mesin_id):
    print(f"\n{'=' * 60}\nMemproses Mesin Bubut {mesin_id}\n{'=' * 60}")
    df = ambil_data_sensor(limit=5000, mesin_id=mesin_id)
    print(f"Total data mesin {mesin_id}: {len(df)} baris")

    if len(df) < 50:
        print(f"Data mesin {mesin_id} terlalu sedikit untuk training bermakna. Dilewati.")
        return

    baris_terakhir = df.iloc[[-1]]
    id_terakhir = int(baris_terakhir.iloc[0]["id"])

    model_rf = latih_klasifikasi_rf(df, mesin_id)
    model_mlp, scaler_mlp = latih_klasifikasi_mlp(df, mesin_id)

    pred_rf = model_rf.predict(baris_terakhir[["suhu", "kecepatan_getaran"]])[0]
    kirim_hasil_analisis(
        data_id=id_terakhir, prediksi_kondisi=pred_rf,
        sumber="random_forest_klasifikasi_v1", keterangan="Prediksi kondisi terbaru (RandomForest)",
        mesin_id=mesin_id,
    )

    X_terakhir_scaled = scaler_mlp.transform(baris_terakhir[["suhu", "kecepatan_getaran"]])
    pred_mlp = model_mlp.predict(X_terakhir_scaled)[0]
    kirim_hasil_analisis(
        data_id=id_terakhir, prediksi_kondisi=pred_mlp,
        sumber="mlp_klasifikasi_v1", keterangan="Prediksi kondisi terbaru (Neural Network/MLP)",
        mesin_id=mesin_id,
    )

    df_if = latih_isolation_forest(df, mesin_id)
    for _, row in df_if[df_if["anomali_if"]].iterrows():
        kirim_hasil_analisis(
            data_id=int(row["id"]), skor_anomali=float(row["skor_anomali_if"]),
            sumber="isolation_forest_v1", keterangan="Anomali terdeteksi (Isolation Forest)",
            mesin_id=mesin_id,
        )

    df_ae = latih_autoencoder(df, mesin_id)
    for _, row in df_ae[df_ae["anomali_ae"]].iterrows():
        kirim_hasil_analisis(
            data_id=int(row["id"]), skor_anomali=float(row["skor_anomali_ae"]),
            sumber="autoencoder_v1", keterangan="Anomali terdeteksi (Autoencoder/Deep Learning)",
            mesin_id=mesin_id,
        )

    df_cluster, _ = latih_clustering(df, mesin_id, jumlah_cluster=3)
    for _, row in df_cluster.tail(100).iterrows():
        kirim_hasil_analisis(
            data_id=int(row["id"]), sumber="kmeans_cluster_v1",
            keterangan=f"Cluster {int(row['cluster'])}",
            mesin_id=mesin_id,
        )

    hasil_shap = hitung_shap(df, model_rf)
    kirim_hasil_analisis(
        data_id=id_terakhir, sumber="shap_importance_v1",
        keterangan=json.dumps(hasil_shap),
        mesin_id=mesin_id,
    )

    forecast_arima = buat_forecast_arima(df, mesin_id, jam_ke_depan=24)
    if forecast_arima:
        kirim_forecast(forecast_arima, mesin_id=mesin_id)
        print(f"Terkirim {len(forecast_arima)} titik forecast ARIMA (mesin {mesin_id})")

    forecast_lstm = buat_forecast_lstm(df, mesin_id, langkah_ke_depan=24)
    if forecast_lstm:
        kirim_forecast(forecast_lstm, mesin_id=mesin_id)
        print(f"Terkirim {len(forecast_lstm)} titik forecast LSTM (mesin {mesin_id})")


def main():
    for mesin_id in DAFTAR_MESIN:
        try:
            proses_satu_mesin(mesin_id)
        except Exception as e:
            # Penting: kalau 1 mesin gagal, mesin lain tetap lanjut diproses,
            # tidak seperti sebelumnya di mana 1 error menghentikan SEMUANYA.
            print(f"\n⚠️ Gagal memproses Mesin {mesin_id}: {e}")
            continue

    print("\n=== Selesai training & pengiriman hasil untuk semua mesin ===")


if __name__ == "__main__":
    main()
