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

from api_client import ambil_data_sensor, kirim_hasil_analisis, kirim_forecast, hapus_forecast_lama
from pembersihan import bersihkan_data

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

DAFTAR_MESIN = [1, 2, 3, 4, 5]

BATAS_SUHU_MIN = 27
BATAS_SUHU_MAX = 150
GETARAN_BOLEH_NEGATIF = False

# Berapa hari ke belakang yang mau dibuatkan forecast "backtest" per-harinya
# (khusus forecast resolusi 1 jam / horizon panjang: 6/12/24 jam).
# Semakin besar angka ini, semakin banyak tanggal yang bisa dipilih di dashboard,
# tapi proses training juga akan semakin lama (karena ARIMA/LSTM di-fit ulang
# untuk tiap hari). Silakan turunkan angka ini kalau proses training terasa lambat.
JUMLAH_HARI_RIWAYAT_FORECAST = 14

# --- Konfigurasi forecast horizon PENDEK (30 menit / 1 jam / 2 jam) ---
# Dibuat terpisah dari forecast per-jam di atas karena butuh resolusi lebih halus
# (15 menit). Tidak ada backtest historis untuk horizon pendek ini karena
# 1) horizon-nya sangat singkat sehingga nilai edukatifnya kecil, dan
# 2) retraining LSTM per-hari dengan resolusi 15 menit jauh lebih berat.
RESOLUSI_PENDEK = "15min"
LANGKAH_PENDEK = 8  # 8 x 15 menit = 2 jam ke depan (cukup untuk opsi 30m/1j/2j)
JUMLAH_HARI_DATA_PENDEK = 3  # ambil 3 hari terakhir saja sebagai basis training resolusi tinggi

# Berapa banyak baris cluster terakhir yang dikirim ke DB. None = kirim SEMUA baris,
# supaya jumlah data yang ditampilkan di dashboard (card "Clustering Pola Operasi")
# selalu sama persis dengan jumlah data mentah yang diambil, bukan cuma 100 baris
# terakhir seperti sebelumnya.
MAKS_BARIS_KIRIM_CLUSTER = None


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
    try:
        X = df[["suhu", "kecepatan_getaran"]]
        sampel = X.sample(min(len(X), 500), random_state=42)

        explainer = shap.TreeExplainer(model_rf)
        shap_values = explainer.shap_values(sampel)

        if isinstance(shap_values, list):
            array_gabungan = np.stack(shap_values, axis=0)
            rata_abs = np.abs(array_gabungan).mean(axis=(0, 1))
        else:
            arr = np.array(shap_values)
            if arr.ndim == 3:
                rata_abs = np.abs(arr).mean(axis=(0, 2))
            else:
                rata_abs = np.abs(arr).mean(axis=0)

        hasil = {"suhu": float(rata_abs[0]), "kecepatan_getaran": float(rata_abs[1])}
        print(f"Rata-rata pengaruh (SHAP): {hasil}")
        return hasil

    except Exception as e:
        print(f"⚠️ SHAP gagal dihitung: {e} — dilewati, lanjut ke langkah berikutnya.")
        return None


def _buat_entry_forecast(waktu_target, kolom, nilai, sumber):
    entry = {
        "target_waktu": waktu_target.strftime("%Y-%m-%d %H:%M:%S"),
        "nilai_suhu_prediksi": None,
        "nilai_getaran_prediksi": None,
        "sumber": sumber,
    }
    if kolom == "suhu":
        entry["nilai_suhu_prediksi"] = float(nilai)
    else:
        entry["nilai_getaran_prediksi"] = float(nilai)
    return entry


def buat_forecast_arima(df, mesin_id, jam_ke_depan=24, jumlah_hari_riwayat=JUMLAH_HARI_RIWAYAT_FORECAST):
    """
    Membuat forecast ARIMA resolusi 1 jam dengan dua bagian:
    1. Forecast masa depan: dari titik data terakhir, seperti sebelumnya.
    2. Backtest per-hari: untuk `jumlah_hari_riwayat` hari terakhir (selain hari ini),
       model di-fit HANYA memakai data sebelum hari itu, lalu memprediksi 24 jam
       hari itu saja. Tidak digabung lintas hari — tiap hari punya forecast sendiri.

    Dipakai untuk opsi horizon "panjang" di dashboard: 6 / 12 / 24 jam.
    """
    print(f"\n=== [Mesin {mesin_id}] ARIMA: Forecasting per hari ({jam_ke_depan} jam/hari) ===")

    if df.empty:
        print(f"⚠️ Data kosong untuk mesin {mesin_id}")
        return []

    df_ts = df.set_index("created_at").sort_index()

    if df_ts.index.max() is pd.NaT:
        print(f"⚠️ Tidak ada timestamp valid untuk mesin {mesin_id}")
        return []

    waktu_terakhir = df_ts.index.max()
    hasil_forecast = []

    for kolom in ["suhu", "kecepatan_getaran"]:
        try:
            seri_full = df_ts[kolom].resample("1h").mean().interpolate()

            if len(seri_full) < 10:
                print(f"⚠️ Data {kolom} hanya {len(seri_full)} titik, minimum 10. ARIMA dilewati.")
                continue

            # --- 1. Forecast masa depan (dari titik data terakhir) ---
            try:
                model = ARIMA(seri_full, order=(2, 1, 2))
                hasil_fit = model.fit()
                prediksi = hasil_fit.forecast(steps=jam_ke_depan)
                for i, nilai in enumerate(prediksi):
                    waktu_target = waktu_terakhir + pd.Timedelta(hours=i + 1)
                    hasil_forecast.append(
                        _buat_entry_forecast(waktu_target, kolom, nilai, "arima_forecast_v1")
                    )
            except Exception as e:
                print(f"⚠️ ARIMA (forecast masa depan) gagal untuk {kolom}: {e}")

            # --- 2. Backtest per hari (tidak termasuk hari ini) ---
            tanggal_tersedia = sorted(set(seri_full.index.date))
            tanggal_hari_ini = waktu_terakhir.date()
            tanggal_backtest = [t for t in tanggal_tersedia if t < tanggal_hari_ini][-jumlah_hari_riwayat:]

            jumlah_sukses = 0
            for tanggal_target in tanggal_backtest:
                batas_cutoff = pd.Timestamp(tanggal_target)
                seri_latih = seri_full[seri_full.index < batas_cutoff]

                if len(seri_latih) < 10:
                    continue

                try:
                    model_bt = ARIMA(seri_latih, order=(2, 1, 2))
                    fit_bt = model_bt.fit()
                    prediksi_bt = fit_bt.forecast(steps=jam_ke_depan)
                except Exception as e:
                    print(f"⚠️ ARIMA (backtest {tanggal_target}) gagal untuk {kolom}: {e}")
                    continue

                for i, nilai in enumerate(prediksi_bt):
                    waktu_target = batas_cutoff + pd.Timedelta(hours=i)
                    hasil_forecast.append(
                        _buat_entry_forecast(waktu_target, kolom, nilai, "arima_forecast_v1")
                    )
                jumlah_sukses += 1

            print(f"✅ ARIMA {kolom}: forecast masa depan dibuat, backtest berhasil {jumlah_sukses}/{len(tanggal_backtest)} hari")

        except Exception as e:
            print(f"⚠️ ARIMA gagal untuk {kolom}: {e}")
            continue

    if not hasil_forecast:
        print(f"⚠️ ARIMA tidak menghasilkan forecast untuk mesin {mesin_id}")
    else:
        print(f"✅ ARIMA total menghasilkan {len(hasil_forecast)} titik forecast untuk mesin {mesin_id}")

    return hasil_forecast


def buat_forecast_arima_pendek(df, mesin_id, resolusi=RESOLUSI_PENDEK, langkah=LANGKAH_PENDEK,
                                jumlah_hari_data=JUMLAH_HARI_DATA_PENDEK):
    """
    Forecast ARIMA resolusi TINGGI (default 15 menit) untuk horizon pendek
    (dipakai dashboard untuk opsi 30 menit / 1 jam / 2 jam).
    Hanya forecast masa depan (tanpa backtest historis per-hari) — lihat catatan
    di konstanta RESOLUSI_PENDEK di atas.
    """
    print(f"\n=== [Mesin {mesin_id}] ARIMA: Forecasting horizon pendek ({resolusi}, {langkah} langkah) ===")

    if df.empty:
        return []

    df_ts = df.set_index("created_at").sort_index()
    if df_ts.index.max() is pd.NaT:
        return []

    waktu_terakhir = df_ts.index.max()
    batas_awal = waktu_terakhir - pd.Timedelta(days=jumlah_hari_data)
    df_ts_recent = df_ts[df_ts.index >= batas_awal]

    hasil_forecast = []
    delta = pd.Timedelta(resolusi)

    for kolom in ["suhu", "kecepatan_getaran"]:
        try:
            seri = df_ts_recent[kolom].resample(resolusi).mean().interpolate()
            if len(seri) < 10:
                print(f"⚠️ Data {kolom} hanya {len(seri)} titik resolusi {resolusi}, minimum 10. Dilewati.")
                continue

            model = ARIMA(seri, order=(2, 1, 2))
            hasil_fit = model.fit()
            prediksi = hasil_fit.forecast(steps=langkah)

            for i, nilai in enumerate(prediksi):
                waktu_target = waktu_terakhir + delta * (i + 1)
                hasil_forecast.append(
                    _buat_entry_forecast(waktu_target, kolom, nilai, "arima_forecast_pendek_v1")
                )
            print(f"✅ ARIMA pendek {kolom}: {langkah} titik forecast dibuat")
        except Exception as e:
            print(f"⚠️ ARIMA pendek gagal untuk {kolom}: {e}")
            continue

    return hasil_forecast


def _latih_lstm_dan_prediksi(seri_latih, jendela, langkah_ke_depan):
    """Melatih 1 model LSTM dari sebuah seri, lalu memprediksi ke depan. Dipakai bareng
    oleh forecast masa depan maupun backtest per-hari supaya tidak duplikasi kode."""
    from tensorflow import keras

    nilai = seri_latih.values.reshape(-1, 1)
    scaler = StandardScaler()
    nilai_scaled = scaler.fit_transform(nilai).flatten()

    X, y = [], []
    for i in range(len(nilai_scaled) - jendela):
        X.append(nilai_scaled[i:i + jendela])
        y.append(nilai_scaled[i + jendela])

    if len(X) == 0:
        return None, None

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

    return prediksi_asli, model


def buat_forecast_lstm(df, mesin_id, langkah_ke_depan=24, jendela=10, jumlah_hari_riwayat=JUMLAH_HARI_RIWAYAT_FORECAST):
    """
    Sama seperti buat_forecast_arima, tapi pakai LSTM (resolusi 1 jam, horizon panjang
    6/12/24 jam):
    1. Forecast masa depan dari titik data terakhir.
    2. Backtest per-hari untuk `jumlah_hari_riwayat` hari terakhir (selain hari ini),
       tiap hari dilatih ulang HANYA dengan data sebelum hari itu.

    Catatan: karena LSTM dilatih ulang untuk tiap hari, proses ini lebih lambat
    dibanding ARIMA. Turunkan jumlah_hari_riwayat kalau training terlalu lama.
    """
    print(f"\n=== [Mesin {mesin_id}] LSTM: Forecasting per hari ({langkah_ke_depan} jam/hari, Deep Learning) ===")

    if df.empty:
        print(f"⚠️ Data kosong untuk mesin {mesin_id}")
        return []

    df_ts = df.set_index("created_at").sort_index()

    if df_ts.index.max() is pd.NaT:
        print(f"⚠️ Tidak ada timestamp valid untuk mesin {mesin_id}")
        return []

    waktu_terakhir = df_ts.index.max()
    hasil_forecast = []

    for kolom in ["suhu", "kecepatan_getaran"]:
        try:
            seri_full = df_ts[kolom].resample("1h").mean().interpolate().dropna()

            if len(seri_full) < jendela + 10:
                print(f"⚠️ Data {kolom} hanya {len(seri_full)} titik, minimum {jendela + 10}. LSTM dilewati.")
                continue

            # --- 1. Forecast masa depan (dari titik data terakhir) ---
            prediksi_asli, model_final = _latih_lstm_dan_prediksi(seri_full, jendela, langkah_ke_depan)
            if prediksi_asli is not None:
                for i, nilai_pred in enumerate(prediksi_asli):
                    waktu_target = waktu_terakhir + pd.Timedelta(hours=i + 1)
                    hasil_forecast.append(
                        _buat_entry_forecast(waktu_target, kolom, nilai_pred, "lstm_forecast_v1")
                    )
                model_final.save(f"model_lstm_forecast_{kolom}_mesin{mesin_id}.keras")
            else:
                print(f"⚠️ Data {kolom} tidak cukup untuk membuat sequence LSTM (forecast masa depan)")

            # --- 2. Backtest per hari (tidak termasuk hari ini) ---
            tanggal_tersedia = sorted(set(seri_full.index.date))
            tanggal_hari_ini = waktu_terakhir.date()
            tanggal_backtest = [t for t in tanggal_tersedia if t < tanggal_hari_ini][-jumlah_hari_riwayat:]

            jumlah_sukses = 0
            for tanggal_target in tanggal_backtest:
                batas_cutoff = pd.Timestamp(tanggal_target)
                seri_latih = seri_full[seri_full.index < batas_cutoff]

                if len(seri_latih) < jendela + 10:
                    continue

                try:
                    prediksi_bt, _ = _latih_lstm_dan_prediksi(seri_latih, jendela, langkah_ke_depan)
                except Exception as e:
                    print(f"⚠️ LSTM (backtest {tanggal_target}) gagal untuk {kolom}: {e}")
                    continue

                if prediksi_bt is None:
                    continue

                for i, nilai_pred in enumerate(prediksi_bt):
                    waktu_target = batas_cutoff + pd.Timedelta(hours=i)
                    hasil_forecast.append(
                        _buat_entry_forecast(waktu_target, kolom, nilai_pred, "lstm_forecast_v1")
                    )
                jumlah_sukses += 1

            print(f"✅ LSTM {kolom}: forecast masa depan dibuat, backtest berhasil {jumlah_sukses}/{len(tanggal_backtest)} hari")

        except Exception as e:
            print(f"⚠️ LSTM gagal untuk {kolom}: {e}")
            continue

    if not hasil_forecast:
        print(f"⚠️ LSTM tidak menghasilkan forecast untuk mesin {mesin_id}")
    else:
        print(f"✅ LSTM total menghasilkan {len(hasil_forecast)} titik forecast untuk mesin {mesin_id}")

    return hasil_forecast


def buat_forecast_lstm_pendek(df, mesin_id, resolusi=RESOLUSI_PENDEK, langkah=LANGKAH_PENDEK,
                               jendela=8, jumlah_hari_data=JUMLAH_HARI_DATA_PENDEK):
    """
    Forecast LSTM resolusi TINGGI (default 15 menit) untuk horizon pendek.
    Hanya forecast masa depan, tanpa backtest per-hari (lihat catatan di
    buat_forecast_arima_pendek).
    """
    print(f"\n=== [Mesin {mesin_id}] LSTM: Forecasting horizon pendek ({resolusi}, {langkah} langkah) ===")

    if df.empty:
        return []

    df_ts = df.set_index("created_at").sort_index()
    if df_ts.index.max() is pd.NaT:
        return []

    waktu_terakhir = df_ts.index.max()
    batas_awal = waktu_terakhir - pd.Timedelta(days=jumlah_hari_data)
    df_ts_recent = df_ts[df_ts.index >= batas_awal]

    hasil_forecast = []
    delta = pd.Timedelta(resolusi)

    for kolom in ["suhu", "kecepatan_getaran"]:
        try:
            seri = df_ts_recent[kolom].resample(resolusi).mean().interpolate().dropna()
            if len(seri) < jendela + 10:
                print(f"⚠️ Data {kolom} hanya {len(seri)} titik resolusi {resolusi}, minimum {jendela + 10}. Dilewati.")
                continue

            prediksi_asli, model_final = _latih_lstm_dan_prediksi(seri, jendela, langkah)
            if prediksi_asli is None:
                continue

            for i, nilai_pred in enumerate(prediksi_asli):
                waktu_target = waktu_terakhir + delta * (i + 1)
                hasil_forecast.append(
                    _buat_entry_forecast(waktu_target, kolom, nilai_pred, "lstm_forecast_pendek_v1")
                )
            model_final.save(f"model_lstm_forecast_pendek_{kolom}_mesin{mesin_id}.keras")
            print(f"✅ LSTM pendek {kolom}: {langkah} titik forecast dibuat")
        except Exception as e:
            print(f"⚠️ LSTM pendek gagal untuk {kolom}: {e}")
            continue

    return hasil_forecast


def proses_satu_mesin(mesin_id):
    print(f"\n{'=' * 60}")
    print(f"Memproses Mesin Bubut {mesin_id}")
    print(f"{'=' * 60}")

    df = ambil_data_sensor(limit=10000, mesin_id=mesin_id)
    print(f"Total data mesin {mesin_id} SEBELUM dibersihkan: {len(df)} baris")

    df, ringkasan_bersih = bersihkan_data(
        df,
        batas_suhu_min=BATAS_SUHU_MIN,
        batas_suhu_max=BATAS_SUHU_MAX,
        getaran_boleh_negatif=GETARAN_BOLEH_NEGATIF,
        hapus_outlier=False,  # Training tidak hapus outlier agar deteksi anomali tetap bisa
    )
    print(f"Ringkasan pembersihan mesin {mesin_id}: {ringkasan_bersih}")
    print(f"Total data mesin {mesin_id} SESUDAH dibersihkan: {len(df)} baris")

    if len(df) < 50:
        print(f"Data mesin {mesin_id} terlalu sedikit setelah dibersihkan. Dilewati.")
        return

    baris_terakhir = df.iloc[[-1]]
    id_terakhir = int(baris_terakhir.iloc[0]["id"])

    # 1. RandomForest Classifier
    model_rf = latih_klasifikasi_rf(df, mesin_id)
    pred_rf = model_rf.predict(baris_terakhir[["suhu", "kecepatan_getaran"]])[0]
    kirim_hasil_analisis(
        data_id=id_terakhir,
        prediksi_kondisi=pred_rf,
        sumber="random_forest_klasifikasi_v1",
        keterangan="Prediksi kondisi terbaru (RandomForest)",
        mesin_id=mesin_id,
    )

    # 2. MLP Classifier
    model_mlp, scaler_mlp = latih_klasifikasi_mlp(df, mesin_id)
    X_terakhir_scaled = scaler_mlp.transform(baris_terakhir[["suhu", "kecepatan_getaran"]])
    pred_mlp = model_mlp.predict(X_terakhir_scaled)[0]
    kirim_hasil_analisis(
        data_id=id_terakhir,
        prediksi_kondisi=pred_mlp,
        sumber="mlp_klasifikasi_v1",
        keterangan="Prediksi kondisi terbaru (Neural Network/MLP)",
        mesin_id=mesin_id,
    )

    # 3. Isolation Forest (Deteksi Anomali)
    df_if = latih_isolation_forest(df, mesin_id)
    for _, row in df_if[df_if["anomali_if"]].iterrows():
        kirim_hasil_analisis(
            data_id=int(row["id"]),
            skor_anomali=float(row["skor_anomali_if"]),
            sumber="isolation_forest_v1",
            keterangan="Anomali terdeteksi (Isolation Forest)",
            mesin_id=mesin_id,
        )

    # 4. Autoencoder (Deteksi Anomali Deep Learning)
    df_ae = latih_autoencoder(df, mesin_id)
    for _, row in df_ae[df_ae["anomali_ae"]].iterrows():
        kirim_hasil_analisis(
            data_id=int(row["id"]),
            skor_anomali=float(row["skor_anomali_ae"]),
            sumber="autoencoder_v1",
            keterangan="Anomali terdeteksi (Autoencoder/Deep Learning)",
            mesin_id=mesin_id,
        )

    # 5. K-Means Clustering
    # PENTING: kirim label cluster untuk SEMUA baris (bukan cuma 100 baris terakhir
    # seperti sebelumnya), supaya jumlah data yang tergabung & ditampilkan di
    # dashboard cocok dengan jumlah data mentah yang diambil pengguna.
    df_cluster, _ = latih_clustering(df, mesin_id, jumlah_cluster=3)
    baris_untuk_dikirim = (
        df_cluster if MAKS_BARIS_KIRIM_CLUSTER is None else df_cluster.tail(MAKS_BARIS_KIRIM_CLUSTER)
    )
    for _, row in baris_untuk_dikirim.iterrows():
        kirim_hasil_analisis(
            data_id=int(row["id"]),
            sumber="kmeans_cluster_v1",
            keterangan=f"Cluster {int(row['cluster'])}",
            mesin_id=mesin_id,
        )
    print(f"✅ Cluster dikirim untuk {len(baris_untuk_dikirim)}/{len(df_cluster)} baris (mesin {mesin_id})")

    # 6. SHAP (Feature Importance)
    hasil_shap = hitung_shap(df, model_rf)
    if hasil_shap is not None:
        kirim_hasil_analisis(
            data_id=id_terakhir,
            sumber="shap_importance_v1",
            keterangan=json.dumps(hasil_shap),
            mesin_id=mesin_id,
        )

    # 7. ARIMA Forecasting resolusi 1 jam (horizon panjang: 6/12/24 jam, + backtest per hari)
    forecast_arima = buat_forecast_arima(df, mesin_id, jam_ke_depan=24)
    if forecast_arima:
        # Hapus forecast ARIMA lama mesin ini dulu, supaya tabel forecast_mesin
        # tidak terus menumpuk baris duplikat/kadaluarsa tiap training diulang.
        # Aman: batch baru yang dikirim sesudahnya selalu mencakup ulang seluruh
        # rentang hari yang sama (masa depan + backtest N hari terakhir).
        try:
            hapus_forecast_lama(mesin_id=mesin_id, sumber="arima_forecast_v1")
        except Exception as e:
            print(f"⚠️ Gagal menghapus forecast ARIMA lama (mesin {mesin_id}): {e} — tetap lanjut kirim yang baru.")

        kirim_forecast(forecast_arima, mesin_id=mesin_id)
        print(f"✅ Terkirim {len(forecast_arima)} titik forecast ARIMA (mesin {mesin_id})")

    # 8. LSTM Forecasting resolusi 1 jam (horizon panjang: 6/12/24 jam, + backtest per hari)
    forecast_lstm = buat_forecast_lstm(df, mesin_id, langkah_ke_depan=24)
    if forecast_lstm:
        try:
            hapus_forecast_lama(mesin_id=mesin_id, sumber="lstm_forecast_v1")
        except Exception as e:
            print(f"⚠️ Gagal menghapus forecast LSTM lama (mesin {mesin_id}): {e} — tetap lanjut kirim yang baru.")

        kirim_forecast(forecast_lstm, mesin_id=mesin_id)
        print(f"✅ Terkirim {len(forecast_lstm)} titik forecast LSTM (mesin {mesin_id})")

    # 9. ARIMA Forecasting resolusi 15 menit (horizon pendek: 30 menit / 1 jam / 2 jam)
    forecast_arima_pendek = buat_forecast_arima_pendek(df, mesin_id)
    if forecast_arima_pendek:
        try:
            hapus_forecast_lama(mesin_id=mesin_id, sumber="arima_forecast_pendek_v1")
        except Exception as e:
            print(f"⚠️ Gagal menghapus forecast ARIMA pendek lama (mesin {mesin_id}): {e} — tetap lanjut kirim yang baru.")

        kirim_forecast(forecast_arima_pendek, mesin_id=mesin_id)
        print(f"✅ Terkirim {len(forecast_arima_pendek)} titik forecast ARIMA pendek (mesin {mesin_id})")

    # 10. LSTM Forecasting resolusi 15 menit (horizon pendek: 30 menit / 1 jam / 2 jam)
    forecast_lstm_pendek = buat_forecast_lstm_pendek(df, mesin_id)
    if forecast_lstm_pendek:
        try:
            hapus_forecast_lama(mesin_id=mesin_id, sumber="lstm_forecast_pendek_v1")
        except Exception as e:
            print(f"⚠️ Gagal menghapus forecast LSTM pendek lama (mesin {mesin_id}): {e} — tetap lanjut kirim yang baru.")

        kirim_forecast(forecast_lstm_pendek, mesin_id=mesin_id)
        print(f"✅ Terkirim {len(forecast_lstm_pendek)} titik forecast LSTM pendek (mesin {mesin_id})")

    print(f"\n✅ Selesai memproses Mesin {mesin_id}")


def main():
    print("=" * 60)
    print("PROSES TRAINING DAN ANALISIS DATA SENSOR MESIN BUBUT")
    print("=" * 60)

    for mesin_id in DAFTAR_MESIN:
        try:
            proses_satu_mesin(mesin_id)
        except Exception as e:
            print(f"\n⚠️ Gagal memproses Mesin {mesin_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    print("✅ SELESAI: Training & pengiriman hasil untuk semua mesin")
    print("=" * 60)


if __name__ == "__main__":
    main()
