"""
Training model untuk data sensor mesin.
Jalankan: python train_model.py

Mencakup:
1. Regresi linear (suhu -> getaran) - untuk lihat kekuatan hubungan
2. Klasifikasi kondisi (suhu + getaran -> NORMAL/PERINGATAN/TIDAK NORMAL)
3. Deteksi anomali (Isolation Forest)
4. Kirim beberapa contoh hasil prediksi balik ke API (opsional, lihat bagian akhir)
"""

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, r2_score
from sklearn.model_selection import train_test_split

from api_client import ambil_data_sensor, kirim_hasil_analisis

MODEL_REGRESI_PATH = "model_regresi_suhu_getaran.pkl"
MODEL_KLASIFIKASI_PATH = "model_klasifikasi_kondisi.pkl"
MODEL_ANOMALI_PATH = "model_anomali.pkl"


def latih_regresi(df):
    print("\n=== 1. Regresi Linear: suhu -> kecepatan_getaran ===")
    X = df[["suhu"]]
    y = df["kecepatan_getaran"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    r2 = r2_score(y_test, model.predict(X_test))
    print(f"R² pada data uji: {r2:.3f}")
    print(f"Koefisien: {model.coef_[0]:.4f}, Intercept: {model.intercept_:.4f}")
    if r2 < 0.3:
        print(
            "Catatan: R² rendah, artinya suhu sendirian TIDAK cukup untuk memprediksi "
            "getaran secara linear. Ini wajar berdasarkan korelasi yang kita lihat di EDA (~0.43). "
            "Hubungan keduanya kemungkinan tidak murni linear / dipengaruhi faktor lain."
        )

    joblib.dump(model, MODEL_REGRESI_PATH)
    print(f"Model disimpan: {MODEL_REGRESI_PATH}")
    return model


def latih_klasifikasi(df):
    print("\n=== 2. Klasifikasi Kondisi (RandomForest) ===")
    X = df[["suhu", "kecepatan_getaran"]]
    y = df["kondisi"]

    # data TIDAK NORMAL sangat sedikit -> pakai class_weight='balanced'
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    print("Pentingnya fitur:")
    for fitur, skor in zip(X.columns, model.feature_importances_):
        print(f"  {fitur}: {skor:.3f}")

    joblib.dump(model, MODEL_KLASIFIKASI_PATH)
    print(f"Model disimpan: {MODEL_KLASIFIKASI_PATH}")
    return model


def latih_deteksi_anomali(df):
    print("\n=== 3. Deteksi Anomali (Isolation Forest) ===")
    X = df[["suhu", "kecepatan_getaran"]]

    model = IsolationForest(contamination=0.02, random_state=42)
    model.fit(X)

    skor = model.decision_function(X)  # makin rendah = makin anomali
    label = model.predict(X)  # -1 = anomali, 1 = normal

    df_hasil = df.copy()
    df_hasil["skor_anomali"] = skor
    df_hasil["anomali"] = label == -1

    jumlah_anomali = df_hasil["anomali"].sum()
    print(f"Ditemukan {jumlah_anomali} baris terindikasi anomali dari {len(df)} total data")
    print(df_hasil[df_hasil["anomali"]][["id", "suhu", "kecepatan_getaran", "kondisi", "skor_anomali"]])

    joblib.dump(model, MODEL_ANOMALI_PATH)
    print(f"Model disimpan: {MODEL_ANOMALI_PATH}")
    return model, df_hasil


def main():
    print("Mengambil data dari API...")
    df = ambil_data_sensor(limit=5000)
    print(f"Total data untuk training: {len(df)} baris")

    latih_regresi(df)
    model_klasifikasi = latih_klasifikasi(df)
    _, df_anomali = latih_deteksi_anomali(df)

    # ---------- Kirim prediksi kondisi utk data TERBARU ke API ----------
    # Ini dipakai dashboard biar tidak perlu baca file model lokal (penting untuk cloud)
    baris_terakhir = df.iloc[[-1]]
    prediksi_terkini = model_klasifikasi.predict(baris_terakhir[["suhu", "kecepatan_getaran"]])[0]
    kirim_hasil_analisis(
        data_id=int(baris_terakhir.iloc[0]["id"]),
        prediksi_kondisi=prediksi_terkini,
        sumber="random_forest_klasifikasi_v1",
        keterangan="Prediksi kondisi untuk data terbaru",
    )
    print(f"\nPrediksi kondisi terkini: {prediksi_terkini} (dikirim ke API)")

    # ---------- Kirim hasil deteksi anomali untuk baris yang terindikasi anomali saja ----------
    baris_anomali = df_anomali[df_anomali["anomali"]]
    if len(baris_anomali) > 0:
        print(f"\nMengirim {len(baris_anomali)} hasil anomali ke API...")
        for _, row in baris_anomali.iterrows():
            kirim_hasil_analisis(
                data_id=int(row["id"]),
                skor_anomali=float(row["skor_anomali"]),
                sumber="isolation_forest_v1",
                keterangan="Terindikasi anomali oleh Isolation Forest",
            )
        print("Selesai mengirim.")


if __name__ == "__main__":
    main()
