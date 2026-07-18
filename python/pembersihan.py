"""
Modul pembersihan & verifikasi data sensor mesin.
Dipakai bersama oleh train_model.py (pembersihan sebelum training)
dan dashboard.py (menampilkan status verifikasi & tombol pembersihan manual).

PRINSIP PENTING:
- Yang dibersihkan/dihapus di sini HANYA "sampah data" (rusak/tidak valid secara fisik,
  kosong, duplikat, label tidak konsisten) -- BUKAN outlier statistik.
- Outlier statistik (nilai ekstrem tapi masih masuk akal secara fisik) SENGAJA TIDAK
  dihapus, karena itu justru target yang mau dideteksi oleh Isolation Forest & Autoencoder.
  Outlier cuma DILAPORKAN di audit, tidak pernah dihapus otomatis.
"""

import pandas as pd
import numpy as np


def audit_data(df: pd.DataFrame, batas_suhu_min: float = 0, batas_suhu_max: float = 150,
                getaran_boleh_negatif: bool = False) -> dict:
    """
    Periksa data TANPA mengubah apa pun. Kembalikan laporan lengkap kondisi data.
    """
    laporan = {}
    total_baris = len(df)
    laporan["total_baris"] = total_baris

    if total_baris == 0:
        laporan["ada_masalah_kritis"] = False
        laporan["jumlah_masalah_kritis"] = 0
        return laporan

    # 1. Kelengkapan (missing values)
    kolom_wajib = [k for k in ["suhu", "kecepatan_getaran", "kondisi", "created_at"] if k in df.columns]
    laporan["nilai_kosong"] = df[kolom_wajib].isna().sum().to_dict()
    laporan["total_baris_ada_kosong"] = int(df[kolom_wajib].isna().any(axis=1).sum())

    # 2. Validitas fisik
    mask_suhu_invalid = ~df["suhu"].between(batas_suhu_min, batas_suhu_max)
    if getaran_boleh_negatif:
        mask_getaran_invalid = pd.Series(False, index=df.index)
    else:
        mask_getaran_invalid = df["kecepatan_getaran"] < 0
    laporan["suhu_tidak_masuk_akal"] = int(mask_suhu_invalid.sum())
    laporan["getaran_tidak_masuk_akal"] = int(mask_getaran_invalid.sum())

    # 3. Duplikasi
    laporan["baris_duplikat_penuh"] = int(df.duplicated().sum())
    laporan["timestamp_duplikat"] = int(df["created_at"].duplicated().sum()) if "created_at" in df.columns else 0

    # 4. Konsistensi label kondisi
    if "kondisi" in df.columns:
        label_valid = {"NORMAL", "PERINGATAN", "TIDAK NORMAL"}
        label_bersih = df["kondisi"].astype(str).str.strip().str.upper()
        laporan["label_tidak_baku"] = int((~label_bersih.isin(label_valid)).sum())
        laporan["distribusi_label"] = label_bersih.value_counts().to_dict()
    else:
        laporan["label_tidak_baku"] = 0
        laporan["distribusi_label"] = {}

    # 5. Outlier statistik (metode IQR) -- HANYA DILAPORKAN, TIDAK PERNAH DIHAPUS
    outlier_info = {}
    for kolom in ["suhu", "kecepatan_getaran"]:
        data_kolom = df[kolom].dropna()
        if len(data_kolom) < 4:
            outlier_info[kolom] = 0
            continue
        q1, q3 = data_kolom.quantile([0.25, 0.75])
        iqr = q3 - q1
        batas_bawah = q1 - 1.5 * iqr
        batas_atas = q3 + 1.5 * iqr
        outlier_info[kolom] = int(((data_kolom < batas_bawah) | (data_kolom > batas_atas)).sum())
    laporan["outlier_statistik"] = outlier_info

    # 6. Kesinambungan waktu (celah > 2 jam dianggap gap signifikan)
    if "created_at" in df.columns and total_baris > 1:
        waktu_urut = pd.to_datetime(df["created_at"]).sort_values()
        selisih = waktu_urut.diff().dropna()
        laporan["jumlah_celah_waktu_besar"] = int((selisih > pd.Timedelta(hours=2)).sum())
    else:
        laporan["jumlah_celah_waktu_besar"] = 0

    # Kesimpulan status siap-latih (HANYA berdasarkan masalah kritis, BUKAN outlier statistik)
    masalah_kritis = (
        laporan["total_baris_ada_kosong"]
        + laporan["suhu_tidak_masuk_akal"]
        + laporan["getaran_tidak_masuk_akal"]
        + laporan["baris_duplikat_penuh"]
        + laporan["label_tidak_baku"]
    )
    laporan["ada_masalah_kritis"] = masalah_kritis > 0
    laporan["jumlah_masalah_kritis"] = int(masalah_kritis)

    return laporan


def bersihkan_data(df: pd.DataFrame, batas_suhu_min: float = 0, batas_suhu_max: float = 150,
                    getaran_boleh_negatif: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Bersihkan HANYA "sampah data" (BUKAN outlier statistik).
    Kembalikan (df_bersih, ringkasan_perubahan).
    """
    ringkasan = {"baris_sebelum": len(df)}
    df_bersih = df.copy()

    # 1. Hapus baris dengan nilai kosong di kolom penting
    kolom_wajib = [k for k in ["suhu", "kecepatan_getaran", "kondisi", "created_at"] if k in df_bersih.columns]
    sebelum = len(df_bersih)
    df_bersih = df_bersih.dropna(subset=kolom_wajib)
    ringkasan["dihapus_karena_kosong"] = sebelum - len(df_bersih)

    # 2. Hapus duplikat penuh (semua kolom sama persis)
    sebelum = len(df_bersih)
    df_bersih = df_bersih.drop_duplicates()
    ringkasan["dihapus_karena_duplikat"] = sebelum - len(df_bersih)

    # 3. Hapus nilai yang mustahil secara fisik
    sebelum = len(df_bersih)
    df_bersih = df_bersih[df_bersih["suhu"].between(batas_suhu_min, batas_suhu_max)]
    if not getaran_boleh_negatif:
        df_bersih = df_bersih[df_bersih["kecepatan_getaran"] >= 0]
    ringkasan["dihapus_karena_tidak_masuk_akal"] = sebelum - len(df_bersih)

    # 4. Standarisasi & saring label kondisi tidak baku
    if "kondisi" in df_bersih.columns:
        df_bersih["kondisi"] = df_bersih["kondisi"].astype(str).str.strip().str.upper()
        label_valid = {"NORMAL", "PERINGATAN", "TIDAK NORMAL"}
        sebelum = len(df_bersih)
        df_bersih = df_bersih[df_bersih["kondisi"].isin(label_valid)]
        ringkasan["dihapus_karena_label_tidak_baku"] = sebelum - len(df_bersih)
    else:
        ringkasan["dihapus_karena_label_tidak_baku"] = 0

    ringkasan["baris_sesudah"] = len(df_bersih)
    ringkasan["total_dihapus"] = ringkasan["baris_sebelum"] - ringkasan["baris_sesudah"]

    return df_bersih, ringkasan
