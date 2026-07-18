"""
Modul pembersihan & verifikasi data sensor mesin.
Dipakai bersama oleh train_model.py dan dashboard.py.

PRINSIP:
- Data yang dibersihkan: kosong, duplikat, nilai tidak masuk akal secara fisik, 
  label tidak konsisten, dan OPSIONAL outlier statistik (IQR).
- Outlier statistik bisa dihapus jika pengguna menginginkannya.
"""

import pandas as pd
import numpy as np
from api_client import hapus_data_sensor


def audit_data(df: pd.DataFrame, batas_suhu_min: float = 27, batas_suhu_max: float = 150,
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

    kolom_wajib = [k for k in ["suhu", "kecepatan_getaran", "kondisi", "created_at"] if k in df.columns]
    laporan["nilai_kosong"] = df[kolom_wajib].isna().sum().to_dict()
    laporan["total_baris_ada_kosong"] = int(df[kolom_wajib].isna().any(axis=1).sum())

    mask_suhu_invalid = ~df["suhu"].between(batas_suhu_min, batas_suhu_max)
    if getaran_boleh_negatif:
        mask_getaran_invalid = pd.Series(False, index=df.index)
    else:
        mask_getaran_invalid = df["kecepatan_getaran"] < 0
    laporan["suhu_tidak_masuk_akal"] = int(mask_suhu_invalid.sum())
    laporan["getaran_tidak_masuk_akal"] = int(mask_getaran_invalid.sum())

    laporan["baris_duplikat_penuh"] = int(df.duplicated().sum())
    laporan["timestamp_duplikat"] = int(df["created_at"].duplicated().sum()) if "created_at" in df.columns else 0

    if "kondisi" in df.columns:
        label_valid = {"NORMAL", "PERINGATAN", "TIDAK NORMAL"}
        label_bersih = df["kondisi"].astype(str).str.strip().str.upper()
        laporan["label_tidak_baku"] = int((~label_bersih.isin(label_valid)).sum())
        laporan["distribusi_label"] = label_bersih.value_counts().to_dict()
    else:
        laporan["label_tidak_baku"] = 0
        laporan["distribusi_label"] = {}

    # Outlier statistik (IQR) — tetap dihitung untuk laporan
    outlier_info = {}
    for kolom in ["suhu", "kecepatan_getaran"]:
        data_kolom = df[kolom].dropna()
        if len(data_kolom) < 4:
            outlier_info[kolom] = 0            continue
        q1, q3 = data_kolom.quantile([0.25, 0.75])
        iqr = q3 - q1
        batas_bawah = q1 - 1.5 * iqr
        batas_atas = q3 + 1.5 * iqr
        outlier_info[kolom] = int(((data_kolom < batas_bawah) | (data_kolom > batas_atas)).sum())
    laporan["outlier_statistik"] = outlier_info

    if "created_at" in df.columns and total_baris > 1:
        waktu_urut = pd.to_datetime(df["created_at"]).sort_values()
        selisih = waktu_urut.diff().dropna()
        laporan["jumlah_celah_waktu_besar"] = int((selisih > pd.Timedelta(hours=2)).sum())
    else:
        laporan["jumlah_celah_waktu_besar"] = 0

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


def bersihkan_data(df: pd.DataFrame, batas_suhu_min: float = 27, batas_suhu_max: float = 150,
                    getaran_boleh_negatif: bool = False, hapus_outlier: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Bersihkan data: kosong, duplikat, nilai tidak masuk akal, label tidak baku,
    dan OPSIONAL outlier statistik (IQR).
    
    Returns:
        tuple: (df_bersih, ringkasan_perubahan)
    """
    ringkasan = {"baris_sebelum": len(df)}
    df_bersih = df.copy()

    # 1. Hapus baris dengan nilai kosong
    kolom_wajib = [k for k in ["suhu", "kecepatan_getaran", "kondisi", "created_at"] if k in df_bersih.columns]
    sebelum = len(df_bersih)
    df_bersih = df_bersih.dropna(subset=kolom_wajib)
    ringkasan["dihapus_karena_kosong"] = sebelum - len(df_bersih)

    # 2. Hapus duplikat
    sebelum = len(df_bersih)
    df_bersih = df_bersih.drop_duplicates()
    ringkasan["dihapus_karena_duplikat"] = sebelum - len(df_bersih)

    # 3. Hapus nilai yang mustahil secara fisik
    sebelum = len(df_bersih)
    df_bersih = df_bersih[df_bersih["suhu"].between(batas_suhu_min, batas_suhu_max)]
    if not getaran_boleh_negatif:
        df_bersih = df_bersih[df_bersih["kecepatan_getaran"] >= 0]
    ringkasan["dihapus_karena_tidak_masuk_akal"] = sebelum - len(df_bersih)

    # 4. Standarisasi label kondisi
    if "kondisi" in df_bersih.columns:
        df_bersih["kondisi"] = df_bersih["kondisi"].astype(str).str.strip().str.upper()
        label_valid = {"NORMAL", "PERINGATAN", "TIDAK NORMAL"}
        sebelum = len(df_bersih)
        df_bersih = df_bersih[df_bersih["kondisi"].isin(label_valid)]
        ringkasan["dihapus_karena_label_tidak_baku"] = sebelum - len(df_bersih)
    else:
        ringkasan["dihapus_karena_label_tidak_baku"] = 0

    # 5. Hapus outlier statistik (IQR) — OPSIONAL
    ringkasan["dihapus_karena_outlier"] = 0
    if hapus_outlier and len(df_bersih) >= 4:
        sebelum = len(df_bersih)
        for kolom in ["suhu", "kecepatan_getaran"]:
            data_kolom = df_bersih[kolom].dropna()
            if len(data_kolom) >= 4:
                q1, q3 = data_kolom.quantile([0.25, 0.75])
                iqr = q3 - q1
                batas_bawah = q1 - 1.5 * iqr
                batas_atas = q3 + 1.5 * iqr
                df_bersih = df_bersih[
                    (df_bersih[kolom] >= batas_bawah) & 
                    (df_bersih[kolom] <= batas_atas)
                ]
        ringkasan["dihapus_karena_outlier"] = sebelum - len(df_bersih)

    ringkasan["baris_sesudah"] = len(df_bersih)
    ringkasan["total_dihapus"] = ringkasan["baris_sebelum"] - ringkasan["baris_sesudah"]

    return df_bersih, ringkasan


def hapus_data_dari_db(df_original: pd.DataFrame, df_bersih: pd.DataFrame, mesin_id: int) -> dict:
    """
    Menghapus data yang tidak lolos pembersihan dari database.
    
    Args:
        df_original: DataFrame asli sebelum pembersihan
        df_bersih: DataFrame setelah pembersihan
        mesin_id: ID mesin
    
    Returns:
        dict: Hasil penghapusan dari API
    """
    ids_original = set(df_original["id"].tolist())
    ids_bersih = set(df_bersih["id"].tolist())
    ids_dihapus = list(ids_original - ids_bersih)
    
    if not ids_dihapus:
        return {"total_dihapus": 0, "message": "Tidak ada data yang perlu dihapus"}
    
    # Hapus dalam batch (maksimal 100 per request)
    total_dihapus = 0
    for i in range(0, len(ids_dihapus), 100):
        batch = ids_dihapus[i:i+100]
        try:
            hasil = hapus_data_sensor(batch, mesin_id=mesin_id)
            total_dihapus += len(batch)
        except Exception as e:
            print(f"Gagal menghapus batch: {e}")
            continue
    
    return {
        "total_dihapus": total_dihapus,
        "message": f"Berhasil menghapus {total_dihapus} baris data"
    }
