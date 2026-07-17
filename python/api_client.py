"""
Modul untuk komunikasi dengan API CI4 di hosting.
Dipakai bareng oleh: eda.py, train_model.py, dan dashboard.py

Sebelum pakai, pastikan sudah bikin file .env (lihat .env.contoh)
"""

import os
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()  # baca file .env kalau ada (tidak error kalau tidak ada, misal di cloud)


def _ambil_kredensial():
    """
    Cari API_BASE_URL & API_KEY dari 3 sumber, urut prioritas:
    1. Environment variable biasa (GitHub Actions, atau .env lokal)
    2. Streamlit Cloud secrets (st.secrets), kalau dijalankan sebagai app Streamlit
    """
    base_url = os.getenv("API_BASE_URL")
    api_key = os.getenv("API_KEY")

    if not base_url or not api_key:
        try:
            import streamlit as st  # hanya ada kalau dijalankan via streamlit run
            base_url = base_url or st.secrets.get("API_BASE_URL")
            api_key = api_key or st.secrets.get("API_KEY")
        except Exception:
            pass

    return base_url, api_key


API_BASE_URL, API_KEY = _ambil_kredensial()

if not API_BASE_URL or not API_KEY:
    raise RuntimeError(
        "API_BASE_URL / API_KEY belum diisi. "
        "Kalau jalan lokal: salin .env.contoh jadi .env lalu isi nilainya. "
        "Kalau di GitHub Actions: isi lewat Settings > Secrets. "
        "Kalau di Streamlit Cloud: isi lewat menu Secrets di dashboard app."
    )

HEADERS = {
    "X-API-KEY": API_KEY,
    # Beberapa hosting shared (mod_security/Imunify360) memblokir request yang tidak
    # punya User-Agent seperti browser. Ini membantu request kita "lolos" dari filter itu.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}


def _get(url, **kwargs):
    """Wrapper requests.get dengan pesan error yang lebih jelas."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30, **kwargs)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Tidak bisa konek ke {url}. Kemungkinan penyebab: "
            "(1) domain salah/typo, (2) hosting memblokir request otomatis "
            "(coba buka URL ini langsung di browser untuk pastikan), "
            f"(3) SSL/sertifikat bermasalah. Detail asli: {e}"
        ) from e

    if not resp.ok:
        raise RuntimeError(
            f"Server merespons tapi dengan error {resp.status_code}: {resp.text[:300]}"
        )
    return resp


def _post(url, **kwargs):
    """Wrapper requests.post dengan pesan error yang lebih jelas."""
    try:
        resp = requests.post(url, headers=HEADERS, timeout=30, **kwargs)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Tidak bisa konek ke {url}. Kemungkinan penyebab: "
            "(1) domain salah/typo, (2) hosting memblokir request otomatis, "
            f"(3) SSL/sertifikat bermasalah. Detail asli: {e}"
        ) from e

    if not resp.ok:
        raise RuntimeError(
            f"Server merespons tapi dengan error {resp.status_code}: {resp.text[:300]}"
        )
    return resp


def ambil_data_sensor(limit: int = 1000, sejak_id: int | None = None) -> pd.DataFrame:
    """
    Ambil data dari endpoint GET /api/data-sensor dan kembalikan sebagai DataFrame.

    Contoh:
        df = ambil_data_sensor(limit=2000)
        df_baru = ambil_data_sensor(sejak_id=739)  # hanya data setelah id 739
    """
    params = {"limit": limit}
    if sejak_id is not None:
        params["sejak_id"] = sejak_id

    # 🔥 PERBAIKAN: tambahkan /api/ di depan endpoint
    resp = _get(f"{API_BASE_URL}/api/data-sensor", params=params)

    payload = resp.json()
    df = pd.DataFrame(payload["data"])

    if not df.empty:
        df["suhu"] = df["suhu"].astype(float)
        df["kecepatan_getaran"] = df["kecepatan_getaran"].astype(float)
        df["created_at"] = pd.to_datetime(df["created_at"])

    return df


def kirim_hasil_analisis(
    data_id: int,
    prediksi_kondisi: str | None = None,
    skor_anomali: float | None = None,
    sumber: str = "python_local",
    keterangan: str | None = None,
) -> dict:
    """
    Kirim satu hasil analisis ke endpoint POST /api/hasil-analisis.
    """
    body = {
        "data_id": data_id,
        "prediksi_kondisi": prediksi_kondisi,
        "skor_anomali": skor_anomali,
        "sumber": sumber,
        "keterangan": keterangan,
    }
    # 🔥 PERBAIKAN: tambahkan /api/ di depan endpoint
    resp = _post(f"{API_BASE_URL}/api/hasil-analisis", json=body)
    return resp.json()


def ambil_hasil_terbaru(limit: int = 50) -> pd.DataFrame:
    """
    Ambil hasil analisis/prediksi terbaru dari endpoint GET /api/hasil-terbaru.
    Dipakai dashboard supaya tidak perlu baca file model .pkl lokal.
    """
    # 🔥 PERBAIKAN: tambahkan /api/ di depan endpoint
    resp = _get(f"{API_BASE_URL}/api/hasil-terbaru", params={"limit": limit})
    payload = resp.json()
    df = pd.DataFrame(payload["data"])
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df

def kirim_forecast(daftar_forecast: list) -> dict:
    """
    Kirim banyak titik forecast sekaligus ke endpoint POST /api/kirim-forecast.

    daftar_forecast: list of dict, contoh:
        [{"target_waktu": "2026-07-18 10:00:00", "nilai_suhu_prediksi": 42.1,
          "nilai_getaran_prediksi": None, "sumber": "arima_forecast_v1"}, ...]
    """
    resp = _post(f"{API_BASE_URL}/kirim-forecast", json={"data": daftar_forecast})
    return resp.json()


def ambil_forecast_terbaru(sumber: str | None = None, limit: int = 100) -> pd.DataFrame:
    """
    Ambil hasil forecast dari endpoint GET /api/forecast-terbaru.
    """
    params = {"limit": limit}
    if sumber:
        params["sumber"] = sumber

    resp = _get(f"{API_BASE_URL}/forecast-terbaru", params=params)
    payload = resp.json()
    df = pd.DataFrame(payload["data"])
    if not df.empty:
        df["target_waktu"] = pd.to_datetime(df["target_waktu"])
    return df

if __name__ == "__main__":
    # Tes cepat: jalankan "python api_client.py" untuk memastikan koneksi ke API berhasil
    df = ambil_data_sensor(limit=10)
    print(f"Berhasil ambil {len(df)} baris data terbaru:")
    print(df)
