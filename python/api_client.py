"""
Modul untuk komunikasi dengan API CI4 di hosting.
Dipakai bareng oleh: eda.py, train_model.py, dan dashboard.py
Sebelum pakai, pastikan sudah bikin file .env (lihat .env.contoh)
"""

import os
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()


def _ambil_kredensial():
    base_url = os.getenv("API_BASE_URL")
    api_key = os.getenv("API_KEY")

    if not base_url or not api_key:
        try:
            import streamlit as st
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
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}


def _get(url, **kwargs):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30, **kwargs)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Tidak bisa konek ke {url}. Detail asli: {e}"
        ) from e

    if not resp.ok:
        raise RuntimeError(
            f"Server merespons dengan error {resp.status_code}: {resp.text[:300]}"
        )
    return resp


def _post(url, **kwargs):
    try:
        resp = requests.post(url, headers=HEADERS, timeout=30, **kwargs)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Tidak bisa konek ke {url}. Detail asli: {e}"
        ) from e

    if not resp.ok:
        raise RuntimeError(
            f"Server merespons dengan error {resp.status_code}: {resp.text[:300]}"
        )
    return resp


def ambil_data_sensor(limit: int = 1000, sejak_id: int | None = None, mesin_id: int = 1) -> pd.DataFrame:
    params = {"limit": limit, "mesin_id": mesin_id}
    if sejak_id is not None:
        params["sejak_id"] = sejak_id

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
    mesin_id: int = 1,
) -> dict:
    body = {
        "data_id": data_id,
        "prediksi_kondisi": prediksi_kondisi,
        "skor_anomali": skor_anomali,
        "sumber": sumber,
        "keterangan": keterangan,
        "mesin_id": mesin_id,
    }
    resp = _post(f"{API_BASE_URL}/api/hasil-analisis", json=body)
    return resp.json()


def ambil_hasil_terbaru(limit: int = 50, mesin_id: int | None = None) -> pd.DataFrame:
    params = {"limit": limit}
    if mesin_id is not None:
        params["mesin_id"] = mesin_id

    resp = _get(f"{API_BASE_URL}/api/hasil-terbaru", params=params)
    payload = resp.json()
    df = pd.DataFrame(payload["data"])
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def kirim_forecast(daftar_forecast: list, mesin_id: int = 1) -> dict:
    for item in daftar_forecast:
        item.setdefault("mesin_id", mesin_id)
    resp = _post(f"{API_BASE_URL}/api/kirim-forecast", json={"data": daftar_forecast})
    return resp.json()


def ambil_forecast_terbaru(sumber: str | None = None, limit: int = 100, mesin_id: int | None = None) -> pd.DataFrame:
    params = {"limit": limit}
    if sumber:
        params["sumber"] = sumber
    if mesin_id is not None:
        params["mesin_id"] = mesin_id

    resp = _get(f"{API_BASE_URL}/api/forecast-terbaru", params=params)
    payload = resp.json()
    df = pd.DataFrame(payload["data"])
    if not df.empty:
        df["target_waktu"] = pd.to_datetime(df["target_waktu"])
    return df


def hapus_data_sensor(ids: list[int], mesin_id: int = 1) -> dict:
    """Menghapus data sensor berdasarkan ID dari database."""
    body = {
        "ids": ids,
        "mesin_id": mesin_id
    }
    resp = _post(f"{API_BASE_URL}/api/hapus-data-sensor", json=body)
    return resp.json()


if __name__ == "__main__":
    df = ambil_data_sensor(limit=10)
    print(f"Berhasil ambil {len(df)} baris data terbaru:")
    print(df)
