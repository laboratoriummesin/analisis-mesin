"""
Dashboard Streamlit — Monitoring Mesin dengan ML & DL
Jalankan: streamlit run dashboard.py
"""

from __future__ import annotations

import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from api_client import (
    ambil_data_sensor,
    ambil_hasil_terbaru,
    ambil_forecast_terbaru,
    hapus_data_sensor,
)
from pembersihan import audit_data, bersihkan_data, hapus_data_dari_db


# ============================================================
# Konfigurasi halaman
# ============================================================
st.set_page_config(
    page_title="Monitoring Mesin — ML & DL",
    page_icon="🛠️",
    layout="wide",
)

st.markdown(
    """
    <style>
      div[data-testid="stHorizontalBlock"] { margin-bottom: 0.75rem; }
      div[data-testid="stButton"] > button { white-space: nowrap; }
      .section-title { margin-top: 1.25rem; margin-bottom: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Helper
# ============================================================
def _to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cast kolom ke numeric, aman kalau kolom tidak ada."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _to_dt(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True).dt.tz_convert(None)
    return df


def _bersihkan_duplikat_forecast(df_forecast: pd.DataFrame) -> pd.DataFrame:
    """Rata-ratakan prediksi yang punya target_waktu sama supaya index unik."""
    if df_forecast.empty or "target_waktu" not in df_forecast.columns:
        return df_forecast

    df = df_forecast.copy()
    kolom_numerik = [
        c for c in ("nilai_suhu_prediksi", "nilai_getaran_prediksi") if c in df.columns
    ]
    df = _to_num(df, kolom_numerik)
    df = _to_dt(df, ["target_waktu"])

    if not kolom_numerik:
        return df.drop_duplicates(subset=["target_waktu"])

    return (
        df.dropna(subset=["target_waktu"])
        .groupby("target_waktu", as_index=False)[kolom_numerik]
        .mean()
    )


def _hitung_mae(df_aktual: pd.DataFrame, df_prediksi: pd.DataFrame, kolom: str) -> float | None:
    """Hitung MAE antara nilai aktual dan prediksi berdasarkan waktu."""
    if df_aktual.empty or df_prediksi.empty:
        return None
    if kolom not in df_aktual.columns:
        return None

    kolom_pred = f"{kolom}_prediksi"
    if kolom_pred not in df_prediksi.columns:
        return None

    a = df_aktual[["created_at", kolom]].copy()
    a = _to_dt(a, ["created_at"])
    a = _to_num(a, [kolom]).dropna()
    a = a.set_index("created_at")
    a = a[~a.index.duplicated(keep="last")]

    p = df_prediksi[["target_waktu", kolom_pred]].copy()
    p = _to_dt(p, ["target_waktu"])
    p = _to_num(p, [kolom_pred]).dropna()
    p = p.groupby("target_waktu")[kolom_pred].mean()

    gabungan = pd.concat(
        [a[kolom].rename("aktual"), p.rename("prediksi")],
        axis=1,
        join="inner",
    )
    if gabungan.empty:
        return None

    return float((gabungan["aktual"] - gabungan["prediksi"]).abs().mean())


@st.cache_data(ttl=15)
def _load_sensor(limit: int, mesin_id: int | None) -> pd.DataFrame:
    df = ambil_data_sensor(limit=limit, mesin_id=mesin_id) if mesin_id else ambil_data_sensor(limit=limit)
    if df.empty:
        return df
    df = _to_dt(df, ["created_at"])
    df = _to_num(df, ["nilai_suhu", "nilai_getaran"])
    return df.sort_values("created_at")


@st.cache_data(ttl=15)
def _load_hasil(limit: int) -> pd.DataFrame:
    df = ambil_hasil_terbaru(limit=limit)
    if df.empty:
        return df
    df = _to_dt(df, ["created_at"])
    return df.sort_values("created_at", ascending=False)


@st.cache_data(ttl=15)
def _load_forecast(sumber: str | None, limit: int) -> pd.DataFrame:
    df = ambil_forecast_terbaru(sumber=sumber, limit=limit)
    if df.empty:
        return df
    df = _bersihkan_duplikat_forecast(df)
    return df.sort_values("target_waktu")


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("⚙️ Pengaturan")
    limit = st.slider("Jumlah data terbaru", 50, 2000, 300, step=50)
    mesin_id_input = st.text_input("Filter Mesin ID (opsional)", value="")
    try:
        mesin_id = int(mesin_id_input) if mesin_id_input.strip() else None
    except ValueError:
        st.warning("Mesin ID harus angka. Diabaikan.")
        mesin_id = None

    auto_refresh = st.checkbox("Auto refresh (30 dtk)", value=False)
    if st.button("🔄 Refresh sekarang", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ============================================================
# Ambil data
# ============================================================
try:
    df_sensor = _load_sensor(limit, mesin_id)
    df_hasil = _load_hasil(limit)
    df_arima = _load_forecast("ARIMA", limit)
    df_lstm = _load_forecast("LSTM", limit)
except Exception as e:  # noqa: BLE001
    st.error(f"Gagal memuat data dari API: {e}")
    st.stop()


# ============================================================
# Header + metrik
# ============================================================
st.title("🛠️ Monitoring Mesin — ML & DL")
st.caption("Data sensor, prediksi ARIMA & LSTM, serta riwayat hasil.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total data sensor", f"{len(df_sensor):,}")
m2.metric(
    "Suhu terakhir",
    f"{df_sensor['nilai_suhu'].iloc[-1]:.2f}" if not df_sensor.empty else "-",
)
m3.metric(
    "Getaran terakhir",
    f"{df_sensor['nilai_getaran'].iloc[-1]:.3f}" if not df_sensor.empty else "-",
)

mae_arima = _hitung_mae(df_sensor, df_arima, "nilai_suhu")
mae_lstm = _hitung_mae(df_sensor, df_lstm, "nilai_suhu")
m4.metric(
    "MAE Suhu (ARIMA / LSTM)",
    f"{(mae_arima or 0):.2f} / {(mae_lstm or 0):.2f}",
)


# ============================================================
# Grafik utama
# ============================================================
st.markdown("### 📈 Grafik Sensor & Prediksi", help="Aktual vs ARIMA vs LSTM")

if df_sensor.empty:
    st.info("Belum ada data sensor.")
else:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_sensor["created_at"],
            y=df_sensor["nilai_suhu"],
            name="Aktual (suhu)",
            mode="lines",
        )
    )
    if not df_arima.empty and "nilai_suhu_prediksi" in df_arima.columns:
        fig.add_trace(
            go.Scatter(
                x=df_arima["target_waktu"],
                y=df_arima["nilai_suhu_prediksi"],
                name="ARIMA",
                mode="lines",
                line=dict(dash="dash"),
            )
        )
    if not df_lstm.empty and "nilai_suhu_prediksi" in df_lstm.columns:
        fig.add_trace(
            go.Scatter(
                x=df_lstm["target_waktu"],
                y=df_lstm["nilai_suhu_prediksi"],
                name="LSTM",
                mode="lines",
                line=dict(dash="dot"),
            )
        )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.1),
        xaxis_title="Waktu",
        yaxis_title="Suhu",
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Riwayat Prediksi — tombol di baris sendiri (tidak menutupi teks)
# ============================================================
st.markdown("<h3 class='section-title'>📜 Riwayat Prediksi</h3>", unsafe_allow_html=True)

if "riwayat_filter" not in st.session_state:
    st.session_state.riwayat_filter = "Data Aktual"

c1, c2, c3, _ = st.columns([1, 1, 1, 6])
with c1:
    if st.button("Data Aktual", use_container_width=True, key="btn_aktual"):
        st.session_state.riwayat_filter = "Data Aktual"
with c2:
    if st.button("ARIMA", use_container_width=True, key="btn_arima"):
        st.session_state.riwayat_filter = "ARIMA"
with c3:
    if st.button("LSTM", use_container_width=True, key="btn_lstm"):
        st.session_state.riwayat_filter = "LSTM"

st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
st.caption(f"Menampilkan: **{st.session_state.riwayat_filter}**")

pilihan = st.session_state.riwayat_filter
if pilihan == "Data Aktual":
    df_tampil = df_sensor.sort_values("created_at", ascending=False)
elif pilihan == "ARIMA":
    df_tampil = df_arima.sort_values("target_waktu", ascending=False)
else:
    df_tampil = df_lstm.sort_values("target_waktu", ascending=False)

if df_tampil.empty:
    st.info("Belum ada data untuk filter ini.")
else:
    st.dataframe(df_tampil, use_container_width=True, height=320)


# ============================================================
# Hasil model (klasifikasi/anomaly, dsb.)
# ============================================================
st.markdown("### 🧠 Hasil Model Terbaru")
if df_hasil.empty:
    st.info("Belum ada hasil model.")
else:
    st.dataframe(df_hasil, use_container_width=True, height=280)


# ============================================================
# Audit & pembersihan data
# ============================================================
with st.expander("🧹 Audit & Pembersihan Data", expanded=False):
    if df_sensor.empty:
        st.info("Tidak ada data untuk diaudit.")
    else:
        try:
            audit = audit_data(df_sensor)
            st.json(audit)
        except Exception as e:  # noqa: BLE001
            st.warning(f"Audit gagal: {e}")

        colA, colB = st.columns(2)
        with colA:
            if st.button("Bersihkan (lokal)", use_container_width=True):
                try:
                    df_bersih = bersihkan_data(df_sensor)
                    st.success(f"Data bersih: {len(df_bersih)} baris")
                    st.dataframe(df_bersih.head(50), use_container_width=True)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Gagal membersihkan: {e}")
        with colB:
            ids_str = st.text_input("Hapus ID (pisah koma)", value="")
            if st.button("Hapus dari DB", use_container_width=True, type="primary"):
                try:
                    ids = [int(x) for x in ids_str.split(",") if x.strip()]
                    if not ids:
                        st.warning("Isi minimal 1 ID.")
                    else:
                        resp = hapus_data_sensor(ids=ids, mesin_id=mesin_id or 0)
                        st.success(f"Berhasil dihapus: {resp}")
                        st.cache_data.clear()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Gagal hapus: {e}")


# ============================================================
# Auto refresh
# ============================================================
if auto_refresh:
    time.sleep(30)
    st.cache_data.clear()
    st.rerun()
