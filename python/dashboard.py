"""
Dashboard Streamlit untuk monitoring data sensor mesin secara live.
Jalankan: streamlit run dashboard.py
Lalu buka browser ke alamat yang muncul di terminal (biasanya http://localhost:8501)
"""

import time

import plotly.express as px
import streamlit as st

from api_client import ambil_data_sensor, ambil_hasil_terbaru

st.set_page_config(page_title="Monitoring Mesin", layout="wide")

st.title("🏭 Dashboard Monitoring Sensor Mesin")

# ---------- Sidebar ----------
st.sidebar.header("Pengaturan")
jumlah_data = st.sidebar.slider("Jumlah data terbaru yang ditampilkan", 50, 2000, 500, step=50)
auto_refresh = st.sidebar.checkbox("Auto-refresh tiap 30 detik", value=False)

# ---------- Ambil data ----------
with st.spinner("Mengambil data terbaru dari API..."):
    df = ambil_data_sensor(limit=jumlah_data)

if df.empty:
    st.warning("Belum ada data.")
    st.stop()

# ---------- Ringkasan atas ----------
data_terakhir = df.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Suhu terakhir", f"{data_terakhir['suhu']:.1f} °C")
col2.metric("Getaran terakhir", f"{data_terakhir['kecepatan_getaran']:.2f}")
col3.metric("Kondisi terakhir", data_terakhir["kondisi"])
col4.metric("Total data ditampilkan", len(df))

if data_terakhir["kondisi"] == "TIDAK NORMAL":
    st.error("⚠️ Kondisi mesin saat ini TIDAK NORMAL!")
elif data_terakhir["kondisi"] == "PERINGATAN":
    st.warning("⚠️ Kondisi mesin saat ini PERINGATAN")
else:
    st.success("✅ Kondisi mesin NORMAL")

# ---------- Grafik time series ----------
st.subheader("Tren Suhu & Getaran")
fig1 = px.line(df, x="created_at", y="suhu", title="Suhu terhadap Waktu")
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.line(df, x="created_at", y="kecepatan_getaran", title="Getaran terhadap Waktu")
st.plotly_chart(fig2, use_container_width=True)

# ---------- Scatter & distribusi kondisi ----------
col_a, col_b = st.columns(2)
with col_a:
    fig3 = px.scatter(
        df, x="suhu", y="kecepatan_getaran", color="kondisi",
        title="Suhu vs Getaran per Kondisi"
    )
    st.plotly_chart(fig3, use_container_width=True)

with col_b:
    fig4 = px.pie(df, names="kondisi", title="Distribusi Kondisi")
    st.plotly_chart(fig4, use_container_width=True)

# ---------- Prediksi & anomali terbaru (diambil dari API, hasil dari train_model.py) ----------
st.subheader("Hasil Model Terbaru")
df_hasil = ambil_hasil_terbaru(limit=50)

if df_hasil.empty:
    st.info(
        "Belum ada hasil model. Jalankan `train_model.py` "
        "(lokal atau lewat GitHub Actions terjadwal) supaya ada data di sini."
    )
else:
    prediksi_terbaru = df_hasil[df_hasil["prediksi_kondisi"].notna()].sort_values(
        "created_at", ascending=False
    )
    if not prediksi_terbaru.empty:
        st.info(f"Prediksi kondisi terkini dari model: **{prediksi_terbaru.iloc[0]['prediksi_kondisi']}**")

    anomali_terbaru = df_hasil[df_hasil["skor_anomali"].notna()].sort_values(
        "created_at", ascending=False
    )
    if not anomali_terbaru.empty:
        st.warning(f"{len(anomali_terbaru)} anomali terdeteksi dalam 50 hasil terakhir")
        st.dataframe(anomali_terbaru[["data_id", "skor_anomali", "keterangan", "created_at"]])

# ---------- Tabel data mentah ----------
with st.expander("Lihat data mentah"):
    st.dataframe(df.sort_values("created_at", ascending=False))

# ---------- Auto refresh ----------
if auto_refresh:
    time.sleep(30)
    st.rerun()
