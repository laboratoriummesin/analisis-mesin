"""
Dashboard Streamlit — Monitoring Mesin dengan ML & DL
Jalankan: streamlit run dashboard.py
"""

import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from api_client import (
    ambil_data_sensor, 
    ambil_hasil_terbaru, 
    ambil_forecast_terbaru,
    hapus_data_sensor
)
from pembersihan import audit_data, bersihkan_data, hapus_data_dari_db

st.set_page_config(page_title="Monitoring Mesin — ML/DL", layout="wide", page_icon="🤖")

# =========================================================================
# CSS — Tema Dark Profesional
# =========================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(129, 140, 248, 0.15);
    }
    .main-header h1 {
        font-weight: 700; font-size: 1.75rem; color: #F1F5F9;
        margin: 0; display: flex; align-items: center; gap: 0.75rem;
    }
    .main-header p {
        color: #94A3B8; font-size: 0.9rem; margin: 0.25rem 0 0 0;
    }

    .section-card {
        background: #1E293B;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
        border: 1px solid rgba(129, 140, 248, 0.08);
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    .section-title {
        font-weight: 600; font-size: 1.05rem; color: #F1F5F9;
        margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem;
    }
    .section-title .badge {
        font-size: 0.6rem; font-weight: 700; padding: 0.15rem 0.6rem;
        border-radius: 999px; text-transform: uppercase; letter-spacing: 0.04em;
    }
    .badge-ml { background: rgba(52, 211, 153, 0.15); color: #6EE7B7; }
    .badge-dl { background: rgba(129, 140, 248, 0.15); color: #A5B4FC; }

    .metric-card {
        background: #0F172A;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        border-left: 4px solid #818CF8;
    }
    .metric-label {
        font-size: 0.7rem; color: #94A3B8; text-transform: uppercase;
        letter-spacing: 0.05em; font-weight: 600;
    }
    .metric-value {
        font-size: 1.4rem; font-weight: 700; color: #F1F5F9;
    }
    .metric-sub {
        font-size: 0.75rem; color: #64748B;
    }

    .status-banner {
        border-radius: 10px; padding: 0.75rem 1.25rem;
        font-weight: 600; font-size: 0.95rem;
        display: flex; align-items: center; gap: 0.6rem;
    }
    .status-normal { background: rgba(16,185,129,0.1); color: #34D399; border: 1px solid rgba(52,211,153,0.2); }
    .status-peringatan { background: rgba(245,158,11,0.1); color: #FBBF24; border: 1px solid rgba(251,191,36,0.2); }
    .status-tidaknormal { background: rgba(239,68,68,0.1); color: #F87171; border: 1px solid rgba(248,113,113,0.2); }

    .stButton > button {
        background: #818CF8;
        color: #0F172A;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.4rem 1.2rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #6366F1;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(129, 140, 248, 0.3);
    }

    div[data-testid="stDataFrame"] {
        background: #0F172A;
        border-radius: 8px;
        border: 1px solid #1E293B;
    }

    .cleanup-stats {
        background: #0F172A;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        border: 1px solid #1E293B;
    }
    .cleanup-stats .stat-item {
        display: flex; justify-content: space-between;
        padding: 0.2rem 0; font-size: 0.85rem;
    }
    .cleanup-stats .stat-item .label { color: #94A3B8; }
    .cleanup-stats .stat-item .value { color: #F1F5F9; font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_TEMPLATE = "plotly_dark"
WARNA_AKSEN = "#818CF8"
WARNA_HIJAU = "#34D399"
WARNA_ORANYE = "#F97316"
WARNA_BIRU = "#3B82F6"
WARNA_ABU = "#64748B"


# =========================================================================
# Header
# =========================================================================
st.markdown(
    """
    <div class="main-header">
        <h1>🤖 Monitoring Mesin Bubut</h1>
        <p>Analisis pola operasi dengan K-Means Clustering & Forecasting dengan ARIMA dan LSTM</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================================
# Sidebar
# =========================================================================
st.sidebar.header("⚙️ Pengaturan")
mesin_pilihan = st.sidebar.selectbox(
    "Pilih Mesin",
    options=[1, 2, 3, 4, 5],
    format_func=lambda x: f"Mesin Bubut {x}",
)
jumlah_data = st.sidebar.slider("Jumlah data terbaru", 50, 10000, 500, step=250)
auto_refresh = st.sidebar.checkbox("Auto-refresh tiap 30 detik", value=False)

# =========================================================================
# Ambil Data
# =========================================================================
with st.spinner(f"Mengambil data Mesin Bubut {mesin_pilihan}..."):
    df = ambil_data_sensor(limit=jumlah_data, mesin_id=mesin_pilihan)
    df_hasil = ambil_hasil_terbaru(limit=500, mesin_id=mesin_pilihan)

if df.empty:
    st.warning("Belum ada data sensor untuk mesin ini.")
    st.stop()

# =========================================================================
# Status Terkini
# =========================================================================
data_terakhir = df.iloc[-1]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card" style="border-left-color: {WARNA_ORANYE};">
            <div class="metric-label">🌡️ Suhu Terakhir</div>
            <div class="metric-value">{data_terakhir['suhu']:.1f} °C</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card" style="border-left-color: {WARNA_BIRU};">
            <div class="metric-label">📳 Getaran Terakhir</div>
            <div class="metric-value">{data_terakhir['kecepatan_getaran']:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card" style="border-left-color: {WARNA_AKSEN};">
            <div class="metric-label">📊 Kondisi</div>
            <div class="metric-value">{data_terakhir['kondisi']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card" style="border-left-color: {WARNA_HIJAU};">
            <div class="metric-label">📦 Total Data</div>
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-sub">baris terakhir</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Status Banner
status_map = {
    "TIDAK NORMAL": ("status-tidaknormal", "⚠️ Kondisi mesin TIDAK NORMAL — segera periksa!"),
    "PERINGATAN": ("status-peringatan", "⚠️ Kondisi mesin dalam status PERINGATAN"),
}
kelas, teks = status_map.get(data_terakhir["kondisi"], ("status-normal", "✅ Kondisi mesin NORMAL"))
st.markdown(f'<div class="status-banner {kelas}">{teks}</div>', unsafe_allow_html=True)

# =========================================================================
# Pembersihan Data
# =========================================================================
st.markdown("""
<div class="section-card">
    <div class="section-title">🧹 Pembersihan & Verifikasi Data</div>
</div>
""", unsafe_allow_html=True)

with st.container():
    col_left, col_right = st.columns([3, 2])

    with col_left:
        batas_suhu_min = 27  # Tetap 27 derajat
        batas_suhu_max = st.slider("Batas suhu maksimum (°C)", 50, 300, 150, key="suhu_max")
        getaran_negatif = st.checkbox("Getaran boleh bernilai negatif?", value=False, key="getaran_negatif")
        hapus_outlier = st.checkbox("Hapus outlier statistik (IQR)", value=False, key="hapus_outlier")

        laporan = audit_data(
            df,
            batas_suhu_min=batas_suhu_min,
            batas_suhu_max=batas_suhu_max,
            getaran_boleh_negatif=getaran_negatif,
        )

    with col_right:
        st.markdown("#### 📋 Status Data")
        if laporan.get("ada_masalah_kritis"):
            st.error(f"⚠️ {laporan['jumlah_masalah_kritis']} masalah ditemukan")
        else:
            st.success("✅ Data siap diproses")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Nilai Kosong", laporan.get("total_baris_ada_kosong", 0))
            st.metric("Duplikat", laporan.get("baris_duplikat_penuh", 0))
        with col_b:
            st.metric("Suhu Invalid", laporan.get("suhu_tidak_masuk_akal", 0))
            st.metric("Label Tidak Baku", laporan.get("label_tidak_baku", 0))

# Tombol Pembersihan
col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    if st.button("🔍 Audit Data", use_container_width=True):
        st.toast("Audit selesai — lihat status di atas")

with col_btn2:
    if st.button("🧹 Bersihkan & Hapus dari DB", use_container_width=True):
        with st.spinner("Membersihkan data..."):
            df_bersih, ringkasan = bersihkan_data(
                df,
                batas_suhu_min=27,
                batas_suhu_max=batas_suhu_max,
                getaran_boleh_negatif=getaran_negatif,
                hapus_outlier=hapus_outlier,
            )
            
            # Hapus data yang dibersihkan dari database
            ids_dihapus = list(set(df.index) - set(df_bersih.index))
            if ids_dihapus:
                try:
                    hasil_hapus = hapus_data_dari_db(df, df_bersih, mesin_pilihan)
                    st.success(f"✅ {hasil_hapus['total_dihapus']} baris dihapus dari database")
                except Exception as e:
                    st.error(f"❌ Gagal menghapus dari database: {e}")
            else:
                st.info("Tidak ada data yang perlu dibersihkan")

            st.session_state["ringkasan_bersih"] = ringkasan

with col_btn3:
    if st.button("📊 Lihat Hasil Pembersihan", use_container_width=True):
        if "ringkasan_bersih" in st.session_state:
            r = st.session_state["ringkasan_bersih"]
            st.markdown(f"""
            <div class="cleanup-stats">
                <div class="stat-item"><span class="label">Baris sebelum</span><span class="value">{r['baris_sebelum']}</span></div>
                <div class="stat-item"><span class="label">Baris sesudah</span><span class="value">{r['baris_sesudah']}</span></div>
                <div class="stat-item"><span class="label">Total dihapus</span><span class="value">{r['total_dihapus']}</span></div>
                <div class="stat-item"><span class="label">Kosong</span><span class="value">{r['dihapus_karena_kosong']}</span></div>
                <div class="stat-item"><span class="label">Duplikat</span><span class="value">{r['dihapus_karena_duplikat']}</span></div>
                <div class="stat-item"><span class="label">Tidak masuk akal</span><span class="value">{r['dihapus_karena_tidak_masuk_akal']}</span></div>
                <div class="stat-item"><span class="label">Outlier statistik</span><span class="value">{r.get('dihapus_karena_outlier', 0)}</span></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Belum ada pembersihan yang dilakukan")

# =========================================================================
# K-Means Clustering
# =========================================================================
st.markdown("""
<div class="section-card">
    <div class="section-title">
        🎯 Clustering Pola Operasi
        <span class="badge badge-ml">MACHINE LEARNING</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.container():
    hasil_cluster = df_hasil[df_hasil["sumber"] == "kmeans_cluster_v1"].copy()
    
    if not hasil_cluster.empty:
        hasil_cluster["cluster"] = hasil_cluster["keterangan"].str.extract(r"Cluster (\d+)").astype(float)
        df_gabung = df.merge(
            hasil_cluster[["data_id", "cluster"]],
            left_on="id",
            right_on="data_id",
            how="inner"
        )
        
        if not df_gabung.empty:
            col_c1, col_c2 = st.columns([2, 1])
            
            with col_c1:
                fig_cluster = px.scatter(
                    df_gabung,
                    x="suhu",
                    y="kecepatan_getaran",
                    color="cluster",
                    title="Pengelompokan Pola Operasi Mesin dengan K-Means",
                    template=PLOTLY_TEMPLATE,
                    color_continuous_scale="Viridis",
                    labels={"suhu": "Suhu (°C)", "kecepatan_getaran": "Kecepatan Getaran"},
                )
                fig_cluster.update_layout(
                    plot_bgcolor="#0F172A",
                    paper_bgcolor="#0F172A",
                    legend=dict(bgcolor="#1E293B", bordercolor="#1E293B"),
                )
                st.plotly_chart(fig_cluster, use_container_width=True)
            
            with col_c2:
                st.markdown("#### 📊 Distribusi Cluster")
                dist_cluster = df_gabung["cluster"].value_counts().sort_index()
                for c, count in dist_cluster.items():
                    pct = count / len(df_gabung) * 100
                    st.progress(pct / 100, text=f"Cluster {int(c)}: {count} titik ({pct:.1f}%)")
                
                st.caption(
                    "Tiap warna mewakili 'mode operasi' yang ditemukan otomatis oleh K-Means, "
                    "tanpa menggunakan label kondisi (NORMAL/PERINGATAN/dst) sama sekali."
                )
        else:
            st.info("Data cluster belum bisa dicocokkan dengan data terbaru")
    else:
        st.info("Belum ada hasil clustering. Jalankan `train_model.py` terlebih dahulu.")

# =========================================================================
# Forecasting: ARIMA vs LSTM
# =========================================================================
st.markdown("""
<div class="section-card">
    <div class="section-title">
        📈 Forecasting Suhu & Getaran
        <span class="badge badge-dl">DEEP LEARNING</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.container():
    st.caption("Perbandingan prediksi masa depan antara metode statistik klasik (ARIMA) dan deep learning (LSTM)")

    forecast_arima = ambil_forecast_terbaru(sumber="arima_forecast_v1", limit=100, mesin_id=mesin_pilihan)
    forecast_lstm = ambil_forecast_terbaru(sumber="lstm_forecast_v1", limit=100, mesin_id=mesin_pilihan)

    # Tabs untuk Suhu dan Getaran
    tab_suhu, tab_getaran = st.tabs(["🌡️ Forecast Suhu", "📳 Forecast Getaran"])

    with tab_suhu:
        fig_suhu = go.Figure()
        
        # Data historis
        df_historis = df.tail(48)
        fig_suhu.add_trace(go.Scatter(
            x=df_historis["created_at"],
            y=df_historis["suhu"],
            mode="lines",
            name="Data Aktual",
            line=dict(color="#94A3B8", width=2),
        ))
        
        # ARIMA
        if not forecast_arima.empty:
            data_arima = forecast_arima.dropna(subset=["nilai_suhu_prediksi"])
            if not data_arima.empty:
                fig_suhu.add_trace(go.Scatter(
                    x=data_arima["target_waktu"],
                    y=data_arima["nilai_suhu_prediksi"],
                    mode="lines",
                    name="ARIMA",
                    line=dict(color=WARNA_HIJAU, dash="dash", width=2),
                ))
        
        # LSTM
        if not forecast_lstm.empty:
            data_lstm = forecast_lstm.dropna(subset=["nilai_suhu_prediksi"])
            if not data_lstm.empty:
                fig_suhu.add_trace(go.Scatter(
                    x=data_lstm["target_waktu"],
                    y=data_lstm["nilai_suhu_prediksi"],
                    mode="lines",
                    name="LSTM",
                    line=dict(color=WARNA_AKSEN, dash="dot", width=2),
                ))
        
        fig_suhu.update_layout(
            title="Forecast Suhu — ARIMA vs LSTM",
            template=PLOTLY_TEMPLATE,
            plot_bgcolor="#0F172A",
            paper_bgcolor="#0F172A",
            legend=dict(bgcolor="#1E293B", bordercolor="#1E293B"),
            xaxis_title="Waktu",
            yaxis_title="Suhu (°C)",
        )
        st.plotly_chart(fig_suhu, use_container_width=True)

    with tab_getaran:
        fig_getaran = go.Figure()
        
        # Data historis
        df_historis = df.tail(48)
        fig_getaran.add_trace(go.Scatter(
            x=df_historis["created_at"],
            y=df_historis["kecepatan_getaran"],
            mode="lines",
            name="Data Aktual",
            line=dict(color="#94A3B8", width=2),
        ))
        
        # ARIMA
        if not forecast_arima.empty:
            data_arima = forecast_arima.dropna(subset=["nilai_getaran_prediksi"])
            if not data_arima.empty:
                fig_getaran.add_trace(go.Scatter(
                    x=data_arima["target_waktu"],
                    y=data_arima["nilai_getaran_prediksi"],
                    mode="lines",
                    name="ARIMA",
                    line=dict(color=WARNA_HIJAU, dash="dash", width=2),
                ))
        
        # LSTM
        if not forecast_lstm.empty:
            data_lstm = forecast_lstm.dropna(subset=["nilai_getaran_prediksi"])
            if not data_lstm.empty:
                fig_getaran.add_trace(go.Scatter(
                    x=data_lstm["target_waktu"],
                    y=data_lstm["nilai_getaran_prediksi"],
                    mode="lines",
                    name="LSTM",
                    line=dict(color=WARNA_AKSEN, dash="dot", width=2),
                ))
        
        fig_getaran.update_layout(
            title="Forecast Kecepatan Getaran — ARIMA vs LSTM",
            template=PLOTLY_TEMPLATE,
            plot_bgcolor="#0F172A",
            paper_bgcolor="#0F172A",
            legend=dict(bgcolor="#1E293B", bordercolor="#1E293B"),
            xaxis_title="Waktu",
            yaxis_title="Kecepatan Getaran",
        )
        st.plotly_chart(fig_getaran, use_container_width=True)

    # Tabel ringkasan forecast
    with st.expander("📋 Tabel Forecast Terbaru"):
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("**ARIMA**")
            if not forecast_arima.empty:
                st.dataframe(forecast_arima[["target_waktu", "nilai_suhu_prediksi", "nilai_getaran_prediksi"]].head(10), use_container_width=True)
            else:
                st.info("Belum ada data ARIMA")
        with col_t2:
            st.markdown("**LSTM**")
            if not forecast_lstm.empty:
                st.dataframe(forecast_lstm[["target_waktu", "nilai_suhu_prediksi", "nilai_getaran_prediksi"]].head(10), use_container_width=True)
            else:
                st.info("Belum ada data LSTM")

# =========================================================================
# Data Mentah
# =========================================================================
with st.expander("📄 Lihat Data Mentah"):
    st.dataframe(df.sort_values("created_at", ascending=False), use_container_width=True)

# Auto-refresh
if auto_refresh:
    time.sleep(30)
    st.rerun()
