"""
Dashboard Streamlit — Monitoring & Machine Learning/Deep Learning Sensor Mesin.
Jalankan: streamlit run dashboard.py
"""

import json
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from api_client import ambil_data_sensor, ambil_hasil_terbaru, ambil_forecast_terbaru
from pembersihan import audit_data, bersihkan_data

st.set_page_config(page_title="Monitoring Mesin — ML/DL", layout="wide", page_icon="🤖")

# =========================================================================
# CSS — tema gelap
# =========================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .app-header h1 { font-weight: 800; font-size: 2rem; color: #F1F5F9; margin-bottom: 0.1rem; }
    .app-header p { color: #94A3B8; font-size: 0.95rem; margin-top: 0; }

    .section-header {
        font-size: 1.05rem; font-weight: 700; color: #F1F5F9;
        margin: 1.75rem 0 0.75rem 0; display: flex; align-items: center; gap: 0.5rem;
    }
    .badge-dl {
        background: rgba(129, 140, 248, 0.15); color: #A5B4FC;
        padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.7rem; font-weight: 700;
    }
    .badge-ml {
        background: rgba(52, 211, 153, 0.15); color: #6EE7B7;
        padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.7rem; font-weight: 700;
    }

    .metric-card {
        background: #1E293B; border-radius: 14px; padding: 1.1rem 1.3rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2), 0 6px 16px rgba(0,0,0,0.25);
        border-left: 5px solid #818CF8; height: 100%;
    }
    .metric-label {
        font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;
        letter-spacing: 0.06em; font-weight: 600; margin-bottom: 0.3rem;
    }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #F1F5F9; }

    .status-banner {
        border-radius: 14px; padding: 1rem 1.4rem; font-weight: 600; font-size: 1rem;
        margin: 1rem 0 0.5rem 0; display: flex; align-items: center; gap: 0.6rem;
    }
    .status-normal { background: rgba(16,185,129,0.12); color: #34D399; border: 1px solid rgba(52,211,153,0.3); }
    .status-peringatan { background: rgba(245,158,11,0.12); color: #FBBF24; border: 1px solid rgba(251,191,36,0.3); }
    .status-tidaknormal { background: rgba(239,68,68,0.12); color: #F87171; border: 1px solid rgba(248,113,113,0.3); }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important; background-color: #1E293B !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2), 0 6px 16px rgba(0,0,0,0.25);
    }
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


def kartu_metrik(kolom, label, value, warna="#818CF8"):
    kolom.markdown(
        f"""<div class="metric-card" style="border-left-color: {warna};">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div></div>""",
        unsafe_allow_html=True,
    )


def judul_section(teks, tipe):
    badge = '<span class="badge-dl">DEEP LEARNING</span>' if tipe == "dl" else '<span class="badge-ml">MACHINE LEARNING</span>'
    st.markdown(f'<div class="section-header">{teks} {badge}</div>', unsafe_allow_html=True)


def trigger_training_github():
    """Memicu workflow train_model.yml di GitHub Actions dari jarak jauh."""
    token = st.secrets.get("GITHUB_TOKEN")
    repo = st.secrets.get("GITHUB_REPO")  # contoh: "namauser/analisis-mesin"

    if not token or not repo:
        return False, "GITHUB_TOKEN / GITHUB_REPO belum diisi di menu Secrets Streamlit Cloud."

    url = f"https://api.github.com/repos/{repo}/actions/workflows/train_model.yml/dispatches"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    body = {"ref": "main"}

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=15)
        if resp.status_code == 204:
            return True, "Training berhasil dipicu! Cek progresnya di tab Actions repo GitHub kamu."
        return False, f"Gagal memicu training: {resp.status_code} - {resp.text[:200]}"
    except Exception as e:
        return False, f"Error saat memicu training: {e}"


# =========================================================================
# Header
# =========================================================================
st.markdown(
    """<div class="app-header"><h1>🤖 Monitoring Mesin — ML & Deep Learning</h1>
    <p>Prediksi, deteksi anomali, clustering, dan forecasting berbasis machine learning & deep learning</p>
    </div>""",
    unsafe_allow_html=True,
)

st.sidebar.header("⚙️ Pengaturan")
mesin_pilihan = st.sidebar.selectbox(
    "Pilih Mesin",
    options=[1, 2, 3, 4, 5],
    format_func=lambda x: f"Mesin Bubut {x}",
)
jumlah_data = st.sidebar.slider("Jumlah data terbaru", 50, 10000, 500, step=250)
auto_refresh = st.sidebar.checkbox("Auto-refresh tiap 30 detik", value=False)
with st.spinner(f"Mengambil data Mesin Bubut {mesin_pilihan}..."):
    df = ambil_data_sensor(limit=jumlah_data, mesin_id=mesin_pilihan)
    df_hasil = ambil_hasil_terbaru(limit=500, mesin_id=mesin_pilihan)
if df.empty:
    st.warning("Belum ada data.")
    st.stop()

# =========================================================================
# Status terkini (informasi dasar operasional, bukan analisis)
# =========================================================================
data_terakhir = df.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
kartu_metrik(col1, "Suhu Terakhir", f"{data_terakhir['suhu']:.1f} °C", WARNA_ORANYE)
kartu_metrik(col2, "Getaran Terakhir", f"{data_terakhir['kecepatan_getaran']:.2f}", WARNA_BIRU)
kartu_metrik(col3, "Kondisi Terakhir", data_terakhir["kondisi"], WARNA_AKSEN)
kartu_metrik(col4, "Total Data", f"{len(df):,}", WARNA_HIJAU)

status_map = {
    "TIDAK NORMAL": ("status-tidaknormal", "⚠️ Kondisi mesin saat ini TIDAK NORMAL!"),
    "PERINGATAN": ("status-peringatan", "⚠️ Kondisi mesin saat ini PERINGATAN"),
}
kelas, teks = status_map.get(data_terakhir["kondisi"], ("status-normal", "✅ Kondisi mesin NORMAL"))
st.markdown(f'<div class="status-banner {kelas}">{teks}</div>', unsafe_allow_html=True)

if df_hasil.empty:
    st.info("Belum ada hasil ML/DL. Jalankan `train_model.py` (lokal atau GitHub Actions) dulu.")
    st.stop()

# =========================================================================
# Verifikasi & Pembersihan Data
# =========================================================================
judul_section("🧹 Verifikasi & Pembersihan Data", "ml")
with st.container(border=True):
    st.caption(
        "Cek dulu kondisi data sebelum dilatih. Yang dibersihkan HANYA data rusak/tidak valid "
        "(kosong, duplikat, nilai mustahil secara fisik, label tidak baku) — BUKAN outlier "
        "statistik, karena nilai ekstrem yang masih masuk akal justru yang dicari model deteksi anomali."
    )

    col_batas1, col_batas2 = st.columns(2)
    with col_batas1:
        batas_suhu = st.slider("Rentang suhu yang masuk akal (°C)", 0, 300, (0, 150))
    with col_batas2:
        getaran_negatif = st.checkbox("Getaran boleh bernilai negatif?", value=False)

    laporan = audit_data(
        df, batas_suhu_min=batas_suhu[0], batas_suhu_max=batas_suhu[1],
        getaran_boleh_negatif=getaran_negatif,
    )

    if laporan.get("ada_masalah_kritis"):
        st.error(f"⚠️ Data BELUM siap dilatih — ditemukan {laporan['jumlah_masalah_kritis']} baris bermasalah.")
    else:
        st.success("✅ Data sudah siap dilatih — tidak ada masalah kritis ditemukan.")

    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
    col_v1.metric("Nilai Kosong", laporan.get("total_baris_ada_kosong", 0))
    col_v2.metric("Duplikat", laporan.get("baris_duplikat_penuh", 0))
    col_v3.metric("Suhu Tidak Masuk Akal", laporan.get("suhu_tidak_masuk_akal", 0))
    col_v4.metric("Getaran Tidak Masuk Akal", laporan.get("getaran_tidak_masuk_akal", 0))

    col_v5, col_v6, col_v7 = st.columns(3)
    col_v5.metric("Label Tidak Baku", laporan.get("label_tidak_baku", 0))
    col_v6.metric("Celah Waktu Besar (>2 jam)", laporan.get("jumlah_celah_waktu_besar", 0))
    total_outlier = sum(laporan.get("outlier_statistik", {}).values())
    col_v7.metric("Outlier Statistik (info saja)", total_outlier)
    st.caption("Outlier statistik TIDAK dihapus otomatis — cuma informasi, bisa jadi anomali sungguhan.")

    st.markdown("---")
    col_tombol1, col_tombol2 = st.columns(2)

    with col_tombol1:
        if st.button("🧹 Bersihkan Data Sekarang (pratinjau)", use_container_width=True):
            df_bersih, ringkasan_bersih = bersihkan_data(
                df, batas_suhu_min=batas_suhu[0], batas_suhu_max=batas_suhu[1],
                getaran_boleh_negatif=getaran_negatif,
            )
            st.session_state["df_bersih_preview"] = df_bersih
            st.session_state["ringkasan_bersih"] = ringkasan_bersih

    with col_tombol2:
        if st.button("🚀 Latih Model Sekarang (trigger GitHub Actions)", use_container_width=True):
            berhasil, pesan = trigger_training_github()
            if berhasil:
                st.success(pesan)
            else:
                st.error(pesan)

    if "ringkasan_bersih" in st.session_state:
        ringkasan_bersih = st.session_state["ringkasan_bersih"]
        df_bersih_preview = st.session_state["df_bersih_preview"]

        st.markdown("#### Hasil Pratinjau Pembersihan")
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Baris Sebelum", ringkasan_bersih["baris_sebelum"])
        col_r2.metric("Baris Sesudah", ringkasan_bersih["baris_sesudah"])
        col_r3.metric("Total Dihapus", ringkasan_bersih["total_dihapus"])

        st.dataframe(pd.DataFrame([
            {"Alasan": "Nilai kosong", "Jumlah Dihapus": ringkasan_bersih["dihapus_karena_kosong"]},
            {"Alasan": "Duplikat", "Jumlah Dihapus": ringkasan_bersih["dihapus_karena_duplikat"]},
            {"Alasan": "Tidak masuk akal secara fisik", "Jumlah Dihapus": ringkasan_bersih["dihapus_karena_tidak_masuk_akal"]},
            {"Alasan": "Label tidak baku", "Jumlah Dihapus": ringkasan_bersih["dihapus_karena_label_tidak_baku"]},
        ]), use_container_width=True, hide_index=True)

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            fig_banding_suhu = go.Figure()
            fig_banding_suhu.add_trace(go.Histogram(x=df["suhu"], name="Sebelum", opacity=0.5, marker_color=WARNA_ABU))
            fig_banding_suhu.add_trace(go.Histogram(x=df_bersih_preview["suhu"], name="Sesudah", opacity=0.5, marker_color=WARNA_AKSEN))
            fig_banding_suhu.update_layout(title="Suhu: Sebelum vs Sesudah Dibersihkan", barmode="overlay", template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig_banding_suhu, use_container_width=True)
        with col_h2:
            fig_banding_getaran = go.Figure()
            fig_banding_getaran.add_trace(go.Histogram(x=df["kecepatan_getaran"], name="Sebelum", opacity=0.5, marker_color=WARNA_ABU))
            fig_banding_getaran.add_trace(go.Histogram(x=df_bersih_preview["kecepatan_getaran"], name="Sesudah", opacity=0.5, marker_color=WARNA_AKSEN))
            fig_banding_getaran.update_layout(title="Getaran: Sebelum vs Sesudah Dibersihkan", barmode="overlay", template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig_banding_getaran, use_container_width=True)

        st.caption(
            "Catatan: pratinjau ini HANYA untuk melihat efek pembersihan di dashboard. "
            "train_model.py menerapkan pembersihan yang SAMA sebelum melatih model sungguhan."
        )

# =========================================================================
# 1 & 2. Klasifikasi: RandomForest vs MLP
# =========================================================================
judul_section("🌲 Klasifikasi Kondisi: RandomForest", "ml")
with st.container(border=True):
    hasil_rf = df_hasil[df_hasil["sumber"] == "random_forest_klasifikasi_v1"].sort_values("created_at", ascending=False)
    if not hasil_rf.empty:
        st.metric("Prediksi Kondisi Terkini (RandomForest)", hasil_rf.iloc[0]["prediksi_kondisi"])
    else:
        st.info("Belum ada hasil RandomForest.")

judul_section("🧠 Klasifikasi Kondisi: Neural Network (MLP)", "ml")
with st.container(border=True):
    hasil_mlp = df_hasil[df_hasil["sumber"] == "mlp_klasifikasi_v1"].sort_values("created_at", ascending=False)
    if not hasil_mlp.empty:
        st.metric("Prediksi Kondisi Terkini (MLP)", hasil_mlp.iloc[0]["prediksi_kondisi"])
        if not hasil_rf.empty:
            sama = hasil_rf.iloc[0]["prediksi_kondisi"] == hasil_mlp.iloc[0]["prediksi_kondisi"]
            if sama:
                st.success("RandomForest & MLP sepakat pada prediksi yang sama.")
            else:
                st.warning("RandomForest & MLP memberi prediksi BERBEDA — perlu perhatian lebih.")
    else:
        st.info("Belum ada hasil MLP.")

# =========================================================================
# 3 & 4. Deteksi anomali: Isolation Forest vs Autoencoder
# =========================================================================
judul_section("🌲 Deteksi Anomali: Isolation Forest", "ml")
with st.container(border=True):
    anomali_if = df_hasil[df_hasil["sumber"] == "isolation_forest_v1"].sort_values("created_at", ascending=False)
    if not anomali_if.empty:
        st.warning(f"{len(anomali_if)} anomali terdeteksi (Isolation Forest)")
        st.dataframe(anomali_if[["data_id", "skor_anomali", "created_at"]], use_container_width=True)
    else:
        st.success("Tidak ada anomali terdeteksi (Isolation Forest)")

judul_section("🧠 Deteksi Anomali: Autoencoder", "dl")
with st.container(border=True):
    anomali_ae = df_hasil[df_hasil["sumber"] == "autoencoder_v1"].sort_values("created_at", ascending=False)
    if not anomali_ae.empty:
        st.warning(f"{len(anomali_ae)} anomali terdeteksi (Autoencoder)")
        st.dataframe(anomali_ae[["data_id", "skor_anomali", "created_at"]], use_container_width=True)
        if not anomali_if.empty:
            id_sama = set(anomali_if["data_id"]) & set(anomali_ae["data_id"])
            st.caption(f"{len(id_sama)} titik data dianggap anomali oleh KEDUA model — sinyal yang lebih kuat.")
    else:
        st.success("Tidak ada anomali terdeteksi (Autoencoder)")

# =========================================================================
# 5. Clustering
# =========================================================================
judul_section("🎯 Clustering Pola Operasi (K-Means)", "ml")
with st.container(border=True):
    hasil_cluster = df_hasil[df_hasil["sumber"] == "kmeans_cluster_v1"].copy()
    if not hasil_cluster.empty:
        hasil_cluster["cluster"] = hasil_cluster["keterangan"].str.extract(r"Cluster (\d+)").astype(float)
        df_gabung = df.merge(
            hasil_cluster[["data_id", "cluster"]], left_on="id", right_on="data_id", how="inner"
        )
        if not df_gabung.empty:
            fig_cluster = px.scatter(
                df_gabung, x="suhu", y="kecepatan_getaran", color="cluster",
                title="Pengelompokan Pola Operasi Mesin", template=PLOTLY_TEMPLATE,
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
            st.caption(
                "Tiap warna mewakili 'mode operasi' yang ditemukan otomatis oleh K-Means, "
                "tanpa menggunakan label kondisi (NORMAL/PERINGATAN/dst) sama sekali."
            )
        else:
            st.info("Data cluster belum bisa dicocokkan dengan data terbaru yang ditampilkan.")
    else:
        st.info("Belum ada hasil clustering.")

# =========================================================================
# 6. SHAP Explainability
# =========================================================================
judul_section("🔍 Explainability (SHAP)", "ml")
with st.container(border=True):
    st.caption("Menjelaskan fitur mana yang paling memengaruhi prediksi RandomForest.")
    hasil_shap = df_hasil[df_hasil["sumber"] == "shap_importance_v1"].sort_values("created_at", ascending=False)
    if not hasil_shap.empty:
        try:
            importance = json.loads(hasil_shap.iloc[0]["keterangan"])
            fig_shap = px.bar(
                x=list(importance.keys()), y=list(importance.values()),
                title="Pengaruh Fitur terhadap Prediksi (SHAP)",
                labels={"x": "Fitur", "y": "Rata-rata Pengaruh (|SHAP value|)"},
                template=PLOTLY_TEMPLATE, color_discrete_sequence=[WARNA_AKSEN],
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        except (json.JSONDecodeError, TypeError):
            st.info("Data SHAP belum bisa dibaca.")
    else:
        st.info("Belum ada hasil SHAP.")

# =========================================================================
# 7 & 8. Forecasting: ARIMA vs LSTM
# =========================================================================
judul_section("📈 Forecasting Suhu & Getaran: ARIMA vs LSTM", "dl")
with st.container(border=True):
    st.caption("Perbandingan prediksi masa depan antara metode statistik klasik (ARIMA) dan deep learning (LSTM).")

    forecast_arima = ambil_forecast_terbaru(sumber="arima_forecast_v1", limit=100, mesin_id=mesin_pilihan)
    forecast_lstm = ambil_forecast_terbaru(sumber="lstm_forecast_v1", limit=100, mesin_id=mesin_pilihan)

    for kolom, label in [("nilai_suhu_prediksi", "Suhu"), ("nilai_getaran_prediksi", "Getaran")]:
        fig_forecast = go.Figure()

        df_historis = df.tail(48)
        kolom_historis = "suhu" if kolom == "nilai_suhu_prediksi" else "kecepatan_getaran"
        fig_forecast.add_trace(go.Scatter(
            x=df_historis["created_at"], y=df_historis[kolom_historis],
            mode="lines", name="Data Aktual", line=dict(color="#94A3B8"),
        ))

        if not forecast_arima.empty:
            data_arima = forecast_arima.dropna(subset=[kolom])
            if not data_arima.empty:
                fig_forecast.add_trace(go.Scatter(
                    x=data_arima["target_waktu"], y=data_arima[kolom],
                    mode="lines", name="Forecast ARIMA", line=dict(color=WARNA_HIJAU, dash="dash"),
                ))

        if not forecast_lstm.empty:
            data_lstm = forecast_lstm.dropna(subset=[kolom])
            if not data_lstm.empty:
                fig_forecast.add_trace(go.Scatter(
                    x=data_lstm["target_waktu"], y=data_lstm[kolom],
                    mode="lines", name="Forecast LSTM", line=dict(color=WARNA_AKSEN, dash="dot"),
                ))

        fig_forecast.update_layout(title=f"Forecast {label}", template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig_forecast, use_container_width=True)

# ---------- Tabel data mentah ----------
with st.expander("📄 Lihat data mentah"):
    st.dataframe(df.sort_values("created_at", ascending=False), use_container_width=True)

if auto_refresh:
    time.sleep(30)
    st.rerun()
