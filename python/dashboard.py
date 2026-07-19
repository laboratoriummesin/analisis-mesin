"""
Dashboard Streamlit — Monitoring Mesin dengan ML & DL
Jalankan: streamlit run dashboard.py
"""

import json
import time
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests

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

    /* ---------------------------------------------------------------
       KARTU UNTUK SETIAP SECTION.
       Sebelumnya tiap "card" cuma dibuat dari div HTML yang dibuka lalu
       langsung ditutup lewat st.markdown() — jadi widget Streamlit yang
       muncul SESUDAHNYA (slider, chart, tabel, dll) sebenarnya tidak
       pernah ada DI DALAM kotak itu, makanya tampilannya berantakan.
       Sekarang tiap card memakai st.container(border=True), yang benar-
       benar membungkus semua widget di dalamnya jadi satu kotak utuh,
       dan style di bawah ini yang mengatur tampilan kotaknya.
       --------------------------------------------------------------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #1E293B;
        border-radius: 12px !important;
        border: 1px solid rgba(129, 140, 248, 0.12) !important;
        padding: 1.1rem 1.4rem 1.4rem 1.4rem;
        margin-bottom: 1.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }

    .card-title {
        font-weight: 600;
        font-size: 1.05rem;
        color: #F1F5F9;
        margin-bottom: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .card-title .badge {
        font-size: 0.6rem;
        font-weight: 700;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
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

    /* ---------------------------------------------------------------
       TOMBOL SERAGAM.
       Semua tombol (di semua card) dipaksa punya tinggi, padding, dan
       ukuran font yang sama persis, supaya tidak ada tombol yang
       terlihat "nanggung" lebih besar/kecil dari yang lain — termasuk
       saat teks labelnya panjangnya beda-beda (mis. "Audit Data" vs
       "Latih Model (GitHub Actions)").
       --------------------------------------------------------------- */
    .stButton > button {
        background: #818CF8;
        color: #0F172A;
        font-weight: 600;
        font-size: 0.85rem;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1rem;
        min-height: 2.6rem;
        width: 100%;
        white-space: normal;
        line-height: 1.15;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #6366F1;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(129, 140, 248, 0.3);
    }
    .stButton > button p {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
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

    .kondisi-badge {
        display: inline-flex; align-items: center; gap: 0.4rem;
        border-radius: 8px; padding: 0.6rem 1rem; font-weight: 600;
        font-size: 0.9rem; width: 100%; box-sizing: border-box;
    }
    .kondisi-normal { background: rgba(16,185,129,0.1); color: #34D399; border: 1px solid rgba(52,211,153,0.25); }
    .kondisi-tidaknormal { background: rgba(239,68,68,0.1); color: #F87171; border: 1px solid rgba(248,113,113,0.25); }
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

# Ambang persentase outlier: di atas ini data dianggap "Tidak Normal" secara statistik
AMBANG_PERSEN_TIDAK_NORMAL = 5.0

GITHUB_API_BASE = "https://api.github.com"
NAMA_WORKFLOW = "train_model.yml"


# =========================================================================
# TRIGGER + MONITOR TRAINING VIA GITHUB ACTIONS
# =========================================================================
def _github_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }


def trigger_training_github(token, repo):
    """Memicu workflow training. Mengembalikan (berhasil, pesan, waktu_trigger_utc)."""
    url = f"{GITHUB_API_BASE}/repos/{repo}/actions/workflows/{NAMA_WORKFLOW}/dispatches"
    waktu_trigger = datetime.now(timezone.utc)
    try:
        resp = requests.post(url, headers=_github_headers(token), json={"ref": "main"}, timeout=15)
        if resp.status_code == 204:
            return True, "Training berhasil dipicu di GitHub Actions.", waktu_trigger
        return False, f"Gagal memicu training: {resp.status_code} - {resp.text[:200]}", waktu_trigger
    except Exception as e:
        return False, f"Error saat memicu training: {e}", waktu_trigger


def cari_run_terbaru(token, repo, sejak_waktu, percobaan=8, jeda_detik=3):
    """Cari run workflow yang dibuat SETELAH waktu trigger. Dicoba beberapa kali karena
    GitHub butuh beberapa detik untuk mendaftarkan run baru setelah dispatch."""
    url = f"{GITHUB_API_BASE}/repos/{repo}/actions/workflows/{NAMA_WORKFLOW}/runs"
    for _ in range(percobaan):
        try:
            resp = requests.get(url, headers=_github_headers(token), params={"per_page": 5}, timeout=15)
            if resp.status_code == 200:
                runs = resp.json().get("workflow_runs", [])
                for run in runs:
                    waktu_run = pd.to_datetime(run["created_at"])
                    if waktu_run.tz_localize(None) >= pd.Timestamp(sejak_waktu).tz_localize(None) - pd.Timedelta(seconds=5):
                        return run
        except Exception:
            pass
        time.sleep(jeda_detik)
    return None


def ambil_detail_run(token, repo, run_id):
    url = f"{GITHUB_API_BASE}/repos/{repo}/actions/runs/{run_id}"
    try:
        resp = requests.get(url, headers=_github_headers(token), timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def jalankan_dan_pantau_training():
    """Alur lengkap: trigger -> deteksi run -> pantau status live -> notifikasi akhir."""
    token = st.secrets.get("GITHUB_TOKEN")
    repo = st.secrets.get("GITHUB_REPO")

    if not token or not repo:
        st.error("❌ GITHUB_TOKEN / GITHUB_REPO belum diisi di menu Secrets Streamlit Cloud.")
        return

    with st.status("🚀 Memicu training di GitHub Actions...", expanded=True) as status:
        berhasil, pesan, waktu_trigger = trigger_training_github(token, repo)
        if not berhasil:
            status.update(label=f"❌ {pesan}", state="error", expanded=True)
            return

        status.write(f"✅ {pesan}")
        status.write("⏳ Menunggu GitHub mendaftarkan run baru...")

        run = cari_run_terbaru(token, repo, waktu_trigger)
        if run is None:
            status.update(
                label="⚠️ Training terpicu, tapi run belum terdeteksi otomatis. Cek tab Actions repo GitHub secara manual.",
                state="error", expanded=True,
            )
            return

        url_run = run.get("html_url")
        status.write(f"🔗 Run terdeteksi: [buka di GitHub Actions]({url_run})")

        maks_polling = 100  # ~ 100 x 6 detik = 10 menit
        conclusion = None
        for i in range(maks_polling):
            detail = ambil_detail_run(token, repo, run["id"])
            if detail is None:
                status.write("⚠️ Gagal mengambil status terbaru, mencoba lagi...")
                time.sleep(6)
                continue

            status_run = detail.get("status")  # queued / in_progress / completed
            conclusion = detail.get("conclusion")  # success / failure / cancelled / None

            if status_run == "completed":
                break

            label_status = {"queued": "menunggu antrian", "in_progress": "sedang berjalan"}.get(status_run, status_run)
            status.write(f"⏳ Status training: **{label_status}** (cek ke-{i + 1})")
            time.sleep(6)
        else:
            status.update(
                label=f"⏱️ Training masih berjalan setelah 10 menit dipantau. Silakan cek langsung di [GitHub Actions]({url_run}).",
                state="error", expanded=True,
            )
            return

        if conclusion == "success":
            status.update(label="✅ Training selesai — semua model berhasil diperbarui!", state="complete", expanded=False)
            st.success(f"✅ Training selesai dengan sukses. [Lihat detail run]({url_run})")
        else:
            status.update(label=f"❌ Training selesai dengan status: {conclusion}.", state="error", expanded=True)
            st.error(f"❌ Training gagal ({conclusion}). [Lihat log run]({url_run})")


def _ambil_kolom_target_waktu(df_forecast):
    """Ambil kolom target_waktu dengan aman. Kalau mesin belum punya forecast
    sama sekali, ambil_forecast_terbaru bisa mengembalikan DataFrame kosong
    TANPA kolom target_waktu sama sekali — jadi df[["target_waktu"]] akan error
    (KeyError). Fungsi ini menghindari itu."""
    if df_forecast.empty or "target_waktu" not in df_forecast.columns:
        return pd.DataFrame(columns=["target_waktu"])
    return df_forecast[["target_waktu"]]


def __duplikat_forecast(df_forecast):
    if df_forecast.empty or "target_waktu" not in df_forecast.columns:
        return df_forecast

    kolom_numerik = [c for c in ["nilai_suhu_prediksi", "nilai_getaran_prediksi"]
                     if c in df_forecast.columns]
    if not kolom_numerik:
        return df_forecast

    df_forecast = df_forecast.copy()
    # Kolom prediksi dari API bisa berupa string -> paksa jadi numerik dulu
    for c in kolom_numerik:
        df_forecast[c] = pd.to_numeric(df_forecast[c], errors="coerce")

    # Samakan tipe target_waktu supaya duplikat benar-benar tergabung
    df_forecast["target_waktu"] = pd.to_datetime(df_forecast["target_waktu"], errors="coerce")
    df_forecast = df_forecast.dropna(subset=["target_waktu"])

    return (
        df_forecast.groupby("target_waktu", as_index=False)[kolom_numerik]
        .mean()
        .sort_values("target_waktu")
        .reset_index(drop=True)
    )


def _hitung_mae(df_historis, forecast_df, kolom_aktual, kolom_prediksi):
    """Hitung MAE untuk jam-jam yang tumpang tindih antara aktual & forecast."""
    if df_historis.empty or forecast_df.empty:
        return None
    if kolom_prediksi not in forecast_df.columns or "target_waktu" not in forecast_df.columns:
        return None

    try:
        aktual_per_jam = (
            df_historis.set_index("created_at")[kolom_aktual]
            .resample("1h").mean()
        )
    except Exception:
        return None

    # Pastikan index aktual unik (resample biasanya sudah unik, tapi jaga-jaga
    # kalau ada NaT / tz mixed)
    aktual_per_jam = aktual_per_jam[~aktual_per_jam.index.duplicated(keep="last")]
    aktual_per_jam = aktual_per_jam.dropna()

    prediksi = forecast_df.dropna(subset=[kolom_prediksi]).copy()
    if prediksi.empty:
        return None

    # Normalisasi target_waktu ke datetime tz-naive supaya konsisten dengan aktual
    prediksi["target_waktu"] = pd.to_datetime(prediksi["target_waktu"], errors="coerce")
    prediksi = prediksi.dropna(subset=["target_waktu"])
    if prediksi.empty:
        return None
    try:
        prediksi["target_waktu"] = prediksi["target_waktu"].dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass  # sudah tz-naive

    # Rata-ratakan duplikat target_waktu -> index dijamin unik
    prediksi = (
        prediksi.groupby("target_waktu", as_index=True)[kolom_prediksi].mean()
    )

    # Samakan tz aktual juga (kalau tz-aware)
    try:
        aktual_per_jam.index = aktual_per_jam.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass

    gabungan = pd.concat(
        [aktual_per_jam.rename("aktual"), prediksi.rename("prediksi")],
        axis=1,
        join="inner",
    )
    if gabungan.empty:
        return None

    return float((gabungan["aktual"] - gabungan["prediksi"]).abs().mean())


def _buat_grafik_forecast(df_historis, forecast_arima, forecast_lstm, kolom_aktual, kolom_prediksi,
                           judul, label_sumbu_y, tanggal_str, is_backtest):
    """Grafik forecast yang lebih mudah dibaca:
    - Warna solid + bentuk marker berbeda per garis (bukan dash/dot yang mudah membaur)
    - Garis vertikal penanda batas antara data aktual & forecast
    - Grid halus supaya lebih mudah membaca nilai
    - Hover yang menampilkan nilai per garis dengan jelas
    """
    fig = go.Figure()

    # --- Data Aktual ---
    if not df_historis.empty:
        fig.add_trace(go.Scatter(
            x=df_historis["created_at"],
            y=df_historis[kolom_aktual],
            mode="lines+markers",
            name="Data Aktual",
            line=dict(color="#E2E8F0", width=3),
            marker=dict(size=6, color="#E2E8F0", symbol="circle"),
            hovertemplate="Aktual: %{y:.2f}<extra></extra>",
        ))

    # --- ARIMA ---
    data_arima = forecast_arima.dropna(subset=[kolom_prediksi]) if not forecast_arima.empty else forecast_arima
    if not data_arima.empty:
        fig.add_trace(go.Scatter(
            x=data_arima["target_waktu"],
            y=data_arima[kolom_prediksi],
            mode="lines+markers",
            name="ARIMA",
            line=dict(color="#F59E0B", width=3),
            marker=dict(size=7, color="#F59E0B", symbol="diamond"),
            hovertemplate="ARIMA: %{y:.2f}<extra></extra>",
        ))

    # --- LSTM ---
    data_lstm = forecast_lstm.dropna(subset=[kolom_prediksi]) if not forecast_lstm.empty else forecast_lstm
    if not data_lstm.empty:
        fig.add_trace(go.Scatter(
            x=data_lstm["target_waktu"],
            y=data_lstm[kolom_prediksi],
            mode="lines+markers",
            name="LSTM",
            line=dict(color="#8B5CF6", width=3),
            marker=dict(size=7, color="#8B5CF6", symbol="square"),
            hovertemplate="LSTM: %{y:.2f}<extra></extra>",
        ))

    # --- Garis penanda "mulai forecast" (kalau data aktual & forecast sama-sama ada) ---
    if not df_historis.empty and (not data_arima.empty or not data_lstm.empty):
        waktu_mulai_forecast = min(
            [d["target_waktu"].min() for d in [data_arima, data_lstm] if not d.empty]
        )
        fig.add_vline(
            x=waktu_mulai_forecast,
            line_width=2,
            line_dash="dash",
            line_color="#64748B",
            annotation_text="Mulai Forecast",
            annotation_position="top",
            annotation_font_size=11,
            annotation_font_color="#94A3B8",
        )

    label_tipe = "🕓 Riwayat (bisa dibandingkan dengan data aktual)" if is_backtest else "🔮 Prakiraan (belum terjadi)"

    fig.update_layout(
        title=dict(
            text=f"{judul} — {tanggal_str}<br><sup>{label_tipe}</sup>",
            font=dict(size=18),
        ),
        template=PLOTLY_TEMPLATE,
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(size=13, color="#E2E8F0"),
        legend=dict(
            bgcolor="#1E293B",
            bordercolor="#334155",
            borderwidth=1,
            orientation="h",
            yanchor="bottom", y=1.05,
            xanchor="left", x=0,
        ),
        xaxis=dict(
            title="Waktu",
            showgrid=True, gridcolor="#1E293B", gridwidth=1,
            rangeslider=dict(visible=True, thickness=0.06),
        ),
        yaxis=dict(
            title=label_sumbu_y,
            showgrid=True, gridcolor="#1E293B", gridwidth=1,
        ),
        hovermode="x unified",
        margin=dict(t=90),
        height=480,
    )
    return fig


def _tampilkan_metrik_akurasi(df_historis, forecast_arima, forecast_lstm, kolom_aktual, kolom_prediksi, format_desimal):
    mae_arima = _hitung_mae(df_historis, forecast_arima, kolom_aktual, kolom_prediksi)
    mae_lstm = _hitung_mae(df_historis, forecast_lstm, kolom_aktual, kolom_prediksi)

    if mae_arima is None and mae_lstm is None:
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rata-rata Selisih ARIMA (MAE)", f"{mae_arima:{format_desimal}}" if mae_arima is not None else "—")
    with col2:
        st.metric("Rata-rata Selisih LSTM (MAE)", f"{mae_lstm:{format_desimal}}" if mae_lstm is not None else "—")
    with col3:
        if mae_arima is not None and mae_lstm is not None:
            lebih_akurat = "ARIMA" if mae_arima < mae_lstm else "LSTM"
            st.metric("Lebih Akurat Hari Ini", lebih_akurat)
    st.caption("MAE (Mean Absolute Error) = rata-rata selisih absolut antara nilai prediksi dan nilai aktual pada jam yang sama. Semakin kecil, semakin akurat.")


# =========================================================================
# HEADER
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
# SIDEBAR
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
# AMBIL DATA
# =========================================================================
with st.spinner(f"Mengambil data Mesin Bubut {mesin_pilihan}..."):
    df = ambil_data_sensor(limit=jumlah_data, mesin_id=mesin_pilihan)

    # PENTING (perbaikan jumlah data clustering): tabel hasil analisis berisi
    # banyak jenis sumber sekaligus (rf, mlp, isolation forest, autoencoder,
    # shap, kmeans...). Kalau limit-nya kecil (dulu 500), baris "kmeans_cluster_v1"
    # bisa "terdesak" keluar oleh baris jenis lain, sehingga jumlah cluster yang
    # tampil jauh lebih sedikit dari jumlah data mentah. Limit di sini dinaikkan
    # supaya cukup menampung seluruh baris kmeans untuk jumlah_data yang dipilih.
    limit_hasil = min(max(jumlah_data * 2, 1000), 8000)
    df_hasil = ambil_hasil_terbaru(limit=limit_hasil, mesin_id=mesin_pilihan)

if df.empty:
    st.warning("Belum ada data sensor untuk mesin ini.")
    st.stop()

# =========================================================================
# CARD 1: STATUS TERKINI
# =========================================================================
with st.container(border=True):
    st.markdown('<div class="card-title">📊 Status Terkini Mesin</div>', unsafe_allow_html=True)

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

    status_map = {
        "TIDAK NORMAL": ("status-tidaknormal", "⚠️ Kondisi mesin TIDAK NORMAL — segera periksa!"),
        "PERINGATAN": ("status-peringatan", "⚠️ Kondisi mesin dalam status PERINGATAN"),
    }
    kelas, teks = status_map.get(data_terakhir["kondisi"], ("status-normal", "✅ Kondisi mesin NORMAL"))
    st.markdown(f'<div class="status-banner {kelas}">{teks}</div>', unsafe_allow_html=True)

# =========================================================================
# CARD 2: PEMBERSIHAN & VERIFIKASI DATA
# (termasuk sebaran outlier & tabel detail outlier — semua jadi satu card)
# =========================================================================
with st.container(border=True):
    st.markdown('<div class="card-title">🧹 Pembersihan & Verifikasi Data</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        batas_suhu_min = 27
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

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Nilai Kosong", laporan.get("total_baris_ada_kosong", 0))
            st.metric("Duplikat", laporan.get("baris_duplikat_penuh", 0))
        with col_b:
            st.metric("Suhu Invalid", laporan.get("suhu_tidak_masuk_akal", 0))
            st.metric("Label Tidak Baku", laporan.get("label_tidak_baku", 0))
        with col_c:
            outlier_stats = laporan.get("outlier_statistik", {})
            st.metric("Outlier Suhu", outlier_stats.get("suhu", 0))
            st.metric("Outlier Getaran", outlier_stats.get("kecepatan_getaran", 0))

    st.divider()

    # Tombol Pembersihan & Training (ukuran seragam diatur lewat CSS .stButton di atas)
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

    with col_btn1:
        if st.button("🔍 Audit Data", width="stretch", key="btn_audit_data"):
            st.toast("Audit selesai — lihat status di atas", icon="🔍")

    with col_btn2:
        if st.button("🧹 Bersihkan", width="stretch", key="btn_cleanup_db"):
            with st.spinner("Membersihkan data..."):
                df_bersih, ringkasan = bersihkan_data(
                    df,
                    batas_suhu_min=27,
                    batas_suhu_max=batas_suhu_max,
                    getaran_boleh_negatif=getaran_negatif,
                    hapus_outlier=hapus_outlier,
                )

                if len(df_bersih) < len(df):
                    try:
                        hasil_hapus = hapus_data_dari_db(df, df_bersih, mesin_pilihan)
                        if hasil_hapus['total_dihapus'] > 0:
                            st.toast(f"✅ {hasil_hapus['total_dihapus']} baris dihapus dari database", icon="🧹")
                        else:
                            st.toast("Tidak ada data yang dihapus dari database", icon="ℹ️")
                    except Exception as e:
                        st.error(f"❌ Gagal menghapus dari database: {e}")
                        st.info("Data tetap dibersihkan di tampilan, tapi tidak dihapus dari database")
                else:
                    st.toast("Tidak ada data yang perlu dibersihkan", icon="ℹ️")

                st.session_state["ringkasan_bersih"] = ringkasan
                st.session_state["df_bersih_preview"] = df_bersih

    with col_btn3:
        if st.button("📊 Lihat Hasil Pembersihan", width="stretch", key="btn_view_cleanup"):
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

    with col_btn4:
        # Tombol training: memicu training DAN memantau progresnya secara live
        # (deteksi run baru -> status berjalan -> notifikasi sukses/gagal di akhir).
        if st.button("🚀 Latih Model", width="stretch", key="btn_train_model"):
            st.toast("Permintaan training terkirim, memantau progresnya...", icon="🚀")
            jalankan_dan_pantau_training()

    # ------------------------------------------------------------
    # SCATTER PLOT OUTLIER (bagian dari card yang sama)
    # ------------------------------------------------------------
    outlier_data_detail = laporan.get("outlier_data_detail", {})

    df_temp = df.copy()
    df_temp["is_outlier_suhu"] = False
    df_temp["is_outlier_getaran"] = False

    data_suhu = outlier_data_detail.get("suhu", [])
    if data_suhu:
        ids_suhu = [d["id"] for d in data_suhu]
        df_temp["is_outlier_suhu"] = df_temp["id"].isin(ids_suhu)

    data_getaran = outlier_data_detail.get("kecepatan_getaran", [])
    if data_getaran:
        ids_getaran = [d["id"] for d in data_getaran]
        df_temp["is_outlier_getaran"] = df_temp["id"].isin(ids_getaran)

    if not df_temp.empty:
        outlier_stats = laporan.get("outlier_statistik", {})

        st.divider()
        st.markdown("#### 📊 Sebaran Data dengan Outlier")
        st.caption(
            "Sumbu grafik getaran sengaja dibalik (suhu di sumbu-Y) supaya kedua grafik "
            "tidak terlihat identik, dan masing-masing dilengkapi garis regresi linear (OLS)."
        )

        col_plot1, col_plot2 = st.columns(2)

        persen_outlier_suhu = (outlier_stats.get("suhu", 0) / len(df_temp) * 100) if len(df_temp) else 0
        persen_outlier_getaran = (outlier_stats.get("kecepatan_getaran", 0) / len(df_temp) * 100) if len(df_temp) else 0

        with col_plot1:
            fig_outlier_suhu = px.scatter(
                df_temp,
                x="suhu",
                y="kecepatan_getaran",
                color="is_outlier_suhu",
                color_discrete_map={False: "#3B82F6", True: "#EF4444"},
                title="🔴 Outlier Suhu (Tanda Merah)",
                template=PLOTLY_TEMPLATE,
                labels={
                    "suhu": "Suhu (°C)",
                    "kecepatan_getaran": "Kecepatan Getaran",
                    "is_outlier_suhu": "Status"
                },
                hover_data=["id", "created_at", "kondisi"],
                trendline="ols",
                trendline_scope="overall",
                trendline_color_override="#FBBF24",
            )
            fig_outlier_suhu.update_layout(
                plot_bgcolor="#0F172A",
                paper_bgcolor="#0F172A",
                legend=dict(bgcolor="#1E293B", bordercolor="#1E293B", title="Status"),
                title_font=dict(size=14),
            )
            fig_outlier_suhu.update_traces(marker=dict(size=8, opacity=0.8), selector=dict(mode="markers"))
            st.plotly_chart(fig_outlier_suhu, width="stretch")

            if persen_outlier_suhu > AMBANG_PERSEN_TIDAK_NORMAL:
                st.markdown(
                    f'<div class="kondisi-badge kondisi-tidaknormal">⚠️ Tidak Normal — {outlier_stats.get("suhu", 0)} titik outlier ({persen_outlier_suhu:.1f}% dari data)</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="kondisi-badge kondisi-normal">✅ Normal — {outlier_stats.get("suhu", 0)} titik outlier ({persen_outlier_suhu:.1f}% dari data)</div>',
                    unsafe_allow_html=True,
                )

        with col_plot2:
            fig_outlier_getaran = px.scatter(
                df_temp,
                x="kecepatan_getaran",
                y="suhu",
                color="is_outlier_getaran",
                color_discrete_map={False: "#3B82F6", True: "#EF4444"},
                title="📳 Outlier Getaran (Tanda Merah) — Sumbu Dibalik",
                template=PLOTLY_TEMPLATE,
                labels={
                    "suhu": "Suhu (°C)",
                    "kecepatan_getaran": "Kecepatan Getaran",
                    "is_outlier_getaran": "Status"
                },
                hover_data=["id", "created_at", "kondisi"],
                trendline="ols",
                trendline_scope="overall",
                trendline_color_override="#FBBF24",
            )
            fig_outlier_getaran.update_layout(
                plot_bgcolor="#0F172A",
                paper_bgcolor="#0F172A",
                legend=dict(bgcolor="#1E293B", bordercolor="#1E293B", title="Status"),
                title_font=dict(size=14),
            )
            fig_outlier_getaran.update_traces(marker=dict(size=8, opacity=0.8), selector=dict(mode="markers"))
            st.plotly_chart(fig_outlier_getaran, width="stretch")

            if persen_outlier_getaran > AMBANG_PERSEN_TIDAK_NORMAL:
                st.markdown(
                    f'<div class="kondisi-badge kondisi-tidaknormal">⚠️ Tidak Normal — {outlier_stats.get("kecepatan_getaran", 0)} titik outlier ({persen_outlier_getaran:.1f}% dari data)</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="kondisi-badge kondisi-normal">✅ Normal — {outlier_stats.get("kecepatan_getaran", 0)} titik outlier ({persen_outlier_getaran:.1f}% dari data)</div>',
                    unsafe_allow_html=True,
                )

    # Tabel detail outlier
    if outlier_data_detail and any(outlier_data_detail.values()):
        with st.expander("📋 Lihat Tabel Data Outlier", expanded=False):
            col_outlier_1, col_outlier_2 = st.columns(2)

            with col_outlier_1:
                data_suhu = outlier_data_detail.get("suhu", [])
                if data_suhu:
                    st.markdown(f"#### 🌡️ Outlier Suhu — {len(data_suhu)} data")
                    df_suhu = pd.DataFrame(data_suhu)
                    if "created_at" in df_suhu.columns:
                        df_suhu["created_at"] = pd.to_datetime(df_suhu["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                    st.dataframe(
                        df_suhu[["id", "created_at", "suhu"]],
                        width="stretch",
                        column_config={
                            "id": "ID",
                            "created_at": "Waktu",
                            "suhu": st.column_config.NumberColumn("Suhu (°C)", format="%.2f"),
                        },
                        hide_index=True,
                    )
                else:
                    st.success("✅ Tidak ada outlier suhu")

            with col_outlier_2:
                data_getaran = outlier_data_detail.get("kecepatan_getaran", [])
                if data_getaran:
                    st.markdown(f"#### 📳 Outlier Getaran — {len(data_getaran)} data")
                    df_getaran = pd.DataFrame(data_getaran)
                    if "created_at" in df_getaran.columns:
                        df_getaran["created_at"] = pd.to_datetime(df_getaran["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                    st.dataframe(
                        df_getaran[["id", "created_at", "kecepatan_getaran"]],
                        width="stretch",
                        column_config={
                            "id": "ID",
                            "created_at": "Waktu",
                            "kecepatan_getaran": st.column_config.NumberColumn("Kecepatan Getaran", format="%.4f"),
                        },
                        hide_index=True,
                    )
                else:
                    st.success("✅ Tidak ada outlier getaran")

# =========================================================================
# CARD 3: CLUSTERING POLA OPERASI (K-MEANS)
# =========================================================================
with st.container(border=True):
    st.markdown(
        '<div class="card-title">🎯 Clustering Pola Operasi <span class="badge badge-ml">MACHINE LEARNING</span></div>',
        unsafe_allow_html=True,
    )

    hasil_cluster = df_hasil[df_hasil["sumber"] == "kmeans_cluster_v1"].copy()

    if not hasil_cluster.empty:
        hasil_cluster["cluster"] = hasil_cluster["keterangan"].str.extract(r"Cluster (\d+)").astype(float)
        # drop_duplicates: kalau training pernah dijalankan berkali-kali dan API
        # menyimpan riwayat (bukan overwrite), pastikan tiap data_id cuma dihitung 1x
        hasil_cluster = hasil_cluster.sort_values("data_id").drop_duplicates(subset=["data_id"], keep="last")

        df_gabung = df.merge(
            hasil_cluster[["data_id", "cluster"]],
            left_on="id",
            right_on="data_id",
            how="inner"
        )

        jumlah_tercakup = len(df_gabung)
        jumlah_total = len(df)
        if jumlah_tercakup < jumlah_total:
            st.warning(
                f"⚠️ Menampilkan {jumlah_tercakup:,} dari {jumlah_total:,} baris data terbaru. "
                f"Sisanya belum punya hasil clustering — jalankan 'Latih Model' lagi untuk mencakup seluruh data terbaru."
            )
        else:
            st.caption(f"✅ Seluruh {jumlah_total:,} baris data terbaru sudah tercakup dalam clustering ini.")

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
                st.plotly_chart(fig_cluster, width="stretch")

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
        st.info("Belum ada hasil clustering. Jalankan 'Latih Model' terlebih dahulu.")

# =========================================================================
# CARD 4: FORECASTING (ARIMA vs LSTM) — dengan pilihan HORIZON PREDIKSI
# =========================================================================
with st.container(border=True):
    st.markdown(
        '<div class="card-title">📈 Forecasting Suhu & Getaran <span class="badge badge-dl">DEEP LEARNING</span></div>',
        unsafe_allow_html=True,
    )

    st.caption("Perbandingan prediksi masa depan antara metode statistik klasik (ARIMA) dan deep learning (LSTM)")

    # Horizon pendek (30m/1j/2j) memakai forecast resolusi 15 menit yang dibuat
    # khusus (sumber *_pendek_v1). Horizon panjang (6/12/24j) memakai forecast
    # per-jam yang sudah ada, dipotong sesuai jumlah jam yang dipilih.
    HORIZON_OPTIONS = {
        "30 Menit":  {"mode": "pendek", "langkah": 2},
        "1 Jam":     {"mode": "pendek", "langkah": 4},
        "2 Jam":     {"mode": "pendek", "langkah": 8},
        "6 Jam":     {"mode": "biasa",  "langkah": 6},
        "12 Jam":    {"mode": "biasa",  "langkah": 12},
        "24 Jam":    {"mode": "biasa",  "langkah": 24},
    }

    col_horizon, _ = st.columns([2, 3])
    with col_horizon:
        horizon_terpilih = st.selectbox(
            "⏱️ Horizon Prediksi",
            options=list(HORIZON_OPTIONS.keys()),
            index=5,
            key="horizon_forecast",
            help="Horizon pendek (≤2 jam) memakai model resolusi 15 menit yang lebih presisi untuk jangka dekat; horizon panjang memakai model per-jam.",
        )
    cfg_horizon = HORIZON_OPTIONS[horizon_terpilih]

    # ---------------------------------------------------------------
    # MODE PENDEK: 30 menit / 1 jam / 2 jam — hanya forecast masa depan,
    # real-time dari titik data terakhir, tanpa pemilihan tanggal.
    # ---------------------------------------------------------------
    if cfg_horizon["mode"] == "pendek":
        forecast_arima = ambil_forecast_terbaru(sumber="arima_forecast_pendek_v1", limit=200, mesin_id=mesin_pilihan)
        forecast_lstm = ambil_forecast_terbaru(sumber="lstm_forecast_pendek_v1", limit=200, mesin_id=mesin_pilihan)

        forecast_arima = _bersihkan_duplikat_forecast(forecast_arima).head(cfg_horizon["langkah"])
        forecast_lstm = _bersihkan_duplikat_forecast(forecast_lstm).head(cfg_horizon["langkah"])

        # Konteks: 4 jam terakhir data aktual, supaya grafik tetap terasa nyambung
        df_historis = df[pd.to_datetime(df["created_at"]) >= (pd.to_datetime(df["created_at"]).max() - pd.Timedelta(hours=4))]
        is_backtest = False
        tanggal_str = "Real-time (mulai sekarang)"

        if forecast_arima.empty and forecast_lstm.empty:
            st.warning(
                f"Belum ada forecast resolusi tinggi (15 menit) untuk mesin ini. "
                f"Jalankan 'Latih Model' terlebih dahulu."
            )
        else:
            tab_suhu, tab_getaran = st.tabs(["🌡️ Forecast Suhu", "📳 Forecast Getaran"])

            with tab_suhu:
                fig_suhu = _buat_grafik_forecast(
                    df_historis, forecast_arima, forecast_lstm,
                    kolom_aktual="suhu", kolom_prediksi="nilai_suhu_prediksi",
                    judul=f"Forecast Suhu ({horizon_terpilih})", label_sumbu_y="Suhu (°C)",
                    tanggal_str=tanggal_str, is_backtest=is_backtest,
                )
                st.plotly_chart(fig_suhu, width="stretch")

            with tab_getaran:
                fig_getaran = _buat_grafik_forecast(
                    df_historis, forecast_arima, forecast_lstm,
                    kolom_aktual="kecepatan_getaran", kolom_prediksi="nilai_getaran_prediksi",
                    judul=f"Forecast Kecepatan Getaran ({horizon_terpilih})", label_sumbu_y="Kecepatan Getaran",
                    tanggal_str=tanggal_str, is_backtest=is_backtest,
                )
                st.plotly_chart(fig_getaran, width="stretch")

            st.caption("ℹ️ Horizon pendek hanya menampilkan prakiraan ke depan (belum terjadi) — belum ada data aktual untuk dibandingkan, sehingga metrik akurasi (MAE) tidak ditampilkan di sini.")

    # ---------------------------------------------------------------
    # MODE BIASA: 6 / 12 / 24 jam — memakai forecast per-jam + backtest
    # per-hari yang sudah ada, dengan pemilihan tanggal seperti sebelumnya.
    # ---------------------------------------------------------------
    else:
        forecast_arima_all = ambil_forecast_terbaru(sumber="arima_forecast_v1", limit=3000, mesin_id=mesin_pilihan)
        forecast_lstm_all = ambil_forecast_terbaru(sumber="lstm_forecast_v1", limit=3000, mesin_id=mesin_pilihan)

        all_forecast = pd.concat([
            _ambil_kolom_target_waktu(forecast_arima_all),
            _ambil_kolom_target_waktu(forecast_lstm_all),
        ], ignore_index=True)

        if not all_forecast.empty:
            all_forecast["tanggal"] = pd.to_datetime(all_forecast["target_waktu"]).dt.date
            tanggal_tersedia = sorted(all_forecast["tanggal"].unique())

            tanggal_options = [t.strftime("%Y-%m-%d") for t in tanggal_tersedia]
            tanggal_dict = {t.strftime("%Y-%m-%d"): t for t in tanggal_tersedia}

            default_tanggal = tanggal_options[-1] if tanggal_options else None

            if default_tanggal:
                col_tanggal, col_info = st.columns([2, 3])

                with col_tanggal:
                    tanggal_terpilih_str = st.selectbox(
                        "📅 Pilih Tanggal Forecast",
                        options=tanggal_options,
                        index=len(tanggal_options) - 1,
                        key="tanggal_forecast",
                        help="Hanya tanggal yang punya data forecast yang muncul di daftar ini",
                    )

                    tanggal_terpilih = tanggal_dict[tanggal_terpilih_str]

                with col_info:
                    st.info(f"📊 Menampilkan forecast untuk **{tanggal_terpilih_str}** — {cfg_horizon['langkah']} jam pertama hari itu ({horizon_terpilih})")

                forecast_arima = forecast_arima_all[
                    pd.to_datetime(forecast_arima_all["target_waktu"]).dt.date == tanggal_terpilih
                ]
                forecast_lstm = forecast_lstm_all[
                    pd.to_datetime(forecast_lstm_all["target_waktu"]).dt.date == tanggal_terpilih
                ]

                forecast_arima = _bersihkan_duplikat_forecast(forecast_arima).head(cfg_horizon["langkah"])
                forecast_lstm = _bersihkan_duplikat_forecast(forecast_lstm).head(cfg_horizon["langkah"])

                df_historis = df[
                    pd.to_datetime(df["created_at"]).dt.date == tanggal_terpilih
                ]

                is_backtest = len(df_historis) >= 10

                if len(df_historis) < 10:
                    df_historis = df[
                        pd.to_datetime(df["created_at"]).dt.date <= tanggal_terpilih
                    ].tail(48)
                    st.caption(f"⚠️ Data historis di tanggal {tanggal_terpilih_str} terbatas, menampilkan 48 data terakhir sebelum tanggal tersebut.")

                if not forecast_arima.empty or not forecast_lstm.empty:
                    tab_suhu, tab_getaran = st.tabs(["🌡️ Forecast Suhu", "📳 Forecast Getaran"])

                    with tab_suhu:
                        if is_backtest:
                            _tampilkan_metrik_akurasi(df_historis, forecast_arima, forecast_lstm, "suhu", "nilai_suhu_prediksi", ".2f")

                        fig_suhu = _buat_grafik_forecast(
                            df_historis, forecast_arima, forecast_lstm,
                            kolom_aktual="suhu", kolom_prediksi="nilai_suhu_prediksi",
                            judul=f"Forecast Suhu ({horizon_terpilih})", label_sumbu_y="Suhu (°C)",
                            tanggal_str=tanggal_terpilih_str, is_backtest=is_backtest,
                        )
                        st.plotly_chart(fig_suhu, width="stretch")

                        col_arima_suhu, col_lstm_suhu = st.columns(2)
                        with col_arima_suhu:
                            st.markdown("**ARIMA**")
                            data_arima = forecast_arima.dropna(subset=["nilai_suhu_prediksi"])
                            if not data_arima.empty:
                                st.dataframe(
                                    data_arima[["target_waktu", "nilai_suhu_prediksi"]],
                                    width="stretch",
                                    column_config={
                                        "target_waktu": "Waktu",
                                        "nilai_suhu_prediksi": st.column_config.NumberColumn("Suhu (°C)", format="%.2f"),
                                    },
                                    hide_index=True,
                                )
                            else:
                                st.info("Tidak ada data ARIMA")

                        with col_lstm_suhu:
                            st.markdown("**LSTM**")
                            data_lstm = forecast_lstm.dropna(subset=["nilai_suhu_prediksi"])
                            if not data_lstm.empty:
                                st.dataframe(
                                    data_lstm[["target_waktu", "nilai_suhu_prediksi"]],
                                    width="stretch",
                                    column_config={
                                        "target_waktu": "Waktu",
                                        "nilai_suhu_prediksi": st.column_config.NumberColumn("Suhu (°C)", format="%.2f"),
                                    },
                                    hide_index=True,
                                )
                            else:
                                st.info("Tidak ada data LSTM")

                    with tab_getaran:
                        if is_backtest:
                            _tampilkan_metrik_akurasi(df_historis, forecast_arima, forecast_lstm, "kecepatan_getaran", "nilai_getaran_prediksi", ".4f")

                        fig_getaran = _buat_grafik_forecast(
                            df_historis, forecast_arima, forecast_lstm,
                            kolom_aktual="kecepatan_getaran", kolom_prediksi="nilai_getaran_prediksi",
                            judul=f"Forecast Kecepatan Getaran ({horizon_terpilih})", label_sumbu_y="Kecepatan Getaran",
                            tanggal_str=tanggal_terpilih_str, is_backtest=is_backtest,
                        )
                        st.plotly_chart(fig_getaran, width="stretch")

                        col_arima_getaran, col_lstm_getaran = st.columns(2)
                        with col_arima_getaran:
                            st.markdown("**ARIMA**")
                            data_arima = forecast_arima.dropna(subset=["nilai_getaran_prediksi"])
                            if not data_arima.empty:
                                st.dataframe(
                                    data_arima[["target_waktu", "nilai_getaran_prediksi"]],
                                    width="stretch",
                                    column_config={
                                        "target_waktu": "Waktu",
                                        "nilai_getaran_prediksi": st.column_config.NumberColumn("Kecepatan Getaran", format="%.4f"),
                                    },
                                    hide_index=True,
                                )
                            else:
                                st.info("Tidak ada data ARIMA")

                        with col_lstm_getaran:
                            st.markdown("**LSTM**")
                            data_lstm = forecast_lstm.dropna(subset=["nilai_getaran_prediksi"])
                            if not data_lstm.empty:
                                st.dataframe(
                                    data_lstm[["target_waktu", "nilai_getaran_prediksi"]],
                                    width="stretch",
                                    column_config={
                                        "target_waktu": "Waktu",
                                        "nilai_getaran_prediksi": st.column_config.NumberColumn("Kecepatan Getaran", format="%.4f"),
                                    },
                                    hide_index=True,
                                )
                            else:
                                st.info("Tidak ada data LSTM")
                else:
                    st.warning(f"Tidak ada data forecast untuk tanggal {tanggal_terpilih_str}")
            else:
                st.warning("Tidak ada data forecast tersedia. Jalankan training terlebih dahulu.")
        else:
            st.warning("Belum ada data forecast. Jalankan `train_model.py` terlebih dahulu.")

# =========================================================================
# DATA MENTAH
# =========================================================================
with st.expander("📄 Lihat Data Mentah"):
    st.dataframe(df.sort_values("created_at", ascending=False), width="stretch")

# =========================================================================
# AUTO-REFRESH
# =========================================================================
if auto_refresh:
    time.sleep(30)
    st.rerun()
