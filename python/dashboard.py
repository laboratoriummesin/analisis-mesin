"""
Dashboard Streamlit — Monitoring Mesin dengan ML & DL
Jalankan: streamlit run dashboard.py

CATATAN REVISI:
1. Status training tidak lagi memakai loop polling + session_state (hilang saat
   refresh). Sekarang setiap render dashboard menanyakan LANGSUNG ke GitHub
   Actions API "run terakhir statusnya apa & mulai kapan", lalu durasi dihitung
   dari situ. Jadi kalau browser di-refresh saat training berjalan, statusnya
   tetap muncul benar (karena sumber datanya GitHub, bukan session_state lokal).
   Butuh Streamlit >= 1.37 untuk st.fragment(run_every=...).
2. Badge "Normal/Tidak Normal" di card outlier sekarang ditemani tabel uji
   normalitas (Shapiro-Wilk, D'Agostino K^2, Jarque-Bera, Kolmogorov-Smirnov,
   Anderson-Darling). PENTING: ini konsep berbeda dari "% outlier" — uji
   normalitas menguji apakah SELURUH distribusi menyerupai kurva Gaussian,
   bukan menghitung titik ekstrem. Dengan n besar, uji normalitas hampir selalu
   menyimpulkan "tidak normal" walau datanya terlihat wajar — jangan jadikan
   satu-satunya acuan kondisi mesin.
3. Card Forecasting (mode 6/12/24 jam) sekarang punya dropdown Tanggal + Jam
   Mulai, supaya bisa lihat mis. forecast 2 jam dimulai dari jam 14:00, bukan
   selalu dari jam 00:00.
"""

import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests
from scipy import stats

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

    .normalitas-normal { color: #34D399; font-weight: 600; }
    .normalitas-tidaknormal { color: #F87171; font-weight: 600; }
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

# Ambang persentase outlier (indikator TERPISAH dari uji normalitas — lihat catatan di atas)
AMBANG_PERSEN_TIDAK_NORMAL = 5.0
ALPHA_UJI_NORMALITAS = 0.05

GITHUB_API_BASE = "https://api.github.com"
NAMA_WORKFLOW = "train_model.yml"


# =========================================================================
# TRIGGER + STATUS TRAINING VIA GITHUB ACTIONS
# (didesain supaya GitHub jadi satu-satunya sumber kebenaran status —
#  bukan session_state — sehingga tahan terhadap refresh browser)
# =========================================================================
def _github_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }


def trigger_training_github(token, repo):
    """Memicu workflow training. Tidak menunggu/polling di sini sama sekali —
    status berjalannya akan diambil ulang dari GitHub oleh
    tampilkan_status_training() di render berikutnya."""
    url = f"{GITHUB_API_BASE}/repos/{repo}/actions/workflows/{NAMA_WORKFLOW}/dispatches"
    try:
        resp = requests.post(url, headers=_github_headers(token), json={"ref": "main"}, timeout=15)
        if resp.status_code == 204:
            return True, "Training berhasil dipicu di GitHub Actions."
        return False, f"Gagal memicu training: {resp.status_code} - {resp.text[:200]}"
    except Exception as e:
        return False, f"Error saat memicu training: {e}"


def ambil_run_terbaru(token, repo):
    """Ambil 1 run PALING BARU dari workflow ini, apa pun statusnya."""
    url = f"{GITHUB_API_BASE}/repos/{repo}/actions/workflows/{NAMA_WORKFLOW}/runs"
    try:
        resp = requests.get(url, headers=_github_headers(token), params={"per_page": 1}, timeout=15)
        if resp.status_code == 200:
            runs = resp.json().get("workflow_runs", [])
            return runs[0] if runs else None
    except Exception:
        pass
    return None


def ambil_run_sukses_terakhir(token, repo):
    """Ambil run SUKSES paling baru — dipakai untuk info 'terakhir dilatih pada'
    meskipun run yang paling baru gagal/sedang berjalan."""
    url = f"{GITHUB_API_BASE}/repos/{repo}/actions/workflows/{NAMA_WORKFLOW}/runs"
    try:
        resp = requests.get(
            url, headers=_github_headers(token),
            params={"per_page": 5, "status": "success"}, timeout=15,
        )
        if resp.status_code == 200:
            runs = resp.json().get("workflow_runs", [])
            return runs[0] if runs else None
    except Exception:
        pass
    return None


def _fmt_durasi(detik_total):
    detik_total = max(0, int(detik_total))
    menit, detik = divmod(detik_total, 60)
    jam, menit = divmod(menit, 60)
    if jam > 0:
        return f"{jam}j {menit}m {detik}d"
    return f"{menit}m {detik}d"


def _render_status_training():
    token = st.secrets.get("GITHUB_TOKEN")
    repo = st.secrets.get("GITHUB_REPO")

    if not token or not repo:
        st.caption("ℹ️ Status training tidak bisa ditampilkan — GITHUB_TOKEN / GITHUB_REPO belum diisi di Secrets.")
        return

    run_terbaru = ambil_run_terbaru(token, repo)
    if run_terbaru is None:
        st.caption("Belum ada riwayat training yang tercatat di GitHub Actions.")
        return

    status_run = run_terbaru.get("status")  # queued / in_progress / completed
    conclusion = run_terbaru.get("conclusion")
    url_run = run_terbaru.get("html_url")
    waktu_mulai = pd.to_datetime(run_terbaru["created_at"]).to_pydatetime()

    if status_run != "completed":
        durasi_detik = (datetime.now(timezone.utc) - waktu_mulai).total_seconds()
        label = "menunggu antrian" if status_run == "queued" else "sedang berjalan"
        st.info(
            f"⏳ Training **{label}** — mulai {waktu_mulai.strftime('%H:%M:%S UTC')}, "
            f"sudah berjalan **{_fmt_durasi(durasi_detik)}**. "
            f"[Lihat di GitHub]({url_run})"
        )
    elif conclusion == "success":
        st.success(
            f"✅ Training terakhir **SUKSES** — selesai {waktu_mulai.strftime('%Y-%m-%d %H:%M:%S UTC')}. "
            f"[Detail]({url_run})"
        )
    else:
        st.error(
            f"❌ Training terakhir **GAGAL** ({conclusion}) — {waktu_mulai.strftime('%Y-%m-%d %H:%M:%S UTC')}. "
            f"[Lihat log]({url_run})"
        )

    # Info "terakhir dilatih" — dicari terpisah supaya tetap muncul walau run
    # paling baru gagal/masih berjalan.
    run_sukses = ambil_run_sukses_terakhir(token, repo)
    if run_sukses is not None:
        waktu_sukses = pd.to_datetime(run_sukses.get("updated_at") or run_sukses["created_at"])
        if not (status_run != "completed") and conclusion == "success":
            pass  # sudah tercakup di pesan sukses di atas
        else:
            st.caption(f"🕓 Terakhir kali berhasil dilatih: {waktu_sukses.strftime('%Y-%m-%d %H:%M:%S UTC')}")


try:
    @st.fragment(run_every=5)
    def tampilkan_status_training():
        _render_status_training()
except AttributeError:
    # Fallback untuk Streamlit versi lama tanpa st.fragment(run_every=...):
    # status tetap benar (karena selalu query GitHub live), hanya saja tidak
    # auto-update tiap 5 detik tanpa interaksi/refresh dari pengguna.
    def tampilkan_status_training():
        _render_status_training()


def _ambil_kolom_target_waktu(df_forecast):
    if df_forecast.empty or "target_waktu" not in df_forecast.columns:
        return pd.DataFrame(columns=["target_waktu"])
    return df_forecast[["target_waktu"]]


def _bersihkan_duplikat_forecast(df_forecast):
    if df_forecast.empty or "target_waktu" not in df_forecast.columns:
        return df_forecast

    kolom_numerik = [c for c in ["nilai_suhu_prediksi", "nilai_getaran_prediksi"]
                     if c in df_forecast.columns]
    if not kolom_numerik:
        return df_forecast

    df_forecast = df_forecast.copy()
    for c in kolom_numerik:
        df_forecast[c] = pd.to_numeric(df_forecast[c], errors="coerce")

    df_forecast["target_waktu"] = pd.to_datetime(df_forecast["target_waktu"], errors="coerce")
    df_forecast = df_forecast.dropna(subset=["target_waktu"])

    return (
        df_forecast.groupby("target_waktu", as_index=False)[kolom_numerik]
        .mean()
        .sort_values("target_waktu")
        .reset_index(drop=True)
    )


def _hitung_mae(df_historis, forecast_df, kolom_aktual, kolom_prediksi, resolusi="1h"):
    if df_historis.empty or forecast_df.empty:
        return None
    if kolom_prediksi not in forecast_df.columns or "target_waktu" not in forecast_df.columns:
        return None

    try:
        aktual_per_jam = (
            df_historis.set_index("created_at")[kolom_aktual]
            .resample(resolusi).mean()
        )
    except Exception:
        return None

    aktual_per_jam = aktual_per_jam[~aktual_per_jam.index.duplicated(keep="last")]
    aktual_per_jam = aktual_per_jam.dropna()

    prediksi = forecast_df.dropna(subset=[kolom_prediksi]).copy()
    if prediksi.empty:
        return None

    prediksi["target_waktu"] = pd.to_datetime(prediksi["target_waktu"], errors="coerce")
    prediksi = prediksi.dropna(subset=["target_waktu"])
    if prediksi.empty:
        return None
    try:
        prediksi["target_waktu"] = prediksi["target_waktu"].dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass

    prediksi = (
        prediksi.groupby("target_waktu", as_index=True)[kolom_prediksi].mean()
    )

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


def _format_angka(series, desimal):
    """Ubah kolom angka jadi teks untuk hover, NaN -> '—' (bukan 'nan')."""
    return series.apply(lambda v: "—" if pd.isna(v) else f"{v:.{desimal}f}")


def _buat_grafik_forecast(df_historis, forecast_arima, forecast_lstm, kolom_aktual, kolom_prediksi,
                           judul, label_sumbu_y, tanggal_str, label_tipe):
    fig = go.Figure()

    data_arima = forecast_arima.dropna(subset=[kolom_prediksi]) if not forecast_arima.empty else forecast_arima
    data_lstm = forecast_lstm.dropna(subset=[kolom_prediksi]) if not forecast_lstm.empty else forecast_lstm

    # --- Gaya adaptif: mode HARUS selalu mengandung "markers" — kalau mode-nya
    # cuma "lines", Plotly TIDAK menggambar bulatan apa pun saat titik itu
    # di-hover (ini yang bikin bulatan highlight hilang sebelumnya). Supaya
    # grafik padat (resolusi 1 menit, bisa ~1440 titik/garis) tidak numpuk,
    # yang dikecilkan/ditransparankan adalah UKURAN & OPASITAS marker, bukan
    # mode-nya.
    jumlah_titik_total = len(df_historis) + len(data_arima) + len(data_lstm)
    mode_garis = "lines+markers"
    if jumlah_titik_total <= 200:
        ukuran_marker, opasitas_marker = 7, 1.0
    else:
        ukuran_marker, opasitas_marker = 3, 0.45

    if not df_historis.empty:
        customdata_aktual = None
        hover_aktual = "Tanggal: %{x|%d-%m-%Y %H:%M}<br>" + f"{label_sumbu_y}: " + "%{y:.2f}<extra>Data Aktual</extra>"

        if {"suhu", "kecepatan_getaran"}.issubset(df_historis.columns):
            kondisi_teks = (
                df_historis["kondisi"].fillna("—").astype(str)
                if "kondisi" in df_historis.columns
                else pd.Series(["—"] * len(df_historis))
            )
            customdata_aktual = np.column_stack([
                _format_angka(df_historis["suhu"], 2),
                _format_angka(df_historis["kecepatan_getaran"], 4),
                kondisi_teks,
            ])
            hover_aktual = (
                "Tanggal: %{x|%d-%m-%Y %H:%M}<br>"
                "Suhu: %{customdata[0]} °C<br>"
                "Getaran: %{customdata[1]}<br>"
                "Kondisi: %{customdata[2]}"
                "<extra>Data Aktual</extra>"
            )

        fig.add_trace(go.Scatter(
            x=df_historis["created_at"],
            y=df_historis[kolom_aktual],
            mode=mode_garis,
            name="Data Aktual",
            line=dict(color="#E2E8F0", width=2),
            marker=dict(size=ukuran_marker, color="#E2E8F0", symbol="circle",
                        opacity=opasitas_marker, line=dict(width=1.5, color="#0F172A")),
            customdata=customdata_aktual,
            hovertemplate=hover_aktual,
        ))

    def _tambah_trace_forecast(data_forecast, nama, warna, dash, simbol):
        if data_forecast.empty:
            return
        customdata = np.column_stack([
            _format_angka(data_forecast["nilai_suhu_prediksi"], 2)
            if "nilai_suhu_prediksi" in data_forecast.columns else ["—"] * len(data_forecast),
            _format_angka(data_forecast["nilai_getaran_prediksi"], 4)
            if "nilai_getaran_prediksi" in data_forecast.columns else ["—"] * len(data_forecast),
        ])
        fig.add_trace(go.Scatter(
            x=data_forecast["target_waktu"],
            y=data_forecast[kolom_prediksi],
            mode=mode_garis,
            name=nama,
            line=dict(color=warna, width=2.5, dash=dash),
            marker=dict(size=ukuran_marker, color=warna, symbol=simbol,
                        opacity=opasitas_marker, line=dict(width=1.5, color="#0F172A")),
            customdata=customdata,
            hovertemplate=(
                "Tanggal: %{x|%d-%m-%Y %H:%M}<br>"
                "Prediksi Suhu: %{customdata[0]} °C<br>"
                "Prediksi Getaran: %{customdata[1]}"
                f"<extra>{nama}</extra>"
            ),
        ))

    _tambah_trace_forecast(data_arima, "ARIMA", "#F59E0B", "dot", "diamond")
    _tambah_trace_forecast(data_lstm, "LSTM", "#8B5CF6", "dash", "square")

    if not df_historis.empty and (not data_arima.empty or not data_lstm.empty):
        waktu_mulai_forecast = min(
            [d["target_waktu"].min() for d in [data_arima, data_lstm] if not d.empty]
        )
        fig.add_vline(
            x=waktu_mulai_forecast,
            line_width=1.5,
            line_dash="dash",
            line_color="rgba(100, 116, 139, 0.6)",
            annotation_text="Mulai Prakiraan",
            annotation_position="top",
            annotation_font_size=11,
            annotation_font_color="#94A3B8",
        )

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
            orientation="v",
            yanchor="top", y=0.85,
            xanchor="left", x=1.02,
        ),
        xaxis=dict(
            title="Waktu",
            showgrid=True, gridcolor="#1E293B", gridwidth=1,
            rangeslider=dict(visible=True, thickness=0.06),
            tickformat="%H:%M",
            nticks=12,
        ),
        yaxis=dict(
            title=label_sumbu_y,
            showgrid=True, gridcolor="#1E293B", gridwidth=1,
        ),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#1E293B",
            bordercolor="#334155",
            font=dict(color="#F1F5F9", size=12),
        ),
        margin=dict(t=120, r=140),
        height=480,
    )
    return fig
                              
    def _tambah_trace_forecast(data_forecast, nama, warna, dash, simbol):
        if data_forecast.empty:
            return
        customdata = np.column_stack([
            _format_angka(data_forecast["nilai_suhu_prediksi"], 2)
            if "nilai_suhu_prediksi" in data_forecast.columns else ["—"] * len(data_forecast),
            _format_angka(data_forecast["nilai_getaran_prediksi"], 4)
            if "nilai_getaran_prediksi" in data_forecast.columns else ["—"] * len(data_forecast),
        ])
        fig.add_trace(go.Scatter(
            x=data_forecast["target_waktu"],
            y=data_forecast[kolom_prediksi],
            mode=mode_garis,
            name=nama,
            line=dict(color=warna, width=2.5, dash=dash),
            marker=dict(size=ukuran_marker, color=warna, symbol=simbol,
                        line=dict(width=1.5, color="#0F172A")),
            customdata=customdata,
            hovertemplate=(
                "Tanggal: %{x|%d-%m-%Y %H:%M}<br>"
                "Prediksi Suhu: %{customdata[0]} °C<br>"
                "Prediksi Getaran: %{customdata[1]}"
                f"<extra>{nama}</extra>"
            ),
        ))

    _tambah_trace_forecast(data_arima, "ARIMA", "#F59E0B", "dot", "diamond")
    _tambah_trace_forecast(data_lstm, "LSTM", "#8B5CF6", "dash", "square")

    if not df_historis.empty and (not data_arima.empty or not data_lstm.empty):
        waktu_mulai_forecast = min(
            [d["target_waktu"].min() for d in [data_arima, data_lstm] if not d.empty]
        )
        fig.add_vline(
            x=waktu_mulai_forecast,
            line_width=1.5,
            line_dash="dash",
            line_color="rgba(100, 116, 139, 0.6)",
            annotation_text="Mulai Prakiraan",
            annotation_position="top",
            annotation_font_size=11,
            annotation_font_color="#94A3B8",
        )

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
            orientation="v",
            yanchor="top", y=0.85,
            xanchor="left", x=1.02,
        ),
        xaxis=dict(
            title="Waktu",
            showgrid=True, gridcolor="#1E293B", gridwidth=1,
            rangeslider=dict(visible=True, thickness=0.06),
            tickformat="%H:%M",
            nticks=12,
        ),
        yaxis=dict(
            title=label_sumbu_y,
            showgrid=True, gridcolor="#1E293B", gridwidth=1,
        ),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#1E293B",
            bordercolor="#334155",
            font=dict(color="#F1F5F9", size=12),
        ),
        margin=dict(t=120, r=140),
        height=480,
    )
    return fig


def _tampilkan_metrik_akurasi(df_historis, forecast_arima, forecast_lstm, kolom_aktual, kolom_prediksi, format_desimal, resolusi="1h"):
    mae_arima = _hitung_mae(df_historis, forecast_arima, kolom_aktual, kolom_prediksi, resolusi)
    mae_lstm = _hitung_mae(df_historis, forecast_lstm, kolom_aktual, kolom_prediksi, resolusi)

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
# UJI NORMALITAS (multi-metode)
# =========================================================================
def uji_normalitas(data, alpha=ALPHA_UJI_NORMALITAS):
    """Menjalankan beberapa uji normalitas sekaligus pada satu kolom numerik.
    Mengembalikan DataFrame: Metode | Statistik | p-value | Kesimpulan.

    Catatan penting: uji normalitas menguji bentuk SELURUH distribusi (apakah
    menyerupai kurva Gaussian), bukan menghitung titik outlier. Dengan jumlah
    data besar, uji-uji ini (terutama Shapiro-Wilk & K-S) sangat mudah
    menyimpulkan 'Tidak Normal' walau penyimpangannya kecil secara praktis —
    jangan dipakai sebagai satu-satunya penentu kondisi mesin.
    """
    data = pd.to_numeric(pd.Series(data), errors="coerce").dropna()
    n = len(data)
    baris = []

    if n < 8:
        return pd.DataFrame([{
            "Metode": "—", "Statistik": None, "p-value": None,
            "Kesimpulan": f"Data terlalu sedikit (n={n}) untuk uji normalitas"
        }])

    # Shapiro-Wilk — akurat untuk n kecil-menengah; API scipy membatasi sampel besar
    try:
        sampel = data if n <= 5000 else data.sample(5000, random_state=42)
        stat, p = stats.shapiro(sampel)
        baris.append(("Shapiro-Wilk", stat, p, "Normal" if p > alpha else "Tidak Normal"))
    except Exception as e:
        baris.append(("Shapiro-Wilk", None, None, f"Gagal dihitung ({e})"))

    # D'Agostino K^2
    try:
        stat, p = stats.normaltest(data)
        baris.append(("D'Agostino K²", stat, p, "Normal" if p > alpha else "Tidak Normal"))
    except Exception as e:
        baris.append(("D'Agostino K²", None, None, f"Gagal dihitung ({e})"))

    # Jarque-Bera
    try:
        stat, p = stats.jarque_bera(data)
        baris.append(("Jarque-Bera", stat, p, "Normal" if p > alpha else "Tidak Normal"))
    except Exception as e:
        baris.append(("Jarque-Bera", None, None, f"Gagal dihitung ({e})"))

    # Kolmogorov-Smirnov (dibandingkan distribusi normal dari mean & std data sendiri)
    try:
        stat, p = stats.kstest(data, "norm", args=(data.mean(), data.std(ddof=1)))
        baris.append(("Kolmogorov-Smirnov", stat, p, "Normal" if p > alpha else "Tidak Normal"))
    except Exception as e:
        baris.append(("Kolmogorov-Smirnov", None, None, f"Gagal dihitung ({e})"))

    # Anderson-Darling — tidak pakai p-value, dibandingkan nilai kritis 5%
    try:
        hasil_ad = stats.anderson(data, dist="norm")
        sig_levels = list(hasil_ad.significance_level)
        idx = sig_levels.index(5.0) if 5.0 in sig_levels else min(range(len(sig_levels)), key=lambda i: abs(sig_levels[i] - 5.0))
        kritis_5 = hasil_ad.critical_values[idx]
        kesimpulan_ad = "Normal" if hasil_ad.statistic < kritis_5 else "Tidak Normal"
        baris.append(("Anderson-Darling", hasil_ad.statistic, None, kesimpulan_ad))
    except Exception as e:
        baris.append(("Anderson-Darling", None, None, f"Gagal dihitung ({e})"))

    df_hasil = pd.DataFrame(baris, columns=["Metode", "Statistik", "p-value", "Kesimpulan"])
    return df_hasil


def _tampilkan_tabel_normalitas(df_uji, judul):
    st.markdown(f"**{judul}**")
    df_tampil = df_uji.copy()
    if "Statistik" in df_tampil.columns:
        df_tampil["Statistik"] = df_tampil["Statistik"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    if "p-value" in df_tampil.columns:
        df_tampil["p-value"] = df_tampil["p-value"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    st.dataframe(df_tampil, width="stretch", hide_index=True)

    kesimpulan_valid = [k for k in df_uji["Kesimpulan"] if k in ("Normal", "Tidak Normal")]
    if kesimpulan_valid:
        jumlah_normal = sum(1 for k in kesimpulan_valid if k == "Normal")
        total = len(kesimpulan_valid)
        if jumlah_normal == total:
            st.caption(f"✅ Semua {total} metode uji menyimpulkan data berdistribusi **Normal** (α={ALPHA_UJI_NORMALITAS}).")
        elif jumlah_normal == 0:
            st.caption(f"⚠️ Semua {total} metode uji menyimpulkan data **Tidak Normal** (α={ALPHA_UJI_NORMALITAS}).")
        else:
            st.caption(f"⚠️ {jumlah_normal} dari {total} metode menyimpulkan **Normal**, sisanya Tidak Normal (α={ALPHA_UJI_NORMALITAS}) — hasil beragam, perlu interpretasi hati-hati.")


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

    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

    with col_btn1:
        if st.button("🔍 Audit Data", width="stretch", key="btn_audit_data"):
            st.toast("Audit selesai — lihat status di atas", icon="🔍")

    with col_btn2:
        if st.button("Bersihkan", width="stretch", key="btn_cleanup_db"):
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
        if st.button("Lihat Hasil Pembersihan", width="stretch", key="btn_view_cleanup"):
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
        if st.button("Latih Model", width="stretch", key="btn_train_model"):
            token = st.secrets.get("GITHUB_TOKEN")
            repo = st.secrets.get("GITHUB_REPO")
            if not token or not repo:
                st.error("❌ GITHUB_TOKEN / GITHUB_REPO belum diisi di menu Secrets Streamlit Cloud.")
            else:
                berhasil, pesan = trigger_training_github(token, repo)
                if berhasil:
                    st.toast(pesan, icon="🚀")
                else:
                    st.error(pesan)

    # Status training — selalu ditampilkan (bukan hanya sesudah klik tombol),
    # dan tetap benar walau halaman baru saja di-refresh, karena datanya
    # ditanyakan langsung ke GitHub setiap render.
    tampilkan_status_training()

    # ------------------------------------------------------------
    # SCATTER PLOT OUTLIER + UJI NORMALITAS
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
            "Garis kuning pada tiap grafik adalah garis regresi linear (OLS) yang menunjukkan "
            "tren hubungan antara suhu dan getaran. Sumbu grafik getaran sengaja dibalik "
            "(suhu di sumbu-Y) supaya kedua grafik tidak terlihat identik."
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

            st.markdown(
                f'<div class="kondisi-badge kondisi-{"tidaknormal" if persen_outlier_suhu > AMBANG_PERSEN_TIDAK_NORMAL else "normal"}">'
                f'{"⚠️" if persen_outlier_suhu > AMBANG_PERSEN_TIDAK_NORMAL else "✅"} '
                f'{outlier_stats.get("suhu", 0)} titik outlier ({persen_outlier_suhu:.1f}% dari data) — indikator jumlah titik ekstrem</div>',
                unsafe_allow_html=True,
            )

            uji_suhu = uji_normalitas(df_temp["suhu"])
            with st.expander("📐 Uji Normalitas Distribusi — Suhu", expanded=False):
                _tampilkan_tabel_normalitas(uji_suhu, "Uji Normalitas Suhu")

        with col_plot2:
            fig_outlier_getaran = px.scatter(
                df_temp,
                x="kecepatan_getaran",
                y="suhu",
                color="is_outlier_getaran",
                color_discrete_map={False: "#3B82F6", True: "#EF4444"},
                title="🔴 Outlier Getaran (Tanda Merah)",
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

            st.markdown(
                f'<div class="kondisi-badge kondisi-{"tidaknormal" if persen_outlier_getaran > AMBANG_PERSEN_TIDAK_NORMAL else "normal"}">'
                f'{"⚠️" if persen_outlier_getaran > AMBANG_PERSEN_TIDAK_NORMAL else "✅"} '
                f'{outlier_stats.get("kecepatan_getaran", 0)} titik outlier ({persen_outlier_getaran:.1f}% dari data) — indikator jumlah titik ekstrem</div>',
                unsafe_allow_html=True,
            )

            uji_getaran = uji_normalitas(df_temp["kecepatan_getaran"])
            with st.expander("📐 Uji Normalitas Distribusi — Getaran", expanded=False):
                _tampilkan_tabel_normalitas(uji_getaran, "Uji Normalitas Kecepatan Getaran")

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
# CARD 4: FORECASTING (ARIMA vs LSTM) — REVISI: SEMUA TIMEFRAME RESOLUSI 1 MENIT
# — Tiga dropdown (Tanggal, Jam Mulai, Timeframe) SELALU tampil.
# — Timeframe: 15 Menit, 30 Menit, 1 Jam, 2 Jam, ... 24 Jam.
# — SEMUA titik (data aktual maupun prakiraan) sekarang berjarak 1 MENIT,
#   untuk timeframe apa pun. Tidak ada lagi pembedaan "resolusi tinggi" vs
#   "resolusi rendah" seperti sebelumnya — semuanya satu resolusi: 1 menit.
#   (Ini mengikuti train_model.py yang direvisi: ARIMA & LSTM sekarang cuma
#   punya SATU sumber masing-masing, arima_forecast_v1 & lstm_forecast_v1,
#   keduanya di resolusi 1 menit.)
# — Grafik menampilkan jendela [jam_mulai, jam_mulai + timeframe]: bagian yang
#   sudah punya data aktual ditampilkan sebagai data aktual, bagian yang belum
#   terjadi diisi oleh prakiraan (ARIMA/LSTM) — jadi satu jendela bisa berisi
#   campuran aktual+prakiraan sekaligus, bukan cuma salah satu.
# — Konsistensi jendela: waktu_mulai = tanggal + jam_mulai, waktu_selesai =
#   waktu_mulai + timeframe. Ini SELALU begitu untuk semua timeframe (mis.
#   12 Jam mulai jam 02:00 -> selalu 02:00-14:00). Yang membuat ini benar-benar
#   konsisten adalah backtest ARIMA di train_model.py yang sekarang mencakup
#   1 hari PENUH (00:00-23:59) per hari, jadi jendela manapun yang dipilih user
#   di hari itu selalu punya data forecast, tidak terpotong.
# =========================================================================
TIMEFRAME_OPTIONS = {"15 Menit": 0.25, "30 Menit": 0.5}
for _j in range(1, 25):
    TIMEFRAME_OPTIONS[f"{_j} Jam"] = float(_j)

RESOLUSI_LABEL = "1 Menit"
RESOLUSI_RESAMPLE = "1min"


def _forecast_potong_rentang(df_all, waktu_mulai, waktu_selesai):
    if df_all.empty or "target_waktu" not in df_all.columns:
        return df_all
    d = df_all.copy()
    d["target_waktu"] = pd.to_datetime(d["target_waktu"], errors="coerce")
    d = d.dropna(subset=["target_waktu"])
    return d[(d["target_waktu"] >= waktu_mulai) & (d["target_waktu"] <= waktu_selesai)]


def _resample_aktual_ke_1_menit(df_slice):
    """
    Meresample data AKTUAL ke titik per-1-menit (mean + interpolasi), dengan
    pola yang sama seperti resample forecast di train_model.py. Ini supaya
    data aktual dan data prakiraan sama-sama berjarak 1 menit dan bisa
    disambung mulus dalam satu grafik, untuk timeframe apa pun.
    """
    if df_slice.empty:
        return df_slice

    d = df_slice.copy()
    d["created_at"] = pd.to_datetime(d["created_at"], errors="coerce")
    d = d.dropna(subset=["created_at"]).set_index("created_at").sort_index()

    kolom_numerik = [k for k in ["suhu", "kecepatan_getaran"] if k in d.columns]
    if not kolom_numerik:
        return df_slice

    d_resampled = d[kolom_numerik].resample(RESOLUSI_RESAMPLE).mean().interpolate()
    return d_resampled.reset_index()


with st.container(border=True):
    st.markdown(
        '<div class="card-title">📈 Forecasting Suhu & Getaran <span class="badge badge-dl">DEEP LEARNING</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Pilih jendela waktu (tanggal + jam mulai + panjang timeframe). Bagian jendela yang "
        "datanya sudah masuk akan ditampilkan sebagai data aktual, sisanya yang belum terjadi "
        "diisi otomatis oleh prakiraan ARIMA & LSTM — semua titik (aktual maupun prakiraan) "
        "berjarak 1 menit."
    )

    # ---------------------------------------------------------------
    # Ambil sumber forecast SEKALI di awal (tidak disembunyikan di balik
    # kondisi apa pun, supaya dropdown selalu bisa dipakai). Sekarang cuma
    # ada SATU sumber per model (resolusi 1 menit untuk semua timeframe),
    # jadi tidak perlu lagi mengambil versi "pendek" terpisah.
    # ---------------------------------------------------------------
    forecast_arima_all = ambil_forecast_terbaru(sumber="arima_forecast_v1", limit=10000, mesin_id=mesin_pilihan)
    forecast_lstm_all = ambil_forecast_terbaru(sumber="lstm_forecast_v1", limit=10000, mesin_id=mesin_pilihan)

    waktu_terakhir_aktual = pd.to_datetime(df["created_at"]).max()

    # Opsi tanggal: gabungan tanggal yang punya data aktual DAN/ATAU forecast,
    # supaya dropdown tidak pernah kosong (df dijamin tidak kosong di atas).
    tanggal_dari_data = set(pd.to_datetime(df["created_at"]).dt.date.unique())
    for src in [forecast_arima_all, forecast_lstm_all]:
        if not src.empty and "target_waktu" in src.columns:
            tanggal_dari_data.update(pd.to_datetime(src["target_waktu"], errors="coerce").dt.date.dropna().unique())
    tanggal_tersedia = sorted(tanggal_dari_data)
    tanggal_options = [t.strftime("%Y-%m-%d") for t in tanggal_tersedia]
    tanggal_dict = {t.strftime("%Y-%m-%d"): t for t in tanggal_tersedia}
    index_default_tanggal = tanggal_options.index(waktu_terakhir_aktual.strftime("%Y-%m-%d")) \
        if waktu_terakhir_aktual.strftime("%Y-%m-%d") in tanggal_options else len(tanggal_options) - 1

    # ---------------------------------------------------------------
    # KETIGA DROPDOWN — selalu tampil, tidak ada yang disembunyikan.
    # ---------------------------------------------------------------
    col_tanggal, col_jam, col_timeframe = st.columns(3)

    with col_tanggal:
        tanggal_terpilih_str = st.selectbox(
            "📅 Tanggal",
            options=tanggal_options,
            index=index_default_tanggal,
            key="tanggal_forecast",
        )
        tanggal_terpilih = tanggal_dict[tanggal_terpilih_str]

    with col_jam:
        jam_mulai_forecast = st.selectbox(
            "🕐 Jam Mulai",
            options=list(range(24)),
            index=0,
            format_func=lambda j: f"{j:02d}:00",
            key="jam_mulai_forecast",
            help="Contoh: pilih 04:00 dengan timeframe 3 Jam → jendela 04:00-07:00.",
        )

    with col_timeframe:
        timeframe_terpilih = st.selectbox(
            "⏱️ Timeframe",
            options=list(TIMEFRAME_OPTIONS.keys()),
            index=list(TIMEFRAME_OPTIONS.keys()).index("24 Jam"),
            key="timeframe_forecast",
        )
    timeframe_jam = TIMEFRAME_OPTIONS[timeframe_terpilih]

    # Rumus jendela ini SAMA untuk semua timeframe -> konsisten. Mis. 12 Jam
    # mulai jam 02:00 akan selalu menghasilkan 02:00 -> 14:00, untuk timeframe
    # berapa pun yang dipilih (15 Menit s/d 24 Jam).
    waktu_mulai = pd.Timestamp(tanggal_terpilih) + pd.Timedelta(hours=jam_mulai_forecast)
    waktu_selesai = waktu_mulai + pd.Timedelta(hours=timeframe_jam)

    st.info(f"📊 Jendela ditampilkan: **{waktu_mulai.strftime('%Y-%m-%d %H:%M')} → {waktu_selesai.strftime('%Y-%m-%d %H:%M')}**")

    # ---------------------------------------------------------------
    # Potong forecast sesuai jendela. Resolusinya SELALU 1 menit, untuk
    # timeframe apa pun -> tidak perlu lagi logika fallback resolusi.
    # ---------------------------------------------------------------
    forecast_arima = _bersihkan_duplikat_forecast(_forecast_potong_rentang(forecast_arima_all, waktu_mulai, waktu_selesai))
    forecast_lstm = _bersihkan_duplikat_forecast(_forecast_potong_rentang(forecast_lstm_all, waktu_mulai, waktu_selesai))
    resolusi_label, resolusi_resample = RESOLUSI_LABEL, RESOLUSI_RESAMPLE

    st.caption(f"📐 Resolusi titik yang dipakai (aktual & prakiraan): **{resolusi_label}**")

    # Data aktual dalam jendela yang sama, DIRESAMPLE ke 1 menit supaya
    # selaras dengan resolusi forecast (lihat _resample_aktual_ke_1_menit).
    df_historis_mentah = df[
        (pd.to_datetime(df["created_at"]) >= waktu_mulai) &
        (pd.to_datetime(df["created_at"]) <= waktu_selesai)
    ]
    df_historis = _resample_aktual_ke_1_menit(df_historis_mentah)

    # Label kondisi jendela: penuh riwayat / penuh prakiraan / campuran
    if waktu_selesai <= waktu_terakhir_aktual:
        label_tipe = "🕓 Riwayat penuh — seluruh jendela sudah punya data aktual, prakiraan ditampilkan untuk pembanding"
    elif waktu_mulai >= waktu_terakhir_aktual:
        label_tipe = "🔮 Prakiraan penuh — seluruh jendela belum terjadi"
    else:
        label_tipe = "🔀 Campuran — sebagian jendela sudah aktual, sisanya diisi prakiraan"

    if df_historis.empty and forecast_arima.empty and forecast_lstm.empty:
        st.warning(
            f"Tidak ada data aktual maupun prakiraan untuk jendela {waktu_mulai.strftime('%Y-%m-%d %H:%M')} "
            f"→ {waktu_selesai.strftime('%Y-%m-%d %H:%M')}. Coba jendela lain atau jalankan 'Latih Model' dahulu.\n\n"
            f"Catatan: backtest historis ARIMA sekarang hanya dibuat untuk beberapa hari terakhir "
            f"(lihat JUMLAH_HARI_RIWAYAT_FORECAST di train_model.py), jadi tanggal yang lebih lama dari itu "
            f"memang belum akan punya prakiraan."
        )
    else:
        tab_suhu, tab_getaran = st.tabs(["🌡️ Forecast Suhu", "📳 Forecast Getaran"])

        with tab_suhu:
            _tampilkan_metrik_akurasi(df_historis, forecast_arima, forecast_lstm, "suhu", "nilai_suhu_prediksi", ".2f", resolusi_resample)

            fig_suhu = _buat_grafik_forecast(
                df_historis, forecast_arima, forecast_lstm,
                kolom_aktual="suhu", kolom_prediksi="nilai_suhu_prediksi",
                judul=f"Forecast Suhu ({timeframe_terpilih})", label_sumbu_y="Suhu (°C)",
                tanggal_str=f"{waktu_mulai.strftime('%Y-%m-%d %H:%M')} - {waktu_selesai.strftime('%H:%M')}",
                label_tipe=label_tipe,
            )
            st.plotly_chart(fig_suhu, width="stretch")

            col_arima_suhu, col_lstm_suhu = st.columns(2)
            with col_arima_suhu:
                st.markdown("**ARIMA**")
                data_arima = forecast_arima.dropna(subset=["nilai_suhu_prediksi"]) if not forecast_arima.empty else forecast_arima
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
                    st.info("Tidak ada data ARIMA di jendela ini")

            with col_lstm_suhu:
                st.markdown("**LSTM**")
                data_lstm = forecast_lstm.dropna(subset=["nilai_suhu_prediksi"]) if not forecast_lstm.empty else forecast_lstm
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
                    st.info("Tidak ada data LSTM di jendela ini")

        with tab_getaran:
            _tampilkan_metrik_akurasi(df_historis, forecast_arima, forecast_lstm, "kecepatan_getaran", "nilai_getaran_prediksi", ".4f", resolusi_resample)

            fig_getaran = _buat_grafik_forecast(
                df_historis, forecast_arima, forecast_lstm,
                kolom_aktual="kecepatan_getaran", kolom_prediksi="nilai_getaran_prediksi",
                judul=f"Forecast Kecepatan Getaran ({timeframe_terpilih})", label_sumbu_y="Kecepatan Getaran",
                tanggal_str=f"{waktu_mulai.strftime('%Y-%m-%d %H:%M')} - {waktu_selesai.strftime('%H:%M')}",
                label_tipe=label_tipe,
            )
            st.plotly_chart(fig_getaran, width="stretch")

            col_arima_getaran, col_lstm_getaran = st.columns(2)
            with col_arima_getaran:
                st.markdown("**ARIMA**")
                data_arima = forecast_arima.dropna(subset=["nilai_getaran_prediksi"]) if not forecast_arima.empty else forecast_arima
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
                    st.info("Tidak ada data ARIMA di jendela ini")

            with col_lstm_getaran:
                st.markdown("**LSTM**")
                data_lstm = forecast_lstm.dropna(subset=["nilai_getaran_prediksi"]) if not forecast_lstm.empty else forecast_lstm
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
                    st.info("Tidak ada data LSTM di jendela ini")

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
