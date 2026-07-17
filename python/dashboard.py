"""
Dashboard Streamlit untuk monitoring data sensor mesin secara live.
Jalankan: streamlit run dashboard.py
Lalu buka browser ke alamat yang muncul di terminal (biasanya http://localhost:8501)
"""

import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from api_client import ambil_data_sensor, ambil_hasil_terbaru

st.set_page_config(page_title="Monitoring Mesin", layout="wide", page_icon="🏭")

# =========================================================================
# CSS kustom — font, warna, dan gaya card
# =========================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .app-header {
        padding: 0.5rem 0 1.5rem 0;
    }
    .app-header h1 {
        font-weight: 800;
        font-size: 2rem;
        color: #F1F5F9;
        margin-bottom: 0.1rem;
    }
    .app-header p {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-top: 0;
    }

    .section-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #F1F5F9;
        margin: 1.75rem 0 0.75rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .metric-card {
        background: #1E293B;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2), 0 6px 16px rgba(0,0,0,0.25);
        border-left: 5px solid #818CF8;
        height: 100%;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F1F5F9;
    }

    .status-banner {
        border-radius: 14px;
        padding: 1rem 1.4rem;
        font-weight: 600;
        font-size: 1rem;
        margin: 1rem 0 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .status-normal { background: rgba(16, 185, 129, 0.12); color: #34D399; border: 1px solid rgba(52, 211, 153, 0.3); }
    .status-peringatan { background: rgba(245, 158, 11, 0.12); color: #FBBF24; border: 1px solid rgba(251, 191, 36, 0.3); }
    .status-tidaknormal { background: rgba(239, 68, 68, 0.12); color: #F87171; border: 1px solid rgba(248, 113, 113, 0.3); }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important;
        background-color: #1E293B !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2), 0 6px 16px rgba(0,0,0,0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_TEMPLATE = "plotly_dark"
WARNA_AKSEN = "#818CF8"
WARNA_ABU = "#64748B"

def kartu_metrik(kolom, label, value, warna="#6366F1"):
    kolom.markdown(
        f"""
        <div class="metric-card" style="border-left-color: {warna};">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def terapkan_transformasi(data: pd.Series, metode: str) -> pd.Series:
    """Terapkan transformasi ke data supaya lebih mendekati distribusi normal."""
    data = data.dropna()
    if metode == "Log (log1p)":
        return np.log1p(data.clip(lower=0))
    elif metode == "Akar Kuadrat (sqrt)":
        return np.sqrt(data.clip(lower=0))
    elif metode == "Box-Cox":
        data_positif = data[data > 0]
        hasil, _ = stats.boxcox(data_positif)
        return pd.Series(hasil, index=data_positif.index)
    return data


# =========================================================================
# Header
# =========================================================================
st.markdown(
    """
    <div class="app-header">
        <h1>🏭 Dashboard Monitoring Sensor Mesin</h1>
        <p>Pemantauan suhu & getaran secara live, lengkap dengan analisis statistik</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Pengaturan")
jumlah_data = st.sidebar.slider("Jumlah data terbaru yang ditampilkan", 50, 10000, 500, step=250)
auto_refresh = st.sidebar.checkbox("Auto-refresh tiap 30 detik", value=False)

# ---------- Ambil data ----------
with st.spinner("Mengambil data terbaru dari API..."):
    df = ambil_data_sensor(limit=jumlah_data)

if df.empty:
    st.warning("Belum ada data.")
    st.stop()

# =========================================================================
# Kartu ringkasan atas
# =========================================================================
data_terakhir = df.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
kartu_metrik(col1, "Suhu Terakhir", f"{data_terakhir['suhu']:.1f} °C", "#F97316")
kartu_metrik(col2, "Getaran Terakhir", f"{data_terakhir['kecepatan_getaran']:.2f}", "#3B82F6")
kartu_metrik(col3, "Kondisi Terakhir", data_terakhir["kondisi"], "#6366F1")
kartu_metrik(col4, "Total Data Ditampilkan", f"{len(df):,}", "#10B981")

if data_terakhir["kondisi"] == "TIDAK NORMAL":
    st.markdown(
        '<div class="status-banner status-tidaknormal">⚠️ Kondisi mesin saat ini TIDAK NORMAL!</div>',
        unsafe_allow_html=True,
    )
elif data_terakhir["kondisi"] == "PERINGATAN":
    st.markdown(
        '<div class="status-banner status-peringatan">⚠️ Kondisi mesin saat ini PERINGATAN</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="status-banner status-normal">✅ Kondisi mesin NORMAL</div>',
        unsafe_allow_html=True,
    )

# =========================================================================
# Statistik Deskriptif (EDA)
# =========================================================================
st.markdown('<div class="section-header">📊 Statistik Deskriptif (EDA)</div>', unsafe_allow_html=True)
with st.container(border=True):
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.markdown("**Ringkasan Suhu & Getaran**")
        st.dataframe(df[["suhu", "kecepatan_getaran"]].describe().round(2), use_container_width=True)
    with col_stat2:
        st.markdown("**Rata-rata per Kondisi**")
        st.dataframe(df.groupby("kondisi")[["suhu", "kecepatan_getaran"]].mean().round(2), use_container_width=True)

    korelasi = df["suhu"].corr(df["kecepatan_getaran"])
    st.metric("Korelasi Suhu vs Getaran", f"{korelasi:.3f}")
    if abs(korelasi) < 0.3:
        st.caption("Korelasi lemah — suhu & getaran tidak berhubungan linear secara kuat.")
    elif abs(korelasi) < 0.7:
        st.caption("Korelasi sedang — ada hubungan, tapi tidak terlalu kuat.")
    else:
        st.caption("Korelasi kuat — suhu & getaran bergerak searah secara konsisten.")

    st.markdown("**Distribusi Jumlah Data per Kondisi**")
    st.dataframe(df["kondisi"].value_counts().rename("jumlah"), use_container_width=True)

# =========================================================================
# Tren waktu
# =========================================================================
st.markdown('<div class="section-header">📈 Tren Suhu & Getaran</div>', unsafe_allow_html=True)
with st.container(border=True):
    fig1 = px.line(df, x="created_at", y="suhu", title="Suhu terhadap Waktu", template=PLOTLY_TEMPLATE)
    fig1.update_traces(line_color="#F97316")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.line(df, x="created_at", y="kecepatan_getaran", title="Getaran terhadap Waktu", template=PLOTLY_TEMPLATE)
    fig2.update_traces(line_color="#3B82F6")
    st.plotly_chart(fig2, use_container_width=True)

# =========================================================================
# Histogram distribusi
# =========================================================================
st.markdown('<div class="section-header">📉 Distribusi Data</div>', unsafe_allow_html=True)
with st.container(border=True):
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        fig_h1 = px.histogram(df, x="suhu", nbins=30, title="Distribusi Suhu", template=PLOTLY_TEMPLATE,
                               color_discrete_sequence=["#F97316"])
        st.plotly_chart(fig_h1, use_container_width=True)
    with col_h2:
        fig_h2 = px.histogram(df, x="kecepatan_getaran", nbins=30, title="Distribusi Getaran", template=PLOTLY_TEMPLATE,
                               color_discrete_sequence=["#3B82F6"])
        st.plotly_chart(fig_h2, use_container_width=True)

# =========================================================================
# Regresi Linear
# =========================================================================
st.markdown('<div class="section-header">📐 Analisis Regresi Linear</div>', unsafe_allow_html=True)
with st.container(border=True):
    st.caption(
        "Melihat seberapa kuat hubungan linear suhu terhadap getaran, "
        "dan getaran terhadap suhu (dua arah), lengkap dengan garis regresi."
    )
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(df["suhu"], df["kecepatan_getaran"])
        fig_r1 = px.scatter(
            df, x="suhu", y="kecepatan_getaran", trendline="ols",
            title="Regresi: Suhu → Getaran", template=PLOTLY_TEMPLATE,
            labels={"suhu": "Suhu (°C)", "kecepatan_getaran": "Kecepatan Getaran"},
            color_discrete_sequence=[WARNA_AKSEN],
        )
        st.plotly_chart(fig_r1, use_container_width=True)
        st.markdown(
            f"**Persamaan:** getaran = {slope1:.4f} × suhu + {intercept1:.4f}  \n"
            f"**R²:** {r_value1 ** 2:.3f}"
        )

    with col_r2:
        slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(df["kecepatan_getaran"], df["suhu"])
        fig_r2 = px.scatter(
            df, x="kecepatan_getaran", y="suhu", trendline="ols",
            title="Regresi: Getaran → Suhu", template=PLOTLY_TEMPLATE,
            labels={"kecepatan_getaran": "Kecepatan Getaran", "suhu": "Suhu (°C)"},
            color_discrete_sequence=[WARNA_AKSEN],
        )
        st.plotly_chart(fig_r2, use_container_width=True)
        st.markdown(
            f"**Persamaan:** suhu = {slope2:.4f} × getaran + {intercept2:.4f}  \n"
            f"**R²:** {r_value2 ** 2:.3f}"
        )

    if max(r_value1 ** 2, r_value2 ** 2) < 0.3:
        st.caption(
            "R² rendah untuk kedua arah — konsisten dengan temuan awal EDA bahwa "
            "hubungan suhu & getaran tidak murni linear (korelasi ~0.43)."
        )

# =========================================================================
# Uji Normalitas
# =========================================================================
st.markdown('<div class="section-header">🔔 Uji Normalitas Distribusi Data</div>', unsafe_allow_html=True)
with st.container(border=True):
    st.caption(
        "Mengecek apakah data suhu & getaran mendekati distribusi normal (lonceng), "
        "menggunakan histogram dibanding kurva normal, Q-Q plot, dan uji Shapiro-Wilk."
    )

    def tampilkan_uji_normalitas(data, label, kolom):
        data_bersih = data.dropna()

        x_range = np.linspace(data_bersih.min(), data_bersih.max(), 200)
        kurva_normal = stats.norm.pdf(x_range, data_bersih.mean(), data_bersih.std())

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=data_bersih, nbinsx=30, histnorm="probability density",
            name="Data aktual", opacity=0.6, marker_color=WARNA_ABU,
        ))
        fig.add_trace(go.Scatter(
            x=x_range, y=kurva_normal, mode="lines",
            name="Kurva normal ideal", line=dict(color="#EF4444", width=2),
        ))
        fig.update_layout(title=f"Histogram vs Kurva Normal — {label}", template=PLOTLY_TEMPLATE)
        kolom.plotly_chart(fig, use_container_width=True)

        qq = stats.probplot(data_bersih, dist="norm")
        x_qq, y_qq = qq[0][0], qq[0][1]
        slope_qq, intercept_qq = qq[1][0], qq[1][1]

        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=x_qq, y=y_qq, mode="markers", name="Data", marker_color=WARNA_AKSEN))
        fig_qq.add_trace(go.Scatter(
            x=x_qq, y=slope_qq * x_qq + intercept_qq, mode="lines",
            name="Garis normal ideal", line=dict(color="#EF4444"),
        ))
        fig_qq.update_layout(
            title=f"Q-Q Plot — {label}", template=PLOTLY_TEMPLATE,
            xaxis_title="Kuantil Teoretis", yaxis_title="Kuantil Data",
        )
        kolom.plotly_chart(fig_qq, use_container_width=True)

        sampel = data_bersih.sample(min(len(data_bersih), 5000), random_state=42)
        stat_sw, p_sw = stats.shapiro(sampel)
        kolom.markdown(f"**Uji Shapiro-Wilk:** statistik = {stat_sw:.4f}, p-value = {p_sw:.4f}")
        if p_sw < 0.05:
            kolom.warning(f"p-value < 0.05 → data **{label}** kemungkinan **TIDAK** berdistribusi normal.")
        else:
            kolom.success(f"p-value ≥ 0.05 → data **{label}** kemungkinan berdistribusi normal.")

    col_n1, col_n2 = st.columns(2)
    tampilkan_uji_normalitas(df["suhu"], "Suhu", col_n1)
    tampilkan_uji_normalitas(df["kecepatan_getaran"], "Kecepatan Getaran", col_n2)

# =========================================================================
# Transformasi Data (memaksa data mendekati normal)
# =========================================================================
st.markdown('<div class="section-header">🔄 Transformasi Data untuk Normalitas</div>', unsafe_allow_html=True)
with st.container(border=True):
    st.caption(
        "Jika data asli tidak berdistribusi normal, transformasi berikut bisa membantu "
        "mendekatkan bentuk distribusinya ke kurva normal — berguna untuk kebutuhan statistik "
        "tertentu (misal uji parametrik) yang mengasumsikan normalitas."
    )
    metode_transformasi = st.selectbox(
        "Pilih metode transformasi",
        ["Log (log1p)", "Akar Kuadrat (sqrt)", "Box-Cox"],
    )

    col_t1, col_t2 = st.columns(2)
    for data_kolom, label, kolom_ui in [
        (df["suhu"], "Suhu", col_t1),
        (df["kecepatan_getaran"], "Kecepatan Getaran", col_t2),
    ]:
        data_asli = data_kolom.dropna()
        data_transformasi = terapkan_transformasi(data_asli, metode_transformasi)

        sampel_asli = data_asli.sample(min(len(data_asli), 5000), random_state=42)
        sampel_baru = data_transformasi.sample(min(len(data_transformasi), 5000), random_state=42)
        stat_asli, p_asli = stats.shapiro(sampel_asli)
        stat_baru, p_baru = stats.shapiro(sampel_baru)

        fig_banding = go.Figure()
        fig_banding.add_trace(go.Histogram(
            x=data_asli, histnorm="probability density", name="Sebelum",
            opacity=0.55, marker_color=WARNA_ABU,
        ))
        fig_banding.add_trace(go.Histogram(
            x=data_transformasi, histnorm="probability density", name="Sesudah",
            opacity=0.55, marker_color=WARNA_AKSEN,
        ))
        fig_banding.update_layout(
            title=f"Sebelum vs Sesudah Transformasi — {label}",
            barmode="overlay", template=PLOTLY_TEMPLATE,
        )
        kolom_ui.plotly_chart(fig_banding, use_container_width=True)

        status_asli = "tidak normal" if p_asli < 0.05 else "normal"
        status_baru = "tidak normal" if p_baru < 0.05 else "normal"
        kolom_ui.markdown(
            f"**Shapiro-Wilk sebelum:** p = {p_asli:.4f} ({status_asli})  \n"
            f"**Shapiro-Wilk sesudah:** p = {p_baru:.4f} ({status_baru})"
        )
        if p_baru > p_asli:
            kolom_ui.success("Transformasi ini membantu mendekatkan data ke distribusi normal.")
        else:
            kolom_ui.info("Transformasi ini tidak banyak membantu untuk data ini — coba metode lain.")

# =========================================================================
# Scatter & distribusi kondisi
# =========================================================================
st.markdown('<div class="section-header">🎯 Hubungan Suhu, Getaran & Kondisi</div>', unsafe_allow_html=True)
with st.container(border=True):
    col_a, col_b = st.columns(2)
    with col_a:
        fig3 = px.scatter(
            df, x="suhu", y="kecepatan_getaran", color="kondisi",
            title="Suhu vs Getaran per Kondisi", template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig3, use_container_width=True)
    with col_b:
        fig4 = px.pie(df, names="kondisi", title="Distribusi Kondisi", template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig4, use_container_width=True)

# =========================================================================
# Prediksi & anomali terbaru
# =========================================================================
st.markdown('<div class="section-header">🤖 Hasil Model Terbaru</div>', unsafe_allow_html=True)
with st.container(border=True):
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
            st.dataframe(
                anomali_terbaru[["data_id", "skor_anomali", "keterangan", "created_at"]],
                use_container_width=True,
            )

# ---------- Tabel data mentah ----------
with st.expander("📄 Lihat data mentah"):
    st.dataframe(df.sort_values("created_at", ascending=False), use_container_width=True)

# ---------- Auto refresh ----------
if auto_refresh:
    time.sleep(30)
    st.rerun()
