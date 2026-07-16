"""
Dashboard Streamlit untuk monitoring data sensor mesin secara live.
Jalankan: streamlit run dashboard.py
Lalu buka browser ke alamat yang muncul di terminal (biasanya http://localhost:8501)
"""

import time

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

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

# ---------- Statistik Deskriptif (EDA) ----------
st.subheader("📊 Statistik Deskriptif (EDA)")

col_stat1, col_stat2 = st.columns(2)
with col_stat1:
    st.markdown("**Ringkasan Suhu & Getaran**")
    st.dataframe(df[["suhu", "kecepatan_getaran"]].describe().round(2))

with col_stat2:
    st.markdown("**Rata-rata per Kondisi**")
    st.dataframe(df.groupby("kondisi")[["suhu", "kecepatan_getaran"]].mean().round(2))

korelasi = df["suhu"].corr(df["kecepatan_getaran"])
st.metric("Korelasi Suhu vs Getaran", f"{korelasi:.3f}")
if abs(korelasi) < 0.3:
    st.caption("Korelasi lemah — suhu & getaran tidak berhubungan linear secara kuat.")
elif abs(korelasi) < 0.7:
    st.caption("Korelasi sedang — ada hubungan, tapi tidak terlalu kuat.")
else:
    st.caption("Korelasi kuat — suhu & getaran bergerak searah secara konsisten.")

st.markdown("**Distribusi Jumlah Data per Kondisi**")
st.dataframe(df["kondisi"].value_counts().rename("jumlah"))

# ---------- Grafik time series ----------
st.subheader("Tren Suhu & Getaran")
fig1 = px.line(df, x="created_at", y="suhu", title="Suhu terhadap Waktu")
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.line(df, x="created_at", y="kecepatan_getaran", title="Getaran terhadap Waktu")
st.plotly_chart(fig2, use_container_width=True)

# ---------- Histogram distribusi ----------
col_h1, col_h2 = st.columns(2)
with col_h1:
    fig_h1 = px.histogram(df, x="suhu", nbins=30, title="Distribusi Suhu")
    st.plotly_chart(fig_h1, use_container_width=True)
with col_h2:
    fig_h2 = px.histogram(df, x="kecepatan_getaran", nbins=30, title="Distribusi Getaran")
    st.plotly_chart(fig_h2, use_container_width=True)

# ---------- Regresi Linear: Suhu & Getaran ----------
st.subheader("📈 Analisis Regresi Linear")
st.caption(
    "Melihat seberapa kuat hubungan linear suhu terhadap getaran, "
    "dan getaran terhadap suhu (dua arah), lengkap dengan garis regresi."
)

col_r1, col_r2 = st.columns(2)

with col_r1:
    # Regresi: suhu (X) -> getaran (Y)
    slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(
        df["suhu"], df["kecepatan_getaran"]
    )
    fig_r1 = px.scatter(
        df, x="suhu", y="kecepatan_getaran",
        trendline="ols",
        title="Regresi: Suhu → Getaran",
        labels={"suhu": "Suhu (°C)", "kecepatan_getaran": "Kecepatan Getaran"},
    )
    st.plotly_chart(fig_r1, use_container_width=True)
    st.markdown(
        f"**Persamaan:** getaran = {slope1:.4f} × suhu + {intercept1:.4f}  \n"
        f"**R²:** {r_value1 ** 2:.3f}"
    )

with col_r2:
    # Regresi: getaran (X) -> suhu (Y)
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(
        df["kecepatan_getaran"], df["suhu"]
    )
    fig_r2 = px.scatter(
        df, x="kecepatan_getaran", y="suhu",
        trendline="ols",
        title="Regresi: Getaran → Suhu",
        labels={"kecepatan_getaran": "Kecepatan Getaran", "suhu": "Suhu (°C)"},
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

# ---------- Uji Normalitas Distribusi ----------
st.subheader("🔔 Uji Normalitas Distribusi Data")
st.caption(
    "Mengecek apakah data suhu & getaran mendekati distribusi normal (lonceng), "
    "menggunakan histogram dibanding kurva normal, Q-Q plot, dan uji Shapiro-Wilk."
)


def tampilkan_uji_normalitas(data, label, kolom):
    data_bersih = data.dropna()

    # Histogram vs kurva normal ideal
    x_range = np.linspace(data_bersih.min(), data_bersih.max(), 200)
    kurva_normal = stats.norm.pdf(x_range, data_bersih.mean(), data_bersih.std())

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data_bersih, nbinsx=30, histnorm="probability density",
        name="Data aktual", opacity=0.6
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=kurva_normal, mode="lines",
        name="Kurva normal ideal", line=dict(color="red", width=2)
    ))
    fig.update_layout(title=f"Histogram vs Kurva Normal — {label}")
    kolom.plotly_chart(fig, use_container_width=True)

    # Q-Q plot
    qq = stats.probplot(data_bersih, dist="norm")
    x_qq, y_qq = qq[0][0], qq[0][1]
    slope_qq, intercept_qq = qq[1][0], qq[1][1]

    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(x=x_qq, y=y_qq, mode="markers", name="Data"))
    fig_qq.add_trace(go.Scatter(
        x=x_qq, y=slope_qq * x_qq + intercept_qq, mode="lines",
        name="Garis normal ideal", line=dict(color="red")
    ))
    fig_qq.update_layout(
        title=f"Q-Q Plot — {label}",
        xaxis_title="Kuantil Teoretis", yaxis_title="Kuantil Data"
    )
    kolom.plotly_chart(fig_qq, use_container_width=True)

    # Uji Shapiro-Wilk (maks 5000 sampel, batasan dari scipy)
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
