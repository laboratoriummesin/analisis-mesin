"""
EDA (Exploratory Data Analysis) untuk data sensor mesin.
Jalankan: python eda.py

Hasil:
- Ringkasan statistik dicetak ke terminal
- File-file grafik disimpan di folder output_eda/
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns

from api_client import ambil_data_sensor

OUTPUT_DIR = "output_eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


def main():
    print("Mengambil data dari API...")
    df = ambil_data_sensor(limit=5000)
    print(f"Total data: {len(df)} baris\n")

    # ---------- 1. Statistik deskriptif ----------
    print("=== Statistik deskriptif ===")
    print(df[["suhu", "kecepatan_getaran"]].describe())

    print("\n=== Distribusi kondisi ===")
    print(df["kondisi"].value_counts())

    print("\n=== Rata-rata suhu & getaran per kondisi ===")
    print(df.groupby("kondisi")[["suhu", "kecepatan_getaran"]].mean())

    korelasi = df["suhu"].corr(df["kecepatan_getaran"])
    print(f"\nKorelasi suhu vs getaran: {korelasi:.3f}")

    # ---------- 2. Distribusi suhu & getaran ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["suhu"], kde=True, ax=axes[0], color="orange")
    axes[0].set_title("Distribusi Suhu")
    sns.histplot(df["kecepatan_getaran"], kde=True, ax=axes[1], color="steelblue")
    axes[1].set_title("Distribusi Kecepatan Getaran")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/distribusi.png", dpi=150)
    plt.close()

    # ---------- 3. Boxplot per kondisi ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=df, x="kondisi", y="suhu", ax=axes[0])
    axes[0].set_title("Suhu per Kondisi")
    sns.boxplot(data=df, x="kondisi", y="kecepatan_getaran", ax=axes[1])
    axes[1].set_title("Getaran per Kondisi")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/boxplot_per_kondisi.png", dpi=150)
    plt.close()

    # ---------- 4. Time series ----------
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].plot(df["created_at"], df["suhu"], color="orange", linewidth=0.8)
    axes[0].set_ylabel("Suhu (°C)")
    axes[0].set_title("Tren Suhu terhadap Waktu")

    axes[1].plot(df["created_at"], df["kecepatan_getaran"], color="steelblue", linewidth=0.8)
    axes[1].set_ylabel("Kecepatan Getaran")
    axes[1].set_title("Tren Getaran terhadap Waktu")
    axes[1].set_xlabel("Waktu")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/time_series.png", dpi=150)
    plt.close()

    # ---------- 5. Scatter suhu vs getaran, diwarnai kondisi ----------
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="suhu", y="kecepatan_getaran", hue="kondisi", alpha=0.6)
    plt.title(f"Suhu vs Getaran (korelasi = {korelasi:.2f})")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/scatter_suhu_getaran.png", dpi=150)
    plt.close()

    print(f"\nSemua grafik tersimpan di folder: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
