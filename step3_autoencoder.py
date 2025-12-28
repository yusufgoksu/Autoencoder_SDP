import matplotlib
matplotlib.use("TkAgg")

import os
import glob
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import textwrap

from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models


# =====================================
# LOG AYARLARI
# =====================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)



# ============================================================
# 🔥 RESAMPLE FONKSİYONU (source_file kaybolmaz!)
# ============================================================
def resample_to_hz(df, target_hz, source_name):
    """
    Sayısal kolonları hedef HZ'e resample eder.
    Numeric olmayan kolonlar (MODE_OP, AXIS_ACT vb.) AE’de kullanılmaz.
    Grafik için gerekli olan source_file resample sonrası geri eklenir.
    """

    if target_hz <= 0:
        raise ValueError("target_hz > 0 olmalıdır")

    period_ms = int(1000 / target_hz)

    # --------------------------
    # Numeric olmayan kolonları geçici olarak ayır
    # --------------------------
    non_numeric = df.select_dtypes(exclude=["number"]).columns.tolist()
    non_numeric = [c for c in non_numeric if c not in ["timestamp"]]

    df_numeric = df.drop(columns=non_numeric, errors="ignore")

    # --------------------------
    # RESAMPLE
    # --------------------------
    df_res = (
        df_numeric
        .set_index("timestamp")
        .resample(f"{period_ms}ms")
        .mean()
        .interpolate(method="linear")
        .reset_index()
    )

    # --------------------------
    # source_file geri ekleniyor
    # --------------------------
    df_res["source_file"] = source_name

    return df_res



# ============================================================
# 🔥 TÜM PIPELINE – (batch_size, target_hz parametreli)
# ============================================================
def run_anomaly_detection(batch_size, target_hz):

    logging.info(f"\n=========== RUNNING PIPELINE (batch={batch_size}, hz={target_hz}) ===========")

    # =====================================
    # 1) CSV DOSYALARI
    # =====================================
    DATA_DIR = "Data"
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

    if not csv_files:
        raise RuntimeError("Data klasöründe CSV bulunamadı!")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, sep=";", engine="python")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

        # 🔥 RESAMPLE
        df = resample_to_hz(df, target_hz, os.path.basename(f))
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.sort_values("timestamp").reset_index(drop=True)

    # =====================================
    # 2) FEATURE SET (Autoencoder için)
    # =====================================
    torque_cols = [f"TORQUE_A{i}" for i in range(1, 7)]
    curr_cols   = [f"CURR_A{i}" for i in range(1, 7)]
    temp_cols   = [f"MOT_TEMP_A{i}" for i in range(1, 7)]
    vel_cols    = [f"VEL_AXIS_ACT_A{i}" for i in range(1, 7)]

    feat_cols = torque_cols + curr_cols + temp_cols + vel_cols

    # AE için sayısal veri
    X = df_all[feat_cols].apply(pd.to_numeric, errors="coerce")

    valid_idx = X.dropna().index
    X = X.loc[valid_idx]
    df_all = df_all.loc[valid_idx].reset_index(drop=True)

    # =====================================
    # 3) SCALE
    # =====================================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =====================================
    # 4) AUTOENCODER MODEL
    # =====================================
    model = models.Sequential([
        layers.Input(shape=(X_scaled.shape[1],)),
        layers.Dense(16, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(X_scaled.shape[1], activation="linear")
    ])

    model.compile(optimizer="adam", loss="mse")

    logging.info("Autoencoder eğitimi başlıyor...")

    history = model.fit(
        X_scaled, X_scaled,
        epochs=40,
        batch_size=batch_size,
        shuffle=False,
        verbose=0
    )

    logging.info("Eğitim bitti ✓")

    # =====================================
    # 5) LOSS GRAFİĞİ
    # =====================================
    fig_loss, ax_loss = plt.subplots(figsize=(8,4))
    ax_loss.plot(history.history["loss"], linewidth=2)
    ax_loss.set_title(f"Training Loss (batch={batch_size}, hz={target_hz})")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True)
    plt.tight_layout()
    plt.show(block=True)

    # =====================================
    # 6) ANOMALY SCORE
    # =====================================
    X_pred = model.predict(X_scaled)
    recon_error = np.mean((X_scaled - X_pred)**2, axis=1)

    median = np.median(recon_error)
    mad = np.median(np.abs(recon_error - median))
    threshold = median + 12 * 1.4826 * mad

    logging.info(f"Threshold = {threshold:.4f}")

    # =====================================
    # 7) ANOMALY INDEX
    # =====================================
    anomaly_idx = np.where(recon_error > threshold)[0]

    logging.info(f"Toplam anomali sayısı = {len(anomaly_idx)}")

    for i in anomaly_idx:
        logging.info(f"ANOMALİ → {df_all['timestamp'].iloc[i]}  err={recon_error[i]:.3f}")

    # =====================================
    # 8) ANOMALY GRAPH (DOSYA ADLARI AŞAĞIDA)
    # =====================================
    time_axis = df_all["timestamp"]

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.plot(time_axis, recon_error, linewidth=1, label="Reconstruction Error")
    ax.axhline(threshold, color="red", linestyle="--", label="Threshold")
    ax.set_ylabel("Reconstruction Error")
    ax.set_title(f"KUKA Robot – Anomaly Detection (batch={batch_size}, hz={target_hz})")
    ax.grid(True)
    ax.legend()

    # --- BLOK ARKA PLANLARI ---
    file_ranges = df_all.groupby("source_file")["timestamp"].agg(["min", "max"]).reset_index()
    colors = ["#e3f2fd", "#f1f8e9", "#fce4ec", "#fff3e0", "#ede7f6"]
    y_offset = -0.23  # yazıları x ekseninin altında göstermek için

    for idx, row in file_ranges.iterrows():
        file_name = row["source_file"]
        start_time = row["min"]
        end_time = row["max"]

        # blok
        ax.axvspan(start_time, end_time, color=colors[idx % len(colors)], alpha=0.35)

        # blok çizgileri
        ax.axvline(start_time, color="black", linewidth=1.3)
        ax.axvline(end_time, color="black", linewidth=1.3)

        # dosya adı
        mid_time = start_time + (end_time - start_time) / 2
        wrapped = "\n".join(textwrap.wrap(file_name, width=30))

        ax.text(
            mid_time, y_offset, wrapped,
            ha="center", va="top",
            fontsize=9,
            transform=ax.get_xaxis_transform(),
            bbox=dict(facecolor="white", alpha=0.9)
        )

    # Tarih formatı
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S"))

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

    # =====================================
    # RETURN
    # =====================================
    return (
        recon_error,
        threshold,
        anomaly_idx,
        df_all["timestamp"].iloc[anomaly_idx].tolist(),
        X_scaled,
        X_pred,
        feat_cols
    )



# ============================================================
# 🔥 ROOT CAUSE
# ============================================================
def analyze_root_cause(X_scaled, X_pred, anomaly_index, feature_names):

    err = (X_scaled[anomaly_index] - X_pred[anomaly_index]) ** 2

    mean = err.mean()
    std = err.std() + 1e-9
    z = (err - mean) / std

    df = pd.DataFrame({
        "Feature": feature_names,
        "Error": err,
        "Z-Score": z
    })

    df = df.sort_values("Z-Score", ascending=False)
    root = df.iloc[0]

    return df, root
