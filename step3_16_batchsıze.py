# =====================================
# MATPLOTLIB BACKEND (INTERAKTİF)
# =====================================
import matplotlib
matplotlib.use("TkAgg")

import os
import glob
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

logging.info("STEP-3 anomaly detection başlatıldı")


# =====================================
# 1️⃣ TÜM CSV DOSYALARINI OKU
# =====================================
DATA_DIR = "Data"
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

if not csv_files:
    raise RuntimeError("Data klasöründe CSV bulunamadı")

logging.info("Okunacak CSV dosyaları:")
for f in csv_files:
    logging.info(f"  - {os.path.basename(f)}")

dfs = []
for f in csv_files:
    df = pd.read_csv(f, sep=";", engine="python")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["source_file"] = os.path.basename(f)
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all.dropna(subset=["timestamp"])
df_all = df_all.sort_values("timestamp").reset_index(drop=True)

logging.info(f"Toplam satır (birleşik): {len(df_all)}")


# =====================================
# 2️⃣ FEATURE SEÇ
# =====================================
torque_cols = [f"TORQUE_A{i}" for i in range(1, 7)]
curr_cols   = [f"CURR_A{i}" for i in range(1, 7)]
temp_cols   = [f"MOT_TEMP_A{i}" for i in range(1, 7)]
vel_cols    = [f"VEL_AXIS_ACT_A{i}" for i in range(1, 7)]

feat_cols = torque_cols + curr_cols + temp_cols + vel_cols

X = df_all[feat_cols].apply(pd.to_numeric, errors="coerce")
valid_idx = X.dropna().index

X = X.loc[valid_idx]
df_all = df_all.loc[valid_idx].reset_index(drop=True)

logging.info(f"Feature matrix shape: {X.shape}")


# =====================================
# 3️⃣ SCALE
# =====================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
logging.info("Feature scaling tamamlandı (StandardScaler)")


# =====================================
# 4️⃣ AUTOENCODER MODEL (batch_size = 16)
# =====================================
input_dim = X_scaled.shape[1]

model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),   # bottleneck
    layers.Dense(16, activation="relu"),
    layers.Dense(input_dim, activation="linear")
])

model.compile(optimizer="adam", loss="mse")

logging.info("Autoencoder eğitimi başlıyor (batch=16)")

history = model.fit(
    X_scaled, X_scaled,
    epochs=40,
    batch_size=16,       # 🔥 SADECE BURASI DEĞİŞTİ
    shuffle=False,
    verbose=0
)

logging.info("Autoencoder eğitimi tamamlandı")


# =====================================
# 5️⃣ LOSS GRAFİĞİ
# =====================================
fig_loss, ax_loss = plt.subplots(figsize=(8,4))

ax_loss.plot(history.history["loss"], linewidth=2, label="Train Loss")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("MSE Loss")
ax_loss.set_title("Autoencoder Training Loss (Batch=16)")
ax_loss.legend()
ax_loss.grid(True)

plt.tight_layout()
plt.show(block=True)


# =====================================
# 6️⃣ ANOMALİ SKORU + THRESHOLD
# =====================================
X_pred = model.predict(X_scaled)
recon_error = np.mean((X_scaled - X_pred) ** 2, axis=1)

median = np.median(recon_error)
mad = np.median(np.abs(recon_error - median))
threshold = median + 12 * 1.4826 * mad

logging.info(
    f"Reconstruction error min/mean/max: "
    f"{recon_error.min():.3f} / {recon_error.mean():.3f} / {recon_error.max():.3f}"
)
logging.info(f"Anomaly threshold (MAD): {threshold:.3f}")


# =====================================
# 7️⃣ ANOMALİ ZAMANLARI LOGLA
# =====================================
anomaly_idx = np.where(recon_error > threshold)[0]

logging.info(f"Toplam anomali sayısı: {len(anomaly_idx)}")

for i in anomaly_idx:
    ts = df_all["timestamp"].iloc[i]
    err = recon_error[i]
    src = df_all["source_file"].iloc[i]
    logging.info(f"ANOMALİ → {ts} | error={err:.3f} | file={src}")


# =====================================
# 8️⃣ BLOK GRAFİK + LABEL ÇAKIŞMA ÖNLEME
# =====================================
import textwrap

time_axis = df_all["timestamp"]
fig, ax = plt.subplots(figsize=(18,7))

ax.plot(time_axis, recon_error, linewidth=1, label="Reconstruction Error")
ax.axhline(threshold, color="red", linestyle="--", label="Threshold")
ax.set_ylabel("Reconstruction Error")
ax.set_title("KUKA Robot – Anomaly Detection (Batch=16)")
ax.grid(True)

# Legend dışarıya alınır
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

# Her csv dosyasının tek blok aralığını bul
file_ranges = df_all.groupby("source_file")["timestamp"].agg(["min", "max"]).reset_index()

colors = ["#e3f2fd", "#f1f8e9", "#fce4ec", "#fff3e0", "#ede7f6"]
y_offset = -0.23

for idx, row in file_ranges.iterrows():

    file_name = row["source_file"]
    start_time = row["min"]
    end_time = row["max"]

    # BLOK ARKAPLAN
    ax.axvspan(start_time, end_time, color=colors[idx % len(colors)], alpha=0.35)

    # BLOK SINIR ÇİZGİLERİ
    ax.axvline(start_time, color="black", linewidth=1.3)
    ax.axvline(end_time,   color="black", linewidth=1.3)

    # LABEL
    mid_time = start_time + (end_time - start_time) / 2
    wrapped = "\n".join(textwrap.wrap(file_name, width=30))

    ax.text(
        mid_time, y_offset,
        wrapped,
        ha="center", va="top",
        fontsize=9,
        transform=ax.get_xaxis_transform(),
        bbox=dict(facecolor="white", alpha=0.9)
    )

# X eksenini sadece saat göster
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

fig.autofmt_xdate()
plt.tight_layout()
plt.show()


logging.info("STEP-3 başarıyla tamamlandı (batch=16)")
