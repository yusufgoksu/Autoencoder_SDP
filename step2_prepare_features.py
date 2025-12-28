import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# =====================================
# 1️⃣ TÜM CSV DOSYALARINI OKU
# =====================================
DATA_DIR = "Data"
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

dfs = []

for f in csv_files:
    df = pd.read_csv(f, sep=";", engine="python")
    df["source_file"] = os.path.basename(f)
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# =====================================
# 2️⃣ FEATURE KOLONLARI
# =====================================
torque_cols = [f"TORQUE_A{i}" for i in range(1, 7)]
curr_cols   = [f"CURR_A{i}" for i in range(1, 7)]
temp_cols   = [f"MOT_TEMP_A{i}" for i in range(1, 7)]
vel_cols    = [f"VEL_AXIS_ACT_A{i}" for i in range(1, 7)]

feat_cols = torque_cols + curr_cols + temp_cols + vel_cols

# =====================================
# 3️⃣ FEATURE MATRIX
# =====================================
X = df_all[feat_cols].copy()

for c in feat_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce")

X = X.dropna()

print("Toplam örnek sayısı:", X.shape[0])
print("Feature sayısı:", X.shape[1])

# =====================================
# 4️⃣ SCALE (TEK SCALER!)
# =====================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaled shape:", X_scaled.shape)
print("Mean (ilk 5):", X_scaled.mean(axis=0)[:5])
print("Std (ilk 5):", X_scaled.std(axis=0)[:5])
