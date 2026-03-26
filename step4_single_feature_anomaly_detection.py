import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(BASE_DIR, "Data_For_Test")
TIMESTAMP_COL = "timestamp"
SEP = ";"

TEST_FILES = [
    "testdata2.csv",
]


def load_test_data():
    all_dfs = []

    for fname in TEST_FILES:
        path = os.path.join(TEST_DATA_DIR, fname)

        if not os.path.exists(path):
            print(f"UYARI: Dosya bulunamadi -> {path}")
            continue

        print(f"Okunuyor: {fname}")
        df = pd.read_csv(path, sep=SEP, engine="python")

        if TIMESTAMP_COL not in df.columns:
            print(f"UYARI: {fname} icinde timestamp kolonu yok")
            continue

        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
        df = df.dropna(subset=[TIMESTAMP_COL]).copy()
        df["source_file"] = fname

        all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("Test verisi yuklenemedi")

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all = df_all.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    print("\n====================")
    print("TEST VERISI")
    print("====================")
    print("Toplam satir:", len(df_all))
    print("Baslangic:", df_all[TIMESTAMP_COL].min())
    print("Bitis    :", df_all[TIMESTAMP_COL].max())

    return df_all


def prepare_test_features(df_all, feat_cols, scaler):
    missing_cols = [c for c in feat_cols if c not in df_all.columns]
    if missing_cols:
        raise ValueError(f"Test verisinde eksik kolonlar var: {missing_cols}")

    X = df_all[feat_cols].copy()

    for c in feat_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    valid_idx = X.dropna().index
    X = X.loc[valid_idx].reset_index(drop=True)
    df_all = df_all.loc[valid_idx].reset_index(drop=True)

    X_scaled = scaler.transform(X)

    print("\nTEST FEATURE OZETI")
    print("Toplam ornek sayisi:", X.shape[0])
    print("Feature sayisi:", X.shape[1])
    print("Scaled shape:", X_scaled.shape)

    return df_all, X, X_scaled


def feature_to_text(feature_name, direction):
    direction_text = "yuksek" if direction == "high" else "dusuk"

    if feature_name.startswith("TORQUE_A"):
        return f"{feature_name} normalden {direction_text}"
    if feature_name.startswith("CURR_A"):
        return f"{feature_name} normalden {direction_text}"
    if feature_name.startswith("MOT_TEMP_A"):
        return f"{feature_name} normalden {direction_text}"
    if feature_name.startswith("VEL_AXIS_ACT_A"):
        return f"{feature_name} normalden {direction_text}"

    return f"{feature_name} normalden {direction_text}"


def run_anomaly_detection(model, scaler, feat_cols):
    print("STEP-4 SINGLE FEATURE ANOMALY DETECTION baslatildi")

    df_all = load_test_data()
    df_all, X, X_scaled = prepare_test_features(df_all, feat_cols, scaler)

    X_pred = model.predict(X_scaled, verbose=0)

    error_matrix = (X_scaled - X_pred) ** 2
    signed_diff = X_scaled - X_pred

    max_feature_error = np.max(error_matrix, axis=1)

    top_feature_idx = np.argmax(error_matrix, axis=1)
    top_feature_names = [feat_cols[i] for i in top_feature_idx]

    top_feature_signed_diff = signed_diff[np.arange(len(signed_diff)), top_feature_idx]

    threshold_feature = max_feature_error.mean() + 3 * max_feature_error.std()

    anomaly_idx = np.where(max_feature_error > threshold_feature)[0]

    print("\n====================")
    print("SINGLE FEATURE ANOMALY DETECTION SONUCLARI")
    print("====================")
    print("Toplam test ornegi:", len(df_all))
    print("Anomaly sayisi:", len(anomaly_idx))
    print("Threshold:", threshold_feature)
    print("Max feature error min :", max_feature_error.min())
    print("Max feature error mean:", max_feature_error.mean())
    print("Max feature error max :", max_feature_error.max())

    result_df = df_all.copy()
    result_df["max_feature_error"] = max_feature_error
    result_df["is_anomaly"] = False
    result_df["top_error_feature"] = top_feature_names
    result_df["top_feature_signed_diff"] = top_feature_signed_diff
    result_df["root_cause_text"] = ""

    result_df.loc[anomaly_idx, "is_anomaly"] = True

    print("\nTUM ANOMALYLER:")
    if len(anomaly_idx) == 0:
        print("Anomaly bulunamadi.")
    else:
        for i in anomaly_idx:
            ts = result_df[TIMESTAMP_COL].iloc[i]
            err = max_feature_error[i]

            feat = top_feature_names[i]
            diff = top_feature_signed_diff[i]

            direction = "high" if diff > 0 else "low"
            comment = feature_to_text(feat, direction)
            result_df.at[i, "root_cause_text"] = comment

            print(f"\nANOMALY -> saat={ts.strftime('%H:%M:%S.%f')[:-3]} | error={err:.6f}")
            print(f"  1. {comment} | scaled={diff:.3f}")

    print("\nSonuclar sadece ekranda gosteriliyor, dosyaya kaydedilmiyor.")

    fig, ax = plt.subplots(figsize=(18, 7))

    ax.plot(df_all[TIMESTAMP_COL], max_feature_error, linewidth=1, label="Max Feature Error")
    ax.axhline(threshold_feature, color="red", linestyle="--", label="Threshold")

    if len(anomaly_idx) > 0:
        ax.scatter(
            df_all[TIMESTAMP_COL].iloc[anomaly_idx],
            max_feature_error[anomaly_idx],
            s=20,
            label="Single Feature Anomalies"
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Max Single Feature Error")
    ax.set_title("Single Feature Anomaly Detection on Test Data")
    ax.grid(True)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S.%f"))

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

    return result_df, anomaly_idx