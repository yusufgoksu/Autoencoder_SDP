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
    "testdata3.csv",
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

    valid_mask = X.notna().all(axis=1)
    valid_idx = df_all.index[valid_mask]

    X_valid = X.loc[valid_idx].copy()
    X_scaled = scaler.transform(X_valid)

    print("\nTEST FEATURE OZETI")
    print("Toplam full satir      :", len(df_all))
    print("Gecerli satir sayisi   :", len(valid_idx))
    print("Atilan satir sayisi    :", len(df_all) - len(valid_idx))
    print("Feature sayisi         :", X_valid.shape[1])
    print("Scaled shape           :", X_scaled.shape)

    return X_valid, X_scaled, valid_idx


def feature_to_text(feature_name, direction):
    direction_text = "yuksek" if direction == "high" else "dusuk"

    if feature_name.startswith("TORQUE_A"):
        axis = feature_name.replace("TORQUE_", "")
        return f"{axis} torque normalden {direction_text}"

    if feature_name.startswith("CURR_A"):
        axis = feature_name.replace("CURR_", "")
        return f"{axis} current normalden {direction_text}"

    if feature_name.startswith("MOT_TEMP_A"):
        axis = feature_name.replace("MOT_TEMP_", "")
        return f"{axis} motor temperature normalden {direction_text}"

    if feature_name.startswith("VEL_AXIS_ACT_A"):
        axis = feature_name.replace("VEL_AXIS_ACT_", "")
        return f"{axis} velocity normalden {direction_text}"

    return f"{feature_name} normalden {direction_text}"


def analyze_root_cause(X_scaled, feat_cols, valid_row_pos, top_n=3):
    row = X_scaled[valid_row_pos]
    abs_vals = np.abs(row)
    top_idx = np.argsort(abs_vals)[::-1][:top_n]

    reasons = []
    details = []

    for i in top_idx:
        feat = feat_cols[i]
        val = row[i]
        direction = "high" if val > 0 else "low"
        text = feature_to_text(feat, direction)

        reasons.append(text)
        details.append({
            "feature": feat,
            "scaled_value": float(val),
            "abs_scaled_value": float(abs(val)),
            "comment": text
        })

    return reasons, details


def run_anomaly_detection(model, scaler, feat_cols, fixed_threshold):
    print("STEP-4 ANOMALY DETECTION baslatildi")

    full_df = load_test_data()
    X_valid, X_scaled, valid_idx = prepare_test_features(full_df, feat_cols, scaler)

    X_pred = model.predict(X_scaled, verbose=0)
    recon_error_valid = np.mean((X_scaled - X_pred) ** 2, axis=1)

    anomaly_valid_pos = np.where(recon_error_valid > fixed_threshold)[0]
    anomaly_full_idx = valid_idx[anomaly_valid_pos]

    print("\n====================")
    print("ANOMALY DETECTION SONUCLARI")
    print("====================")
    print("Toplam full test ornegi :", len(full_df))
    print("Gecerli test ornegi     :", len(valid_idx))
    print("Anomaly sayisi          :", len(anomaly_full_idx))
    print("Fixed threshold         :", fixed_threshold)
    print("Max test error          :", np.max(recon_error_valid))
    print("Top 10 test error       :", np.sort(recon_error_valid)[-10:])

    result_df = full_df.copy()
    result_df["reconstruction_error"] = np.nan
    result_df["is_anomaly"] = False
    result_df["root_cause_1"] = ""
    result_df["root_cause_2"] = ""
    result_df["root_cause_3"] = ""

    result_df.loc[valid_idx, "reconstruction_error"] = recon_error_valid
    result_df.loc[anomaly_full_idx, "is_anomaly"] = True

    print("\nTUM ANOMALYLER:")
    if len(anomaly_full_idx) == 0:
        print("Anomaly bulunamadi.")
    else:
        for valid_pos, full_i in zip(anomaly_valid_pos, anomaly_full_idx):
            ts = result_df.loc[full_i, TIMESTAMP_COL]
            err = result_df.loc[full_i, "reconstruction_error"]

            reasons, details = analyze_root_cause(X_scaled, feat_cols, valid_pos, top_n=3)

            if len(reasons) > 0:
                result_df.at[full_i, "root_cause_1"] = reasons[0]
            if len(reasons) > 1:
                result_df.at[full_i, "root_cause_2"] = reasons[1]
            if len(reasons) > 2:
                result_df.at[full_i, "root_cause_3"] = reasons[2]

            print(f"\nANOMALY -> saat={ts.strftime('%H:%M:%S.%f')[:-3]} | error={err:.6f}")
            for j, d in enumerate(details, start=1):
                print(f"  {j}. {d['comment']} | scaled={d['scaled_value']:.3f}")

    plot_df = result_df.dropna(subset=["reconstruction_error"]).copy()

    fig, ax = plt.subplots(figsize=(22, 7))
    ax.plot(
        plot_df[TIMESTAMP_COL],
        plot_df["reconstruction_error"],
        linewidth=1,
        label="Reconstruction Error"
    )
    ax.axhline(
        fixed_threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold = {fixed_threshold:.6f}"
    )

    anomaly_df = plot_df[plot_df["is_anomaly"] == True]
    if not anomaly_df.empty:
        ax.scatter(
            anomaly_df[TIMESTAMP_COL],
            anomaly_df["reconstruction_error"],
            s=30,
            label="Anomalies"
        )

    ax.set_xlim(full_df[TIMESTAMP_COL].min(), full_df[TIMESTAMP_COL].max())
    ax.set_xlabel("Time")
    ax.set_ylabel("Reconstruction Error")
    ax.set_title("Anomaly Detection on Full Test Data")
    ax.grid(True)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

    return result_df, anomaly_full_idx.tolist()