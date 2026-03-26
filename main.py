from step3_autoencoder import train_autoencoder
from step4_anomoly_detection import run_anomaly_detection


def normalize_timestamps(ts_list):
    return set(t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] for t in ts_list)


print("\n=========== TRAINING MODELS ===========\n")

model8, scaler8, feats8, _ = train_autoencoder(batch_size=8)
model16, scaler16, feats16, _ = train_autoencoder(batch_size=16)
model32, scaler32, feats32, _ = train_autoencoder(batch_size=32)

print("\n=========== TESTING (ENSEMBLE) ===========\n")

res8, idx8 = run_anomaly_detection(model8, scaler8, feats8)
res16, idx16 = run_anomaly_detection(model16, scaler16, feats16)
res32, idx32 = run_anomaly_detection(model32, scaler32, feats32)

ts8  = res8.loc[idx8, "timestamp"]
ts16 = res16.loc[idx16, "timestamp"]
ts32 = res32.loc[idx32, "timestamp"]

ms8  = normalize_timestamps(ts8)
ms16 = normalize_timestamps(ts16)
ms32 = normalize_timestamps(ts32)

common_all   = ms8 & ms16 & ms32
common_8_16  = ms8 & ms16
common_8_32  = ms8 & ms32
common_16_32 = ms16 & ms32

unique_8  = ms8  - (ms16 | ms32)
unique_16 = ms16 - (ms8  | ms32)
unique_32 = ms32 - (ms8  | ms16)

print("\n==================== ENSEMBLE RESULT ====================\n")

print("🔥 3 MODEL ORTAK:")
print("  →", "\n  → ".join(sorted(common_all)) if common_all else "  ❌ Yok")

print("\n🔥 8 & 16 ORTAK:")
only_8_16 = sorted(common_8_16 - common_all)
print("  →", "\n  → ".join(only_8_16) if only_8_16 else "  ❌ Yok")

print("\n🔥 8 & 32 ORTAK:")
only_8_32 = sorted(common_8_32 - common_all)
print("  →", "\n  → ".join(only_8_32) if only_8_32 else "  ❌ Yok")

print("\n🔥 16 & 32 ORTAK:")
only_16_32 = sorted(common_16_32 - common_all)
print("  →", "\n  → ".join(only_16_32) if only_16_32 else "  ❌ Yok")

print("\n🔥 SADECE 8:")
only_8 = sorted(unique_8)
print("  →", "\n  → ".join(only_8) if only_8 else "  ❌ Yok")

print("\n🔥 SADECE 16:")
only_16 = sorted(unique_16)
print("  →", "\n  → ".join(only_16) if only_16 else "  ❌ Yok")

print("\n🔥 SADECE 32:")
only_32 = sorted(unique_32)
print("  →", "\n  → ".join(only_32) if only_32 else "  ❌ Yok")

print("\n==========================================================\n")


print("\n==================== ROOT CAUSE ====================\n")


def get_root_cause_line(df, idx):
    vals = [
        df["root_cause_1"].iloc[idx],
        df["root_cause_2"].iloc[idx],
        df["root_cause_3"].iloc[idx]
    ]
    vals = [v for v in vals if isinstance(v, str) and v.strip() != ""]
    return vals


intersection_2plus = (common_8_16 | common_8_32 | common_16_32)

if not intersection_2plus:
    print("⚠️ Ortak anomaly yok")
else:
    for ts in sorted(intersection_2plus):
        print(f"\n🔥 Ortak anomaly: {ts}")

        found = False
        for df, idx_list, src in [
            (res16, idx16, "batch=16"),
            (res8, idx8, "batch=8"),
            (res32, idx32, "batch=32")
        ]:
            matches = [
                i for i in idx_list
                if df["timestamp"].iloc[i].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] == ts
            ]

            if matches:
                idx = matches[0]
                causes = get_root_cause_line(df, idx)

                if causes:
                    print(f"→ ROOT CAUSE ({src}):")
                    for c in causes:
                        print(f"   - {c}")
                else:
                    print(f"→ ROOT CAUSE ({src}): bilgi yok")

                found = True
                break

        if not found:
            print("→ ROOT CAUSE: eslesen satir bulunamadi")

        print("-------------------------------------------")