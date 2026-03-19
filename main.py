# ============================================================
#   ENSEMBLE ANOMALY DETECTION MAIN SCRIPT + ROOT CAUSE
# ============================================================

from step3_autoencoder import run_anomaly_detection, analyze_root_cause


# ============================================================
# Zamanı saniyeye yuvarla
# ============================================================
def normalize_seconds(ts_list):
    return set(t.strftime("%Y-%m-%d %H:%M:%S") for t in ts_list)



# ============================================================
# 1️⃣ MODELLERİ ÇALIŞTIR (HER MODEL AYNI FREKANS İLE)
# ============================================================

TARGET_HZ = 10 # frekans ayar yerı

print(f"\n=========== ENSEMBLE RUN (HZ={TARGET_HZ}) ===========\n")

err8,  thr8,  idx8,  ts8,  Xs8,  Xp8,  feats = run_anomaly_detection(8,  TARGET_HZ)
err16, thr16, idx16, ts16, Xs16, Xp16, _    = run_anomaly_detection(16, TARGET_HZ)
err32, thr32, idx32, ts32, Xs32, Xp32, _    = run_anomaly_detection(32, TARGET_HZ)



# ============================================================
# 2️⃣ SANİYE DÜZELEMİ
# ============================================================
sec8  = normalize_seconds(ts8)
sec16 = normalize_seconds(ts16)
sec32 = normalize_seconds(ts32)



# ============================================================
# 3️⃣ ORTAK ANOMALİLER
# ============================================================
common_all   = sec8 & sec16 & sec32
common_8_16  = sec8 & sec16
common_8_32  = sec8 & sec32
common_16_32 = sec16 & sec32

unique_8  = sec8  - (sec16 | sec32)
unique_16 = sec16 - (sec8  | sec32)
unique_32 = sec32 - (sec8  | sec16)



# ============================================================
# 4️⃣ SONUÇ YAZDIR
# ============================================================
print("\n==================== ENSEMBLE RESULT ====================\n")

print("🔥 3 MODEL ORTAK:")
print("  →", "\n  → ".join(sorted(common_all)) if common_all else "  ❌ Yok")

print("\n🔥 8 & 16 ORTAK:")
print("  →", "\n  → ".join(sorted(common_8_16 - common_all)) or "  ❌ Yok")

print("\n🔥 8 & 32 ORTAK:")
print("  →", "\n  → ".join(sorted(common_8_32 - common_all)) or "  ❌ Yok")

print("\n🔥 16 & 32 ORTAK:")
print("  →", "\n  → ".join(sorted(common_16_32 - common_all)) or "  ❌ Yok")

print("\n🔥 SADECE 8:")
print("  →", "\n  → ".join(sorted(unique_8)) or "  ❌ Yok")

print("\n🔥 SADECE 16:")
print("  →", "\n  → ".join(sorted(unique_16)) or "  ❌ Yok")

print("\n🔥 SADECE 32:")
print("  →", "\n  → ".join(sorted(unique_32)) or "  ❌ Yok")

print("\n==========================================================\n")



# ============================================================
# 5️⃣ ROOT CAUSE ANALYSIS (TS eşleşen + min. 2 model)
# ============================================================

print("\n==================== ROOT CAUSE ANALYSIS ====================\n")

intersection_2plus = (common_8_16 | common_8_32 | common_16_32)

if not intersection_2plus:
    print("⚠️ Hiç ortak anomaly yok, root cause çalıştırılamadı.")

else:
    for ts in sorted(intersection_2plus):

        print(f"\n🔥 Ortak Anomali → {ts}")

        # Hangi modelde varsa o seçilir (öncelik: 16 → 8 → 32)
        if ts in sec16:
            model_ts = ts16
            model_idx = idx16
            Xs = Xs16
            Xp = Xp16
            src = "batch=16"

        elif ts in sec8:
            model_ts = ts8
            model_idx = idx8
            Xs = Xs8
            Xp = Xp8
            src = "batch=8"

        else:
            model_ts = ts32
            model_idx = idx32
            Xs = Xs32
            Xp = Xp32
            src = "batch=32"

        # timestamp → index eşleştirmesi
        real_idx = [
            i for i, tt in zip(model_idx, model_ts)
            if tt.strftime("%Y-%m-%d %H:%M:%S") == ts
        ][0]

        df, root = analyze_root_cause(Xs, Xp, real_idx, feats)

        print(df.to_string(index=False))
        print(f"\n→ ROOT CAUSE ({src}): {root['Feature']} (Z={root['Z-Score']:.2f})")
        print("------------------------------------------------------------")
