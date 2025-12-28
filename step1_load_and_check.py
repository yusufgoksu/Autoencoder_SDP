import pandas as pd

# 1️⃣ Dosya yolu (şimdilik TEK CSV ile başlıyoruz)
file_path = "Data/kuka_log600_scnd_20hz(çanta kaldırıldı yerine koyuldu).csv"

# 2️⃣ CSV oku
df = pd.read_csv(file_path, sep=";", engine="python")

# 3️⃣ İlk kontrol
print("CSV OKUNDU")
print("Kolon sayısı:", len(df.columns))
print(df.columns.tolist())

print("\nİlk 5 satır:")
print(df.head())

check_cols = [
    "TORQUE_A1",
    "CURR_A1",
    "MOT_TEMP_A1",
    "VEL_AXIS_ACT_A1"
]

print("\nKOLON TIPLERI:")
for c in check_cols:
    print(c, "->", df[c].dtype)