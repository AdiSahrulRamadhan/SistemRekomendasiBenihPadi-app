#!/usr/bin/env python3
"""
Analyze the actual dataset structure to update the app
"""

import pandas as pd

# Sample data from user
data_sample = """No,NamaLengkap,AsalDesa,AsalKecamatan,Tahun,VarietasBenihPadi,UmurTanaman ,Kerebahan,TeksturNasi,PotensiHasil,KetahananTerhadapHama,Kerontokan,WarnaGabah,PHTanah
1,Sulaiman,Mrandung,Klampis,2017,inpari 32,120,sedang,agak pulen,4,agak rentan,agak tahan,kuning bersih,5.6
2,H. Fauzan,Mrandung,Klampis,2017,ciherang,116,sedang,pulen,5,sedang,sedang,kuning bersih,5.6
3,Hj. Nasipah,Mrandung,Klampis,2017,ciherang,105,sedang,pulen,5,sedang,sedang,kuning bersih,5.6
4,Amrini,Mrandung,Klampis,2017,situ bagendit,117,sedang,pera,3.6,agak tahan,sedang,kuning bersih,5.6
5,Sunarah,Mrandung,Klampis,2017,situ bagendit,112,sedang,pera,3.55,agak tahan,sedang,kuning bersih,5.6"""

# Save to CSV
with open("sample_data.csv", "w", encoding="utf-8") as f:
    f.write(data_sample)

# Read and analyze
df = pd.read_csv("sample_data.csv")

print("DATASET ANALYSIS")
print("=" * 50)
print(f"Columns: {list(df.columns)}")
print(f"Shape: {df.shape}")
print()

print("UNIQUE VALUES PER COLUMN:")
print("-" * 30)
for col in df.columns:
    if col not in ['No', 'NamaLengkap', 'AsalDesa', 'AsalKecamatan', 'Tahun']:
        unique_vals = df[col].unique()
        print(f"{col}: {unique_vals}")
print()

print("FEATURE COLUMNS FOR ML:")
feature_cols = [col for col in df.columns if col not in ['No', 'NamaLengkap', 'AsalDesa', 'AsalKecamatan', 'Tahun', 'VarietasBenihPadi']]
target_col = 'VarietasBenihPadi'
print(f"Features ({len(feature_cols)}): {feature_cols}")
print(f"Target: {target_col}")
print()

print("VARIETY MAPPING:")
varieties = df[target_col].unique()
variety_mapping = {variety: idx for idx, variety in enumerate(sorted(varieties))}
print(variety_mapping)