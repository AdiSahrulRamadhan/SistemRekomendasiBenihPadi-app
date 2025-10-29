#!/usr/bin/env python3
"""
Contoh Input Valid untuk Testing 3 Varietas Berbeda
Berdasarkan hasil debugging sebelumnya yang menunjukkan diversity
"""

print("🌾 CONTOH INPUT VALID UNTUK 3 VARIETAS BERBEDA")
print("=" * 60)

# Berdasarkan hasil testing sebelumnya yang berhasil
test_cases = [
    {
        "nama": "Low Performance Profile",
        "target": "Inpari 32",
        "input": {
            "Umur Tanaman (hari)": 100,
            "Kerebahan": "Tidak Tahan",           # 4
            "Tekstur Nasi": "Pera",               # 5  
            "Potensi Hasil": 3,
            "Ketahanan Terhadap Hama": "Tidak Tahan",  # 4
            "Kerontokan": "Rentan",               # 4
            "Warna Gabah": "Kuning Jerami",       # 1
            "pH Tanah": 5.0
        }
    },
    {
        "nama": "High Performance Profile", 
        "target": "Ciherang",
        "input": {
            "Umur Tanaman (hari)": 140,
            "Kerebahan": "Tahan",                 # 9
            "Tekstur Nasi": "Pulen",              # 8
            "Potensi Hasil": 8,
            "Ketahanan Terhadap Hama": "Tahan",   # 9
            "Kerontokan": "Tahan",                # 9
            "Warna Gabah": "Kuning Bersih",       # 0
            "pH Tanah": 9.5
        }
    },
    {
        "nama": "Medium Performance Profile",
        "target": "Situ Bagendit", 
        "input": {
            "Umur Tanaman (hari)": 120,
            "Kerebahan": "Sedang",                # 7
            "Tekstur Nasi": "Agak Pulen",         # 7
            "Potensi Hasil": 5,
            "Ketahanan Terhadap Hama": "Sedang",  # 7
            "Kerontokan": "Sedang",               # 7
            "Warna Gabah": "Kuning Bersih",       # 0
            "pH Tanah": 6.5
        }
    }
]

print("\n📋 COPY-PASTE NILAI INI KE APLIKASI STREAMLIT:")
print("-" * 60)

for i, case in enumerate(test_cases, 1):
    print(f"\n{i}. {case['nama']} (Expected: {case['target']}):")
    print("   " + "="*50)
    for field, value in case['input'].items():
        print(f"   {field}: {value}")
    
print("\n" + "="*60)
print("🔍 CARA MENGGUNAKAN:")
print("1. Buka Streamlit app: http://localhost:8501")
print("2. Pilih 'Testing/Prediksi' di sidebar")
print("3. Isi form dengan nilai di atas (satu per satu)")
print("4. Klik 'Prediksi dengan Semua Model'")
print("5. Lihat hasilnya - harus dapat 3 varietas berbeda!")

print("\n💡 TIPS:")
print("- Test satu per satu untuk melihat perbedaan")
print("- Perhatikan 'Debugging Info' untuk memastikan input benar")
print("- Jika tetap sama semua, ada masalah cache/loading")

print("\n🎯 EXPECTED RESULTS:")
print("- Test 1: Inpari 32")  
print("- Test 2: Ciherang")
print("- Test 3: Situ Bagendit")
print("=" * 60)