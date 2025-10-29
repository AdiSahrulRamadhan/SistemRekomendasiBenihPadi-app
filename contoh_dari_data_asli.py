#!/usr/bin/env python3
"""
Contoh Input dari DATA ASLI USER untuk SEMUA 9 Varietas
Berdasarkan data CSV yang diberikan user sebelumnya
"""

print("🌾 CONTOH INPUT DARI DATA ASLI ANDA")
print("=" * 70)
print("Diambil langsung dari dataset CSV yang Anda berikan!")

# Data asli dari user yang ada di analyze_data.py
data_asli_user = [
    {
        "varietas": "Inpari 32",
        "sumber": "Sulaiman, Mrandung, Klampis (Data Asli)",
        "input": {
            "Umur Tanaman (hari)": 120,
            "Kerebahan": "Sedang",
            "Tekstur Nasi": "Agak Pulen",
            "Potensi Hasil": 4.0,
            "Ketahanan Terhadap Hama": "Agak Rentan",
            "Kerontokan": "Agak Tahan",
            "Warna Gabah": "Kuning Bersih",
            "pH Tanah": 5.6
        }
    },
    {
        "varietas": "Ciherang",
        "sumber": "H. Fauzan, Mrandung, Klampis (Data Asli)",
        "input": {
            "Umur Tanaman (hari)": 116,
            "Kerebahan": "Sedang",
            "Tekstur Nasi": "Pulen",
            "Potensi Hasil": 5.0,
            "Ketahanan Terhadap Hama": "Sedang",
            "Kerontokan": "Sedang",
            "Warna Gabah": "Kuning Bersih",
            "pH Tanah": 5.6
        }
    },
    {
        "varietas": "Ciherang",
        "sumber": "Hj. Nasipah, Mrandung, Klampis (Data Asli - Variasi)",
        "input": {
            "Umur Tanaman (hari)": 105,
            "Kerebahan": "Sedang", 
            "Tekstur Nasi": "Pulen",
            "Potensi Hasil": 5.0,
            "Ketahanan Terhadap Hama": "Sedang",
            "Kerontokan": "Sedang",
            "Warna Gabah": "Kuning Bersih",
            "pH Tanah": 5.6
        }
    },
    {
        "varietas": "Situ Bagendit",
        "sumber": "Amrini, Mrandung, Klampis (Data Asli)",
        "input": {
            "Umur Tanaman (hari)": 117,
            "Kerebahan": "Sedang",
            "Tekstur Nasi": "Pera",
            "Potensi Hasil": 3.6,
            "Ketahanan Terhadap Hama": "Agak Tahan",
            "Kerontokan": "Sedang",
            "Warna Gabah": "Kuning Bersih",
            "pH Tanah": 5.6
        }
    },
    {
        "varietas": "Situ Bagendit", 
        "sumber": "Sunarah, Mrandung, Klampis (Data Asli - Variasi)",
        "input": {
            "Umur Tanaman (hari)": 112,
            "Kerebahan": "Sedang",
            "Tekstur Nasi": "Pera", 
            "Potensi Hasil": 3.55,
            "Ketahanan Terhadap Hama": "Agak Tahan",
            "Kerontokan": "Sedang",
            "Warna Gabah": "Kuning Bersih",
            "pH Tanah": 5.6
        }
    }
]

# Varietas yang belum ada di data asli - saya buat berdasarkan karakteristik umum
varietas_tambahan = [
    {
        "varietas": "IR-64",
        "sumber": "Berdasarkan karakteristik umum IR-64",
        "input": {
            "Umur Tanaman (hari)": 115,
            "Kerebahan": "Tahan",
            "Tekstur Nasi": "Pulen",
            "Potensi Hasil": 6.0,
            "Ketahanan Terhadap Hama": "Agak Tahan", 
            "Kerontokan": "Tahan",
            "Warna Gabah": "Kuning Bersih",
            "pH Tanah": 6.0
        }
    },
    {
        "varietas": "Inpari 30",
        "sumber": "Berdasarkan karakteristik umum Inpari 30",
        "input": {
            "Umur Tanaman (hari)": 110,
            "Kerebahan": "Sedang",
            "Tekstur Nasi": "Pulen",
            "Potensi Hasil": 3.0,
            "Ketahanan Terhadap Hama": "Rentan",
            "Kerontokan": "Sedang",
            "Warna Gabah": "Kuning Bersih",
            "pH Tanah": 5.5
        }
    },
    {
        "varietas": "Inpari 42",
        "sumber": "Berdasarkan karakteristik umum Inpari 42",
        "input": {
            "Umur Tanaman (hari)": 125,
            "Kerebahan": "Sedang",
            "Tekstur Nasi": "Pulen",
            "Potensi Hasil": 7.0,
            "Ketahanan Terhadap Hama": "Agak Rentan",
            "Kerontokan": "Tahan",
            "Warna Gabah": "Kuning Jerami",
            "pH Tanah": 6.0
        }
    },
    {
        "varietas": "Inpari 46",
        "sumber": "Berdasarkan karakteristik umum Inpari 46",
        "input": {
            "Umur Tanaman (hari)": 110,
            "Kerebahan": "Sedang",
            "Tekstur Nasi": "Pulen",
            "Potensi Hasil": 6.5,
            "Ketahanan Terhadap Hama": "Agak Rentan",
            "Kerontokan": "Sedang",
            "Warna Gabah": "Kuning Jerami",
            "pH Tanah": 6.2
        }
    },
    {
        "varietas": "Mekongga",
        "sumber": "Berdasarkan karakteristik umum Mekongga",
        "input": {
            "Umur Tanaman (hari)": 130,
            "Kerebahan": "Sedang",
            "Tekstur Nasi": "Agak Pulen",
            "Potensi Hasil": 4.0,
            "Ketahanan Terhadap Hama": "Agak Tahan",
            "Kerontokan": "Sedang",
            "Warna Gabah": "Kuning Bersih",
            "pH Tanah": 6.5
        }
    },
    {
        "varietas": "Sembada B9",
        "sumber": "Berdasarkan karakteristik umum Sembada B9",
        "input": {
            "Umur Tanaman (hari)": 105,
            "Kerebahan": "Sedang",
            "Tekstur Nasi": "Agak Pera",
            "Potensi Hasil": 6.0,
            "Ketahanan Terhadap Hama": "Agak Tahan",
            "Kerontokan": "Sedang",
            "Warna Gabah": "Kuning",
            "pH Tanah": 6.0
        }
    }
]

# Gabungkan semua
semua_contoh = data_asli_user + varietas_tambahan

print("\n📋 COPY-PASTE NILAI INI KE APLIKASI STREAMLIT:")
print("=" * 70)

# Urutkan berdasarkan nama varietas
semua_contoh_sorted = sorted(semua_contoh, key=lambda x: x['varietas'])

for i, case in enumerate(semua_contoh_sorted, 1):
    print(f"\n{i}. VARIETAS: {case['varietas']}")
    print(f"   Sumber: {case['sumber']}")
    print("   " + "-"*65)
    for field, value in case['input'].items():
        print(f"   {field}: {value}")

print("\n" + "="*70)
print("🎯 PRIORITAS TESTING (Mulai dari data asli Anda):")
print("=" * 70)

print("\n✅ DARI DATA ASLI ANDA:")
for case in data_asli_user:
    print(f"• {case['varietas']} - {case['sumber']}")

print("\n📝 TAMBAHAN (berdasarkan karakteristik umum):")
for case in varietas_tambahan:
    print(f"• {case['varietas']} - {case['sumber']}")

print("\n💡 CARA TESTING:")
print("1. Buka Streamlit app: http://localhost:8501")
print("2. Pilih 'Testing/Prediksi' di sidebar")
print("3. MULAI dengan data asli Anda (5 contoh pertama)")
print("4. Lanjut dengan data tambahan")
print("5. Catat hasil setiap prediksi")

print("\n🌾 TARGET HASIL:")
print("- Minimal 5-7 varietas berbeda dari 11 test cases")
print("- Data asli Anda HARUS prediksi sesuai label aslinya!")
print("=" * 70)