#!/usr/bin/env python3
"""
Contoh Input Valid untuk SEMUA 9 Varietas Padi
Berdasarkan karakteristik masing-masing varietas dari dataset
"""

print("🌾 CONTOH INPUT UNTUK SEMUA 9 VARIETAS PADI")
print("=" * 70)

# Berdasarkan analisis dataset dan karakteristik varietas
all_varieties_input = [
    {
        "varietas": "IR-64",
        "karakteristik": "Varietas unggul, tahan rebah, umur sedang",
        "input": {
            "Umur Tanaman (hari)": 115,
            "Kerebahan": "Tahan",
            "Tekstur Nasi": "Pulen", 
            "Potensi Hasil": 6,
            "Ketahanan Terhadap Hama": "Agak Tahan",
            "Kerontokan": "Tahan",
            "Warna Gabah": "Kuning Bersih",
            "pH Tanah": 6.0
        }
    },
    {
        "varietas": "Ciherang",
        "karakteristik": "Varietas populer, produktivitas tinggi, tahan hama",
        "input": {
            "Umur Tanaman (hari)": 116,
            "Kerebahan": "Sedang",
            "Tekstur Nasi": "Pulen",
            "Potensi Hasil": 5,
            "Ketahanan Terhadap Hama": "Tahan", 
            "Kerontokan": "Sedang",
            "Warna Gabah": "Kuning Bersih",
            "pH Tanah": 5.6
        }
    },
    {
        "varietas": "Inpari 30",
        "karakteristik": "Umur pendek, hasil rendah, rentan hama",
        "input": {
            "Umur Tanaman (hari)": 111,
            "Kerebahan": "Sedang",
            "Tekstur Nasi": "Pulen",
            "Potensi Hasil": 2,
            "Ketahanan Terhadap Hama": "Rentan",
            "Kerontokan": "Sedang", 
            "Warna Gabah": "Kuning Bersih",
            "pH Tanah": 5.6
        }
    },
    {
        "varietas": "Inpari 32", 
        "karakteristik": "Umur panjang, tekstur agak pulen, hasil sedang",
        "input": {
            "Umur Tanaman (hari)": 120,
            "Kerebahan": "Sedang",
            "Tekstur Nasi": "Agak Pulen",
            "Potensi Hasil": 4,
            "Ketahanan Terhadap Hama": "Agak Rentan",
            "Kerontokan": "Agak Tahan",
            "Warna Gabah": "Kuning Bersih", 
            "pH Tanah": 6.5
        }
    },
    {
        "varietas": "Inpari 42",
        "karakteristik": "Umur panjang, hasil tinggi, agak rentan",
        "input": {
            "Umur Tanaman (hari)": 126,
            "Kerebahan": "Sedang",
            "Tekstur Nasi": "Pulen",
            "Potensi Hasil": 7,
            "Ketahanan Terhadap Hama": "Agak Rentan",
            "Kerontokan": "Tahan",
            "Warna Gabah": "Kuning Jerami",
            "pH Tanah": 5.9
        }
    },
    {
        "varietas": "Inpari 46",
        "karakteristik": "Umur pendek, hasil tinggi, agak rentan",
        "input": {
            "Umur Tanaman (hari)": 111,
            "Kerebahan": "Sedang", 
            "Tekstur Nasi": "Pulen",
            "Potensi Hasil": 6,
            "Ketahanan Terhadap Hama": "Agak Rentan",
            "Kerontokan": "Sedang",
            "Warna Gabah": "Kuning Jerami",
            "pH Tanah": 6.5
        }
    },
    {
        "varietas": "Mekongga",
        "karakteristik": "Umur panjang, tekstur agak pulen, hasil rendah",
        "input": {
            "Umur Tanaman (hari)": 126,
            "Kerebahan": "Sedang",
            "Tekstur Nasi": "Agak Pulen", 
            "Potensi Hasil": 3,
            "Ketahanan Terhadap Hama": "Agak Tahan",
            "Kerontokan": "Sedang",
            "Warna Gabah": "Kuning Bersih",
            "pH Tanah": 6.5
        }
    },
    {
        "varietas": "Sembada B9",
        "karakteristik": "Umur pendek, tekstur agak pera, hasil tinggi",
        "input": {
            "Umur Tanaman (hari)": 105,
            "Kerebahan": "Sedang",
            "Tekstur Nasi": "Agak Pera",
            "Potensi Hasil": 6,
            "Ketahanan Terhadap Hama": "Agak Tahan",
            "Kerontokan": "Sedang",
            "Warna Gabah": "Kuning",
            "pH Tanah": 6.5
        }
    },
    {
        "varietas": "Situ Bagendit",
        "karakteristik": "Tekstur pera, hasil rendah, agak tahan hama",
        "input": {
            "Umur Tanaman (hari)": 117,
            "Kerebahan": "Sedang",
            "Tekstur Nasi": "Pera",
            "Potensi Hasil": 3,
            "Ketahanan Terhadap Hama": "Agak Tahan", 
            "Kerontokan": "Sedang",
            "Warna Gabah": "Kuning Bersih",
            "pH Tanah": 5.6
        }
    }
]

print("\n📋 COPY-PASTE NILAI INI KE APLIKASI STREAMLIT:")
print("=" * 70)

for i, case in enumerate(all_varieties_input, 1):
    print(f"\n{i}. VARIETAS: {case['varietas']}")
    print(f"   Karakteristik: {case['karakteristik']}")
    print("   " + "-"*60)
    for field, value in case['input'].items():
        print(f"   {field}: {value}")

print("\n" + "="*70)
print("🎯 CARA TESTING SEMUA VARIETAS:")
print("1. Buka Streamlit app: http://localhost:8501")
print("2. Pilih 'Testing/Prediksi' di sidebar") 
print("3. Test satu per satu dengan input di atas")
print("4. Catat hasil setiap prediksi")
print("5. Harus mendapat 9 varietas berbeda!")

print("\n💡 TIPS SUKSES:")
print("- Pastikan Clear Cache sebelum test")
print("- Periksa Scaler signature: [116.06...] (BALANCED)")
print("- Jika ada yang sama, coba variasi kecil pada nilai")
print("- Perhatikan 'Diversity achieved!' message")

print("\n🌾 TARGET HASIL:")
print("Harus mendapat minimal 5-7 varietas berbeda dari 9 test!")
print("=" * 70)