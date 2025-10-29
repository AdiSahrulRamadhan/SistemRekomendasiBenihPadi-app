import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Sistem Rekomendasi Benih Padi",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4682B4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2E8B57;
        margin: 0.5rem 0;
    }
    
    /* Custom Sidebar Styles */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: none;
        padding: 0.8rem 1rem;
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #2E8B57, #3CB371);
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #228B22, #32CD32);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🌾 Sistem Rekomendasi Benih Padi Berdasarkan Indikator Petani di Madura Menggunakan Perbandingan Metode Classifier</h1>', unsafe_allow_html=True)

# Sidebar dengan logo dan judul
with st.sidebar:
    # Logo dan Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; margin-bottom: 2rem;">
        <div style="font-size: 4rem; margin-bottom: 0.5rem;">🌾</div>
        <h2 style="color: #2E8B57; margin: 0; font-size: 1.4rem; font-weight: bold;">
            Sistem Rekomendasi<br>Benih Padi
        </h2>
        <div style="width: 80%; height: 2px; background: linear-gradient(90deg, #2E8B57, #90EE90); margin: 1rem auto;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center; color: #2E8B57; margin-bottom: 1rem;'>Pilih Menu</h3>", unsafe_allow_html=True)
    
    # Navigation buttons dengan style yang bagus
    if st.button("📤 Upload Data", use_container_width=True, key="nav_upload"):
        st.session_state.page = "📤 Upload Data"
    
    if st.button("ℹ️ Informasi Data", use_container_width=True, key="nav_info"):
        st.session_state.page = "ℹ️ Informasi Data"
    
    if st.button("🔧 Preprocessing & Encoding", use_container_width=True, key="nav_preprocess"):
        st.session_state.page = "🔧 Preprocessing & Encoding"
    
    if st.button("🤖 Training Model", use_container_width=True, key="nav_training"):
        st.session_state.page = "🤖 Training Model"
    
    if st.button("🔮 Testing/Prediksi", use_container_width=True, key="nav_testing"):
        st.session_state.page = "🔮 Testing/Prediksi"

# Get current page from session state or default
page = st.session_state.get('page', "📤 Upload Data")

# Global variables and functions
RANDOM_STATE = 42

# Mapping functions
def _norm(s):
    return str(s).strip().lower().replace(" ", "").replace("_", "")

# Manual mappings
map_kerebahan = {_norm(k): v for k, v in {
    "Tahan": 9, "Sedang": 7, "Tidak Tahan": 4
}.items()}

map_teksturnasi = {_norm(k): v for k, v in {
    "Pulen": 8, "Agak Pulen": 7, "Agak Pera": 6, "Pera": 5
}.items()}

map_ketahanan = {_norm(k): v for k, v in {
    "Tahan": 9, "Agak Tahan": 7, "Sedang": 7, "Agak Rentan": 6, "Rentan": 4, "Tidak Tahan": 4
}.items()}

map_kerontokan = {_norm(k): v for k, v in {
    "Tahan": 9, "Toleran": 9, "Kuat": 9, "Sedang": 7, "Agak Tahan": 7, "Mudah": 4, "Rentan": 4
}.items()}

map_warnagabah = {_norm(k): v for k, v in {
    "Kuning Bersih": 0, "Kuning Jerami": 1, "Kuning": 2
}.items()}

MANUAL_MAP = {
    "Kerebahan": map_kerebahan,
    "TeksturNasi": map_teksturnasi,
    "KetahananTerhadapHama": map_ketahanan,
    "Kerontokan": map_kerontokan,
    "WarnaGabah": map_warnagabah,
}

# Varietas mapping
VARIETAS_NAME2IDX = {
    'IR-64': 0, 'ciherang': 1, 'inpari 30': 2, 'inpari 32': 3, 'inpari 42': 4,
    'inpari 46': 5, 'mekongga': 6, 'sembada b9': 7, 'situ bagendit': 8
}

def _fix_name(n: str) -> str:
    return n.upper() if n.strip().lower() == 'ir-64' else n.title()

IDX2VARIETAS = {idx: _fix_name(name) for name, idx in VARIETAS_NAME2IDX.items()}

def idx2name(idx: int) -> str:
    return IDX2VARIETAS.get(int(idx), str(int(idx)))

# Helper functions
def find_col(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def safe_cv_splits(y, target=5, min_allowed=2):
    _, cnt = np.unique(y, return_counts=True)
    return max(min(cnt.min(), target), min_allowed)

def drop_exact_duplicates(X, y):
    df = pd.DataFrame(X).copy()
    df['__y__'] = y
    n_before = len(df)
    df = df.drop_duplicates()
    y_new = df['__y__'].to_numpy()
    X_new = df.drop(columns=['__y__']).to_numpy()
    return X_new, y_new, n_before - len(df)

# Page 1: Upload Data
if page == "📤 Upload Data":
    st.markdown('<h2 class="sub-header">📤 Upload Data</h2>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload file dataset (.csv atau .xlsx)",
        type=['csv', 'xlsx'],
        help="Upload file dataset yang berisi data varietas padi"
    )

    # User guidance / help
    with st.expander("Panduan Upload (Klik untuk melihat)", expanded=False):
        st.markdown(
            """
            **Petunjuk singkat untuk mengunggah data:**

            - Format file yang didukung: **CSV** atau **Excel (.xlsx)**.
            - Separator CSV: aplikasi dapat mendeteksi otomatis; jika angka menggunakan koma sebagai desimal, pastikan format CSV sesuai (contoh: `1,5`).
            - Kolom yang direkomendasikan (dipakai oleh model):
              - `UmurTanaman` (numerik, hari)
              - `Kerebahan`, `TeksturNasi`, `KetahananTerhadapHama`, `Kerontokan`, `WarnaGabah` (kategorikal)
              - `PHTanah` (numerik)
              - `PotensiHasil` (numerik, skala 1-10)
              - `VarietasBenihPadi` atau kolom yang mengandung kata `varietas` sebagai label target
            - Contoh baris CSV:

              `116,Tahan,Pulen,Tahan,Tahan,Kuning Bersih,5.6,5,IR-64`

            - Tips:
              - Pastikan nama kolom ejaan konsisten. Aplikasi mencoba mendeteksi kolom umum, tetapi sebaiknya gunakan nama yang direkomendasikan di atas.
              - Jika data memiliki nilai kosong, aplikasi akan mencoba menangani dengan imputasi sederhana; namun sebaiknya bersihkan data terlebih dahulu bila memungkinkan.
              - Jika file berukuran besar, tunggu sampai proses upload dan pratinjau selesai.
            """,
            unsafe_allow_html=True,
        )

    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, sep=None, engine="python", decimal=",")
            
            st.success(f"✅ File berhasil diupload: {uploaded_file.name}")
            
            # Save to session state
            st.session_state['df_original'] = df.copy()
            
            # Display basic info
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📋 Info Dataset")
                st.write(f"**Jumlah baris:** {df.shape[0]}")
                st.write(f"**Jumlah kolom:** {df.shape[1]}")
                st.write(f"**Ukuran dataset:** {df.size}")
            
            with col2:
                st.markdown("### 📊 Contoh Data")
                st.dataframe(df.head())
            
            st.info("✅ Data berhasil diupload! Lanjutkan ke tahap 'Informasi Data' untuk melihat analisis detail.")
                    
        except Exception as e:
            st.error(f"❌ Error membaca file: {str(e)}")

# Page 2: Informasi Data  
elif page == "ℹ️ Informasi Data":
    st.markdown('<h2 class="sub-header">ℹ️ Informasi Data</h2>', unsafe_allow_html=True)
    
    if 'df_original' not in st.session_state:
        st.warning("⚠️ Silakan upload data terlebih dahulu di halaman 'Upload Data'")
    else:
        df = st.session_state['df_original'].copy()
        
        # Dataset info
        st.markdown("### 📋 Informasi Detail Dataset")
        buffer = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            buffer.append({
                'Kolom': col,
                'Tipe Data': dtype,
                'Nilai Kosong': null_count,
                'Nilai Unik': unique_count
            })
        
        info_df = pd.DataFrame(buffer)
        st.dataframe(info_df, use_container_width=True)
        
        # Statistical description
        st.markdown("### 📈 Statistik Deskriptif")
        st.dataframe(df.describe(include='all'))

        # Quick diagnostics: duplicates, missing percentages, high-cardinality
        st.markdown("### 🔎 Pemeriksaan Cepat Data")
        st.info("Catatan: pemeriksaan di bawah hanya bersifat informasional dan TIDAK mengubah dataset Anda. Untuk tindakan pembersihan, gunakan halaman 'Preprocessing & Encoding'.")

    # Duplikat tidak ditampilkan di sini (halaman Informasi hanya bersifat informasional)

        missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
        missing_df = missing_pct[missing_pct > 0].to_frame(name="Missing (%)")
        if not missing_df.empty:
            st.markdown("#### 📉 Kolom dengan nilai kosong (persentase)")
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.markdown("✅ Tidak ada nilai kosong yang terdeteksi")

        high_cardinality = [c for c in df.columns if df[c].nunique(dropna=True) > 100]
        if high_cardinality:
            st.markdown("#### ⚠️ Kolom dengan banyak nilai unik (high-cardinality)")
            st.write(
                "Kolom berikut memiliki lebih dari 100 nilai unik — ini bisa menyulitkan encoding kategorikal. Pertimbangkan reduksi fitur atau pengelompokan kategori:"
            )
            st.write(high_cardinality)

        # Suggest possible target and check imbalance
        possible_targets = [c for c in df.columns if 'varietas' in c.lower()]
        if possible_targets:
            target_col = possible_targets[0]
            st.markdown(f"#### 🎯 Potensi Kolom Target: **{target_col}**")
            value_counts = df[target_col].value_counts(dropna=True, normalize=True)
            top_classes = value_counts.head(5).to_frame(name='Proportion').reset_index()
            st.dataframe(top_classes, use_container_width=True)
            most_common_prop = value_counts.max()
            if most_common_prop > 0.8:
                st.warning("Dataset tampak sangat tidak seimbang (satu kelas mendominasi >80%). Pertimbangkan sampling atau penyesuaian metriks saat training.")
        else:
            st.info("ℹ️ Tidak ditemukan kolom target otomatis. Jika ada kolom target, pastikan namanya mengandung kata 'varietas' atau gunakan nama 'VarietasBenihPadi'.")

        # Explanations expander
        with st.expander("Apa yang ditampilkan di halaman ini (penjelasan singkat)"):
            st.markdown(
                """
                - **Tipe Data**: Menunjukkan tipe pandas untuk setiap kolom (numerik, objek, dsb.). Berguna untuk mengetahui kolom mana yang perlu encoding.
                - **Nilai Kosong**: Jumlah nilai kosong per kolom. Jika banyak, pertimbangkan imputasi atau penghapusan kolom/row.
                - **Nilai Unik**: Banyaknya nilai unik per kolom. Kolom kategorikal dengan nilai unik sangat banyak perlu penanganan khusus (high-cardinality).
                - **Statistik Deskriptif**: Rangkuman (mean, std, min, max) untuk kolom numerik dan ringkasan untuk kategorikal.
                - **Pemeriksaan Cepat**: Ringkasan duplikat, kolom dengan missing besar, high-cardinality, dan saran kolom target yang terdeteksi otomatis.
                """,
                unsafe_allow_html=True,
            )
        
        # Data visualization
        st.markdown("### 📊 Visualisasi Data")
        
        # Find relevant columns
        tahun_col = find_col(df, "tahun", "Tahun")
        varietas_col = find_col(df, "varietasbenihpadi", "VarietasBenihPadi")
        potensi_col = find_col(df, "potensihasil", "PotensiHasil")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Boxplot tahun vs potensi hasil
            if tahun_col and potensi_col:
                st.markdown("#### Sebaran Potensi Hasil per Tahun")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=df, x=tahun_col, y=potensi_col, ax=ax)
                ax.set_title("Sebaran Potensi Hasil Padi per Tahun")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("⚠️ Kolom 'tahun' atau 'potensihasil' tidak ditemukan")
        
        with viz_col2:
            # Bar chart varietas vs potensi hasil
            if varietas_col and potensi_col:
                st.markdown("#### Rata-rata Potensi Hasil per Varietas")
                fig, ax = plt.subplots(figsize=(12, 6))
                avg_data = df.groupby(varietas_col)[potensi_col].mean().sort_values(ascending=False)
                avg_data.plot(kind="bar", ax=ax)
                ax.set_title("Rata-Rata Potensi Hasil Berdasarkan Varietas")
                ax.set_ylabel(potensi_col)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("⚠️ Kolom varietas atau potensi hasil tidak ditemukan")
        
        # Correlation heatmap
        num_df = df.select_dtypes(include="number")
        if num_df.shape[1] >= 2:
            st.markdown("#### Korelasi Antar Variabel Numerik")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Korelasi Antar Variabel Numerik")
            plt.tight_layout()
            st.pyplot(fig)
        
        st.info("✅ Analisis data selesai! Lanjutkan ke tahap 'Preprocessing & Encoding' untuk mempersiapkan data.")

# Page 3: Preprocessing & Encoding
elif page == "🔧 Preprocessing & Encoding":
    st.markdown('<h2 class="sub-header">🔧 Preprocessing Data</h2>', unsafe_allow_html=True)
    
    if 'df_original' not in st.session_state:
        st.warning("⚠️ Silakan upload data terlebih dahulu di halaman 'Upload Data'")
    else:
        df = st.session_state['df_original'].copy()
        
        st.markdown("### 📋 Tahapan Preprocessing")
        
        # Progress indicator
        progress_container = st.container()
        
        # Step 1: Data Cleaning
        st.markdown("#### 🧹 Tahap 1: Pembersihan Data")
        st.markdown(
            """
            **Penjelasan:**
            - Pada tahap ini, kolom yang tidak bersifat prediktif (mis. ID, nama, alamat, tahun) akan dihapus otomatis.
            - Tujuan: mengurangi noise dan kolom yang tidak relevan agar model fokus pada fitur yang bermakna.
            - Hasil: dataset baru tanpa kolom yang di-drop akan disimpan sementara di sesi (tidak menimpa file asli).
            """
        )
        
        if st.button("🧹 Mulai Pembersihan Data"):
            with st.spinner("Membersihkan data..."):
                # Show before
                st.markdown("##### 📊 Data Sebelum Pembersihan:")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Jumlah baris:** {df.shape[0]}")
                    st.write(f"**Jumlah kolom:** {df.shape[1]}")
                    st.dataframe(df.head(3))
                
                with col2:
                    st.write("**Kolom yang akan dihapus:**")
                    drop_cols = ["No", "NamaLengkap", "AsalDesa", "AsalKecamatan", "Tahun"]
                    drop_cols_exist = [c for c in drop_cols if c in df.columns]
                    st.write(drop_cols_exist)
                
                # Drop non-predictive columns
                df_clean = df.drop(columns=drop_cols_exist)
                
                # Show after
                st.markdown("##### ✅ Data Setelah Pembersihan:")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Jumlah baris:** {df_clean.shape[0]}")
                    st.write(f"**Jumlah kolom:** {df_clean.shape[1]}")
                    st.dataframe(df_clean.head(3))
                
                with col2:
                    st.write("**Kolom yang berhasil dihapus:**")
                    st.write(drop_cols_exist)
                    st.success(f"Berhasil menghapus {len(drop_cols_exist)} kolom")
                
                # Save to session state
                st.session_state['df_step1'] = df_clean
                st.success("✅ Tahap 1 selesai!")
        
        # Step 2: Handle Missing Values
        if 'df_step1' in st.session_state:
            st.markdown("#### 🔍 Tahap 2: Penanganan Nilai Hilang")
            st.markdown(
                """
                **Penjelasan:**
                - Numerik: nilai kosong akan diisi menggunakan *median* kolom (lebih tahan terhadap outlier).
                - Kategorikal: nilai kosong diisi dengan *mode* (nilai yang paling sering muncul) jika tersedia.
                - Catatan: ini adalah strategi sederhana. Untuk dataset dengan missing banyak, pertimbangkan metode imputasi lebih canggih atau pemeriksaan sumber data.
                """
            )
            
            if st.button("🔄 Tangani Nilai Hilang"):
                with st.spinner("Menangani nilai hilang..."):
                    df_step1 = st.session_state['df_step1'].copy()
                    
                    # Show before
                    st.markdown("##### 📊 Sebelum Penanganan Nilai Hilang:")
                    missing_before = df_step1.isnull().sum()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Missing values per kolom:**")
                        if missing_before.sum() > 0:
                            st.dataframe(missing_before[missing_before > 0])
                        else:
                            st.write("Tidak ada nilai hilang")
                    
                    with col2:
                        st.write(f"**Total missing values:** {missing_before.sum()}")
                    
                    # Handle missing values
                    df_step2 = df_step1.copy()
                    
                    # Fill missing values for numeric columns
                    for col in df_step2.select_dtypes(include="number").columns:
                        if df_step2[col].isnull().sum() > 0:
                            df_step2[col].fillna(df_step2[col].median(), inplace=True)
                    
                    # Fill missing values for categorical columns
                    for col in df_step2.select_dtypes(exclude="number").columns:
                        if df_step2[col].isnull().sum() > 0 and len(df_step2[col].mode()) > 0:
                            df_step2[col].fillna(df_step2.col.mode()[0], inplace=True)
                    
                    # Show after
                    st.markdown("##### ✅ Setelah Penanganan Nilai Hilang:")
                    missing_after = df_step2.isnull().sum()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Missing values per kolom:**")
                        if missing_after.sum() > 0:
                            st.dataframe(missing_after[missing_after > 0])
                        else:
                            st.write("Tidak ada nilai hilang")
                    
                    with col2:
                        st.write(f"**Total missing values:** {missing_after.sum()}")
                        if missing_before.sum() > 0:
                            st.success(f"Berhasil menangani {missing_before.sum()} nilai hilang")
                    
                    # Save to session state
                    st.session_state['df_step2'] = df_step2
                    st.success("✅ Tahap 2 selesai!")
        
        # Step 3: Feature Encoding
        if 'df_step2' in st.session_state:
            st.markdown("#### 🔢 Tahap 3: Encoding Fitur")
            st.markdown(
                """
                **Penjelasan:**
                - Mengubah fitur kategorikal menjadi representasi numerik yang bisa diproses model.
                - Beberapa kolom di-encode menggunakan peta manual (mis. `Kerebahan`, `TeksturNasi`) untuk menjaga interpretabilitas.
                - Target (varietas) di-*label encode* sehingga setiap kelas mendapat angka integer.
                - Jika ada nilai yang tidak cocok dengan peta manual, aplikasi akan menandainya agar Anda review.
                """
            )
            
            if st.button("🔄 Mulai Encoding"):
                with st.spinner("Melakukan encoding data..."):
                    try:
                        df_step2 = st.session_state['df_step2'].copy()
                        
                        # Detect target column
                        possible_targets = [c for c in df_step2.columns if "varietas" in c.lower()]
                        if not possible_targets:
                            st.error("❌ Kolom target yang mengandung 'varietas' tidak ditemukan!")
                            st.stop()
                        
                        TARGET_COL = possible_targets[0]
                        y_raw = df_step2[TARGET_COL]
                        X = df_step2.drop(columns=[TARGET_COL])
                        
                        # Show before encoding
                        st.markdown("##### 📊 Sebelum Encoding:")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Target kolom:** {TARGET_COL}")
                            st.write("**Unique target values:**")
                            st.write(y_raw.value_counts().head())
                        
                        with col2:
                            st.write("**Feature columns (sample):**")
                            st.dataframe(X.head(3))
                        
                        # Manual encoding for categorical columns
                        X.columns = [c.strip() for c in X.columns]
                        col_map = {c.lower().replace(" ", "").replace("_", ""): c for c in X.columns}
                        
                        manual_plans = [
                            ("Kerebahan", map_kerebahan),
                            ("TeksturNasi", map_teksturnasi),
                            ("KetahananTerhadapHama", map_ketahanan),
                            ("Kerontokan", map_kerontokan),
                            ("WarnaGabah", map_warnagabah),
                        ]
                        
                        encoding_results = []
                        unmapped_report = {}
                        
                        for col_readable, mapper in manual_plans:
                            key = col_readable.lower().replace(" ", "").replace("_", "")
                            if key not in col_map:
                                continue
                            col = col_map[key]
                            
                            if pd.api.types.is_numeric_dtype(X[col]):
                                encoding_results.append(f"✅ {col}: Sudah numerik")
                                continue
                            
                            # Show before for this column
                            unique_before = X[col].unique()[:5]  # First 5 unique values
                            
                            raw_norm = X[col].astype(str).map(_norm)
                            mapped = raw_norm.map(mapper)
                            
                            gagal = sorted(set(raw_norm[pd.isna(mapped)]))
                            if gagal:
                                unmapped_report[col] = gagal
                                X.loc[pd.isna(mapped), col] = X.loc[pd.isna(mapped), col]
                                X.loc[~pd.isna(mapped), col] = mapped[~pd.isna(mapped)].astype(float)
                                encoding_results.append(f"🟨 {col}: Sebagian berhasil di-encode")
                            else:
                                X[col] = mapped.astype(float)
                                encoding_results.append(f"✅ {col}: Sukses di-encode penuh")
                        
                        # Encode target
                        label_y = LabelEncoder()
                        y = label_y.fit_transform(y_raw.astype(str))
                        
                        # Show after encoding
                        st.markdown("##### ✅ Setelah Encoding:")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Hasil Encoding:**")
                            for result in encoding_results:
                                st.write(result)
                            
                            st.write("**Target Label Mapping:**")
                            label_mapping = {cls: i for i, cls in enumerate(label_y.classes_)}
                            st.json(label_mapping)
                        
                        with col2:
                            st.write("**Feature setelah encoding (sample):**")
                            st.dataframe(X.head(3))
                            
                            if unmapped_report:
                                st.warning("⚠️ Nilai yang tidak berhasil di-encode:")
                                for col, vals in unmapped_report.items():
                                    st.write(f"- {col}: {vals}")
                        
                        # Save to session state
                        st.session_state['X_encoded'] = X
                        st.session_state['y_encoded'] = y
                        st.session_state['y_raw'] = y_raw
                        st.session_state['label_y'] = label_y
                        st.session_state['TARGET_COL'] = TARGET_COL
                        st.success("✅ Tahap 3 selesai!")
                        
                    except Exception as e:
                        st.error(f"❌ Error saat encoding: {str(e)}")
        
        # Step 4: Feature Scaling
        if 'X_encoded' in st.session_state and 'y_encoded' in st.session_state:
            st.markdown("#### 📏 Tahap 4: Scaling Fitur")
            st.markdown(
                """
                **Penjelasan:**
                - Scaling menstandarisasi fitur numerik sehingga memiliki mean=0 dan variance=1 (menggunakan StandardScaler).
                - Berguna terutama untuk model yang sensitif terhadap skala fitur (mis. SVM).
                - Hasil scaler akan disimpan di `preprocess_artifacts.pkl` untuk digunakan saat prediksi.
                """
            )
            
            if st.button("📏 Mulai Scaling"):
                with st.spinner("Melakukan scaling fitur..."):
                    X_encoded = st.session_state['X_encoded'].copy()
                    
                    # Show before scaling
                    st.markdown("##### 📊 Sebelum Scaling:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Statistik fitur sebelum scaling:**")
                        st.dataframe(X_encoded.describe())
                    
                    with col2:
                        st.write("**Sample data:**")
                        st.dataframe(X_encoded.head(3))
                    
                    # Scaling
                    scaler = StandardScaler()
                    X_scaled = pd.DataFrame(scaler.fit_transform(X_encoded), columns=X_encoded.columns, index=X_encoded.index)
                    
                    # Show after scaling
                    st.markdown("##### ✅ Setelah Scaling:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Statistik fitur setelah scaling:**")
                        st.dataframe(X_scaled.describe())
                    
                    with col2:
                        st.write("**Sample data:**")
                        st.dataframe(X_scaled.head(3))
                    
                    # Save preprocessing artifacts
                    preprocess_bundle = {
                        "scaler": scaler,
                        "target_encoder": st.session_state['label_y'],
                        "target_col": st.session_state['TARGET_COL'],
                        "feature_cols": list(X_encoded.columns),
                    }
                    joblib.dump(preprocess_bundle, "preprocess_artifacts.pkl")
                    
                    # Save to session state
                    st.session_state['X_scaled'] = X_scaled
                    st.session_state['preprocessing_complete'] = True
                    
                    st.success("✅ Tahap 4 selesai!")
                    st.success("🎉 Semua tahap preprocessing selesai! Data siap untuk training.")

# Page 4: Model Training
elif page == "🤖 Training Model":
    st.markdown('<h2 class="sub-header">🤖 Training Model Machine Learning</h2>', unsafe_allow_html=True)
    
    # Short explanation for users
    st.markdown(
        """
        **Penjelasan singkat halaman Training:**

        - Halaman ini melatih model-machine learning menggunakan data yang sudah diproses (scaling & encoding).
        - Anda dapat memilih beberapa model sekaligus (Decision Tree, Gaussian NB, SVM RBF).
        - Cross-validation dilakukan berulang (repeated holdout) berdasarkan pengaturan 'Jumlah Repetisi' dan 'Ukuran Test Set' untuk memperkirakan performa yang stabil.
        - Untuk SVM, tersedia opsi untuk memakai Grid Search atau parameter rekomendasi (lebih cepat).
        - Metrik yang dihitung: Accuracy, Precision, Recall, F1 (macro). Confusion matrix akan ditampilkan untuk analisis lebih lanjut.
        - Selama training, data duplikat pada fitur+target dihapus otomatis (readable info ditampilkan jika ada).
        - Hasil training (model final, best params, preprocessing artifacts) disimpan sebagai file `.joblib` dan metrik disimpan di sesi untuk diperiksa di halaman Testing/Prediksi.
        - Perhatian: training dengan banyak repetisi dan grid search dapat memakan waktu. Untuk percobaan cepat, turunkan nilai repetisi atau non-aktifkan grid search.
        """,
        unsafe_allow_html=True,
    )

    if 'preprocessing_complete' not in st.session_state or not st.session_state.get('preprocessing_complete', False):
        st.warning("⚠️ Silakan selesaikan preprocessing data terlebih dahulu di halaman 'Preprocessing & Encoding'")
    else:
        X_scaled = st.session_state['X_scaled']
        y = st.session_state['y_encoded']
        
        st.markdown("### 🚀 Konfigurasi Training Model")
        
        # Model selection
        selected_models = st.multiselect(
            "Pilih model yang ingin dilatih:",
            ["Decision Tree", "Gaussian Naive Bayes", "SVM (RBF)"],
            default=["Decision Tree", "Gaussian Naive Bayes", "SVM (RBF)"]
        )
        
        # Training parameters
        st.markdown("#### ⚙️ Parameter Cross-Validation")
        col1, col2 = st.columns(2)
        with col1:
            n_repeats = st.slider("Jumlah Repetisi Cross-Validation:", 5, 50, 20)
            test_size = st.slider("Ukuran Test Set:", 0.1, 0.4, 0.2)
        
        with col2:
            st.info(f"""
            **Info Dataset:**
            - Dataset: {X_scaled.shape[0]} baris, {X_scaled.shape[1]} fitur
            - Target classes: {len(np.unique(y))}
            - Cross-validation: {n_repeats} repetisi
            """)
        
        # SVM parameters (only show if SVM is selected)
        if "SVM (RBF)" in selected_models:
            st.markdown("#### 🎯 SVM (RBF) Parameters")
            svm_col1, svm_col2, svm_col3 = st.columns(3)
            
            with svm_col1:
                svm_c_mode = st.radio(
                    "Mode Parameter C:",
                    ["Auto (Grid Search)", "Rekomendasi"],
                    help="Auto akan mencari C terbaik, Rekomendasi menggunakan nilai yang disarankan"
                )
                
                if svm_c_mode == "Rekomendasi":
                    svm_c_value = st.number_input(
                        "Nilai C:", 
                        min_value=0.001, 
                        max_value=1000.0, 
                        value=10.0,
                        step=0.1,
                        help="Parameter regularisasi. Nilai lebih besar = less regularization"
                    )
                else:
                    svm_c_options = st.multiselect(
                        "Pilihan C untuk Grid Search:",
                        [0.01, 0.1, 1, 10, 100, 1000],
                        default=[0.1, 1, 10, 100],
                        help="Nilai-nilai C yang akan dicoba dalam grid search"
                    )
            
            with svm_col2:
                svm_gamma_mode = st.radio(
                    "Mode Parameter Gamma:",
                    ["Auto (Grid Search)", "Rekomendasi"],
                    help="Auto akan mencari gamma terbaik, Rekomendasi menggunakan nilai yang disarankan"
                )
                
                if svm_gamma_mode == "Rekomendasi":
                    svm_gamma_type = st.selectbox(
                        "Tipe Gamma:",
                        ["scale", "auto", "custom"],
                        index=0,  # Default to "scale"
                        help="scale=1/(n_features * X.var()), auto=1/n_features, custom=nilai manual"
                    )
                    
                    if svm_gamma_type == "custom":
                        svm_gamma_value = st.number_input(
                            "Nilai Gamma:", 
                            min_value=0.0001, 
                            max_value=10.0, 
                            value=0.1,
                            step=0.01,
                            format="%.4f",
                            help="Koefisien kernel RBF. Nilai lebih besar = overfitting"
                        )
                else:
                    svm_gamma_options = st.multiselect(
                        "Pilihan Gamma untuk Grid Search:",
                        ["scale", "auto", 0.001, 0.01, 0.1, 1],
                        default=["scale", "auto", 0.01, 0.1],
                        help="Nilai-nilai gamma yang akan dicoba dalam grid search"
                    )
            
            with svm_col3:
                svm_probability = st.checkbox(
                    "Enable Probability", 
                    value=True,
                    help="Aktifkan untuk mendapatkan prediksi probabilitas (sedikit lebih lambat)"
                )
        
        if st.button("🚀 Mulai Training Model"):
            results = {}
            
            for model_name in selected_models:
                st.markdown(f"#### 🔄 Training {model_name}...")
                
                with st.spinner(f"Training {model_name}..."):
                    try:
                        # Remove duplicates
                        X_clean, y_clean, n_duplicates = drop_exact_duplicates(X_scaled.values, y)
                        if n_duplicates > 0:
                            st.info(f"Duplikat data dihapus: {n_duplicates}")
                        
                        # Repeated holdout validation
                        sss = StratifiedShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=RANDOM_STATE)
                        accs, precs, recs, f1s = [], [], [], []
                        best_params_list = []
                        labels = np.unique(y_clean)
                        cm_sum = np.zeros((len(labels), len(labels)), dtype=np.int64)
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, (tr, te) in enumerate(sss.split(X_clean, y_clean)):
                            status_text.text(f"Fold {i+1}/{n_repeats} - {model_name}")
                            
                            Xtr, Xte = X_clean[tr], X_clean[te]
                            ytr, yte = y_clean[tr], y_clean[te]
                            
                            # Model-specific training
                            if model_name == "Decision Tree":
                                pipe = Pipeline([("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))])
                                param_grid = {
                                    "clf__max_depth": [None, 3, 5, 7, 9, 12],
                                    "clf__min_samples_leaf": [1, 3, 5, 10],
                                    "clf__min_samples_split": [2, 5, 10],
                                }
                                cv = StratifiedKFold(n_splits=safe_cv_splits(ytr), shuffle=True, random_state=RANDOM_STATE)
                                grid = GridSearchCV(pipe, param_grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)
                                grid.fit(Xtr, ytr)
                                best_model = grid.best_estimator_
                                best_params = grid.best_params_
                                
                            elif model_name == "Gaussian Naive Bayes":
                                # Implementasi sesuai dengan kode Colab - menggunakan function yang sama
                                def make_pipeline_gnb(var_smoothing=None):
                                    clf = GaussianNB() if var_smoothing is None else GaussianNB(var_smoothing=var_smoothing)
                                    return Pipeline([
                                        ("scaler", StandardScaler()),
                                        ("pca", PCA(n_components=0.95, svd_solver="full")),
                                        ("clf", clf)
                                    ])
                                
                                def tune_on_train_gnb(Xtr, ytr):
                                    pipe = make_pipeline_gnb()
                                    param_grid = {"clf__var_smoothing": np.logspace(-12, -6, 7)}
                                    cv = StratifiedKFold(n_splits=safe_cv_splits(ytr), shuffle=True, random_state=RANDOM_STATE)
                                    grid = GridSearchCV(pipe, param_grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)
                                    grid.fit(Xtr, ytr)
                                    return grid.best_estimator_, grid.best_params_
                                
                                # Gunakan function yang sama dengan Colab
                                best_model, best_params = tune_on_train_gnb(Xtr, ytr)
                                
                            elif model_name == "SVM (RBF)":
                                # Determine C parameter
                                if svm_c_mode == "Rekomendasi":
                                    c_param = svm_c_value
                                    param_grid = {}
                                else:
                                    if not svm_c_options:
                                        svm_c_options = [0.1, 1, 10, 100]
                                    param_grid = {"clf__C": svm_c_options}
                                
                                # Determine gamma parameter
                                if svm_gamma_mode == "Rekomendasi":
                                    if svm_gamma_type == "custom":
                                        gamma_param = svm_gamma_value
                                    else:
                                        gamma_param = svm_gamma_type
                                    
                                    if "clf__C" not in param_grid:
                                        param_grid = {}
                                else:
                                    if not svm_gamma_options:
                                        svm_gamma_options = ["scale", "auto", 0.01, 0.1]
                                    if "clf__C" not in param_grid:
                                        param_grid = {"clf__gamma": svm_gamma_options}
                                    else:
                                        param_grid["clf__gamma"] = svm_gamma_options
                                
                                pipe = Pipeline([
                                    ("scaler", StandardScaler()),
                                    ("clf", SVC(
                                        kernel="rbf", 
                                        probability=svm_probability, 
                                        random_state=RANDOM_STATE
                                    ))
                                ])
                                
                                # Use grid search or direct fit based on parameters
                                if param_grid:
                                    cv = StratifiedKFold(n_splits=safe_cv_splits(ytr), shuffle=True, random_state=RANDOM_STATE)
                                    grid = GridSearchCV(pipe, param_grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)
                                    grid.fit(Xtr, ytr)
                                    best_model = grid.best_estimator_
                                    best_params = grid.best_params_
                                else:
                                    # Rekomendasi parameters - direct fit
                                    if svm_c_mode == "Rekomendasi":
                                        pipe.set_params(clf__C=c_param)
                                    if svm_gamma_mode == "Rekomendasi":
                                        pipe.set_params(clf__gamma=gamma_param)
                                    
                                    pipe.fit(Xtr, ytr)
                                    best_model = pipe
                                    best_params = {
                                        "clf__C": c_param if svm_c_mode == "Rekomendasi" else 1.0,
                                        "clf__gamma": gamma_param if svm_gamma_mode == "Rekomendasi" else "scale"
                                    }
                            
                            yhat = best_model.predict(Xte)
                            
                            accs.append(accuracy_score(yte, yhat))
                            precs.append(precision_score(yte, yhat, average="macro", zero_division=0))
                            recs.append(recall_score(yte, yhat, average="macro", zero_division=0))
                            f1s.append(f1_score(yte, yhat, average="macro", zero_division=0))
                            best_params_list.append(best_params)
                            
                            cm_sum += confusion_matrix(yte, yhat, labels=labels)
                            
                            progress_bar.progress((i + 1) / n_repeats)
                        
                        status_text.empty()
                        
                        # Calculate final metrics
                        results[model_name] = {
                            "accuracy": (np.mean(accs), np.std(accs)),
                            "precision": (np.mean(precs), np.std(precs)),
                            "recall": (np.mean(recs), np.std(recs)),
                            "f1": (np.mean(f1s), np.std(f1s)),
                            "confusion_matrix": cm_sum,
                            "labels": labels,
                            "best_params": best_params_list,
                            "raw_scores": {
                                "accuracy": accs,
                                "precision": precs,
                                "recall": recs,
                                "f1": f1s
                            }
                        }
                        
                        # Train final model on full data
                        if model_name == "Decision Tree":
                            hp_counts = Counter([str(p) for p in best_params_list])
                            best_params_str, _ = hp_counts.most_common(1)[0]
                            best_params_final = eval(best_params_str)
                            final_model = Pipeline([("clf", DecisionTreeClassifier(
                                random_state=RANDOM_STATE,
                                **{k.replace('clf__', ''): v for k, v in best_params_final.items()}
                            ))])
                            
                        elif model_name == "Gaussian Naive Bayes":
                            hp_counts = Counter([str(p) for p in best_params_list])
                            best_params_str, _ = hp_counts.most_common(1)[0]
                            best_params_final = eval(best_params_str)
                            final_model = Pipeline([
                                ("scaler", StandardScaler()),
                                ("pca", PCA(n_components=0.95, svd_solver="full")),
                                ("clf", GaussianNB(**{k.replace('clf__', ''): v for k, v in best_params_final.items()}))
                            ])
                            
                        elif model_name == "SVM (RBF)":
                            hp_counts = Counter([str(p) for p in best_params_list])
                            best_params_str, _ = hp_counts.most_common(1)[0]
                            best_params_final = eval(best_params_str)
                            final_model = Pipeline([
                                ("scaler", StandardScaler()),
                                ("clf", SVC(
                                    kernel="rbf", 
                                    probability=svm_probability, 
                                    random_state=RANDOM_STATE,
                                    C=best_params_final.get("clf__C", 1.0),
                                    gamma=best_params_final.get("clf__gamma", "scale")
                                ))
                            ])
                        
                        final_model.fit(X_clean, y_clean)
                        
                        # Save model
                        model_filename = f"{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_final.joblib"
                        joblib.dump(final_model, model_filename)
                        results[model_name]["model_file"] = model_filename
                        results[model_name]["final_model"] = final_model
                        results[model_name]["best_params_final"] = best_params_final
                        
                        st.success(f"✅ {model_name} training selesai!")
                        
                    except Exception as e:
                        st.error(f"❌ Error training {model_name}: {str(e)}")
            
            if results:
                # Save results to session state
                st.session_state['training_results'] = results
                st.session_state['training_complete'] = True
                
                # Display results
                st.markdown("### 📊 Ringkasan Hasil Training")
                
                # Summary table
                summary_data = []
                for model_name, metrics in results.items():
                    acc_mean, acc_std = metrics["accuracy"]
                    prec_mean, prec_std = metrics.get("precision", (0.0, 0.0))
                    rec_mean, rec_std = metrics.get("recall", (0.0, 0.0))
                    f1_mean, f1_std = metrics["f1"]
                    
                    summary_data.append({
                        "Model": model_name,
                        "Accuracy (%)": f"{acc_mean*100:.2f} ± {acc_std*100:.2f}",
                        "Precision": f"{prec_mean:.3f} ± {prec_std:.3f}",
                        "Recall": f"{rec_mean:.3f} ± {rec_std:.3f}",
                        "F1-Score": f"{f1_mean:.3f} ± {f1_std:.3f}",
                        "Status": "✅ Tersimpan"
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Confusion matrices for all models
                st.markdown("### 🎯 Confusion Matrix Semua Model")
                
                for model_name, metrics in results.items():
                    st.markdown(f"#### {model_name}")
                    
                    cm = metrics["confusion_matrix"]
                    labels = metrics["labels"]
                    label_names = [idx2name(int(label)) for label in labels]
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Count matrix
                    disp1 = ConfusionMatrixDisplay(cm, display_labels=label_names)
                    disp1.plot(ax=ax1, cmap="Blues", values_format="d")
                    ax1.set_title(f"Count Matrix - {model_name}")
                    
                    # Normalized matrix
                    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    disp2 = ConfusionMatrixDisplay(cm_norm, display_labels=label_names)
                    disp2.plot(ax=ax2, cmap="Blues", values_format=".2f")
                    ax2.set_title(f"Normalized Matrix - {model_name}")
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                st.success("🎉 Training semua model selesai! Lanjutkan ke halaman 'Testing/Prediksi'.")

# Page 5: Testing/Prediksi
elif page == "🔮 Testing/Prediksi":
    st.markdown('<h2 class="sub-header">🔮 Testing/Prediksi Varietas Padi</h2>', unsafe_allow_html=True)
    
    # Check available models
    model_files = {
        "Decision Tree": "decision_tree_final.joblib",
        "Gaussian Naive Bayes": "gaussian_naive_bayes_final.joblib", 
        "SVM (RBF)": "svm_rbf_final.joblib"
    }
    
    available_models = {}
    
    # Add debugging section
    st.markdown("### 🔍 Model Loading Debug Info")
    
    for name, filename in model_files.items():
        if os.path.exists(filename):
            try:
                # Force reload without caching
                model = joblib.load(filename)
                available_models[name] = model
                
                # Show model type info
                model_type = str(type(model)).split('.')[-1].replace("'>", "")
                file_size = os.path.getsize(filename) / 1024  # KB
                st.success(f"✅ {name}: {model_type} ({file_size:.1f} KB)")
                
            except Exception as e:
                st.error(f"❌ {name}: {str(e)}")
        else:
            st.error(f"❌ {name}: File {filename} tidak ditemukan")
    
    if not available_models:
        st.warning("⚠️ Tidak ada model yang tersedia. Silakan latih model terlebih dahulu di halaman 'Training Model'")
        
        # Show evaluation if available
        if 'training_results' in st.session_state:
            st.markdown("### 📊 Hasil Evaluasi Model")
            results = st.session_state['training_results']
            
            # Display confusion matrices for all trained models
            for model_name, metrics in results.items():
                st.markdown(f"#### 🎯 Confusion Matrix - {model_name}")
                
                cm = metrics["confusion_matrix"]
                labels = metrics["labels"]
                label_names = [idx2name(int(label)) for label in labels]
                
                # Display metrics
                acc_mean, acc_std = metrics["accuracy"]
                prec_mean, prec_std = metrics["precision"]
                rec_mean, rec_std = metrics["recall"]
                f1_mean, f1_std = metrics["f1"]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{acc_mean*100:.2f}%", delta=f"±{acc_std*100:.2f}%")
                with col2:
                    st.metric("Precision", f"{prec_mean:.3f}", delta=f"±{prec_std:.3f}")
                with col3:
                    st.metric("Recall", f"{rec_mean:.3f}", delta=f"±{rec_std:.3f}")
                with col4:
                    st.metric("F1-Score", f"{f1_mean:.3f}", delta=f"±{f1_std:.3f}")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Count matrix
                disp1 = ConfusionMatrixDisplay(cm, display_labels=label_names)
                disp1.plot(ax=ax1, cmap="Blues", values_format="d")
                ax1.set_title(f"Count Matrix - {model_name}")
                
                # Normalized matrix
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                disp2 = ConfusionMatrixDisplay(cm_norm, display_labels=label_names)
                disp2.plot(ax=ax2, cmap="Blues", values_format=".2f")
                ax2.set_title(f"Normalized Matrix - {model_name}")
                
                plt.tight_layout()
                st.pyplot(fig)
                st.write("---")
    
    else:
        st.success(f"✅ Model tersedia: {', '.join(available_models.keys())}")
        
        # Load preprocessing artifacts with explicit checking and forced reload
        try:
            filename = "preprocess_artifacts.pkl"
            if not os.path.exists(filename):
                st.error("❌ File preprocessing tidak ditemukan. Silakan lakukan training terlebih dahulu.")
                st.stop()
            
            # FORCE RELOAD - disable caching completely for this critical function
            import time
            mtime = os.path.getmtime(filename)
            age_minutes = (time.time() - mtime) / 60
            
            # Clear ALL streamlit caches before loading
            st.cache_data.clear()
            st.cache_resource.clear()
            
            # Use joblib.load with fresh read every time
            preprocess_bundle = joblib.load(filename)
            
            scaler = preprocess_bundle["scaler"]
            label_y = preprocess_bundle["target_encoder"]
            feature_cols = preprocess_bundle["feature_cols"]
            
            # Show scaler info safely
            st.success(f"✅ Preprocessing artifacts berhasil dimuat (umur file: {age_minutes:.1f} menit)")
                
        except Exception as e:
            st.error(f"❌ Error memuat preprocessing artifacts: {str(e)}")
            st.stop()
        
        # Show training results if available
        if 'training_results' in st.session_state:
            st.markdown("### 📊 Ringkasan Evaluasi Semua Model")
            results = st.session_state['training_results']
            
            # Summary table
            summary_data = []
            for model_name in available_models.keys():
                if model_name in results:
                    metrics = results[model_name]
                    acc_mean, acc_std = metrics["accuracy"]
                    prec_mean, prec_std = metrics.get("precision", (0.0, 0.0))
                    rec_mean, rec_std = metrics.get("recall", (0.0, 0.0))
                    f1_mean, f1_std = metrics["f1"]
                    
                    summary_data.append({
                        "Model": model_name,
                        "Accuracy (%)": f"{acc_mean*100:.2f} ± {acc_std*100:.2f}",
                        "Precision": f"{prec_mean:.3f} ± {prec_std:.3f}",
                        "Recall": f"{rec_mean:.3f} ± {rec_std:.3f}",
                        "F1-Score": f"{f1_mean:.3f} ± {f1_std:.3f}",
                        "Status": "✅ Siap Prediksi"
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Best model recommendation
                best_f1 = 0
                best_model_name = ""
                for model_name in available_models.keys():
                    if model_name in results:
                        f1_mean, _ = results[model_name]["f1"]
                        if f1_mean > best_f1:
                            best_f1 = f1_mean
                            best_model_name = model_name
                
                if best_model_name:
                    st.info(f"🏆 **Model dengan performa terbaik**: {best_model_name} (F1-Score: {best_f1:.3f})")
        
        # Clear cache button
        if st.button("🔄 Clear Cache & Refresh Models", help="Klik ini jika prediksi tidak berubah"):
            st.cache_data.clear()
            st.cache_resource.clear()
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            # Force reload all files by checking file timestamps
            import os
            import time
            current_time = time.time()
            
            files_to_check = [
                "preprocess_artifacts.pkl",
                "decision_tree_final.joblib", 
                "gaussian_naive_bayes_final.joblib",
                "svm_rbf_final.joblib"
            ]
            
            st.write("**File Status Check:**")
            for filename in files_to_check:
                if os.path.exists(filename):
                    mtime = os.path.getmtime(filename)
                    age_minutes = (current_time - mtime) / 60
                    st.write(f"- {filename}: Modified {age_minutes:.1f} minutes ago")
                else:
                    st.write(f"- {filename}: ❌ Not found")
            
            st.success("✅ Cache & session cleared! Files reloaded!")
            st.rerun()
        
        # Input form for prediction
        st.markdown("### 📝 Input Data untuk Prediksi")
        
        with st.form("prediction_form"):
            st.markdown("Masukkan karakteristik padi yang ingin diprediksi varietasnya:")
            
            # Original defaults
            default_kerebahan = "Tahan"
            default_tekstur = "Pulen"
            default_ketahanan = "Tahan"
            default_kerontokan = "Tahan"
            default_warna = "Kuning Bersih"
            default_umur = 116
            default_ph = 5.6
            default_potensi = 5
            
            col1, col2 = st.columns(2)
            
            with col1:
                kerebahan = st.selectbox(
                    "Kerebahan:", 
                    ["Tahan", "Sedang", "Tidak Tahan"],
                    index=["Tahan", "Sedang", "Tidak Tahan"].index(default_kerebahan),
                    help="Ketahanan tanaman terhadap rebah"
                )
                tekstur_nasi = st.selectbox(
                    "Tekstur Nasi:", 
                    ["Pulen", "Agak Pulen", "Agak Pera", "Pera"],
                    index=["Pulen", "Agak Pulen", "Agak Pera", "Pera"].index(default_tekstur),
                    help="Tekstur nasi yang dihasilkan"
                )
                ketahanan_hama = st.selectbox(
                    "Ketahanan Terhadap Hama:", 
                    ["Tahan", "Agak Tahan", "Sedang", "Agak Rentan", "Rentan", "Tidak Tahan"],
                    index=["Tahan", "Agak Tahan", "Sedang", "Agak Rentan", "Rentan", "Tidak Tahan"].index(default_ketahanan),
                    help="Ketahanan tanaman terhadap serangan hama"
                )
                kerontokan = st.selectbox(
                    "Kerontokan:", 
                    ["Tahan", "Toleran", "Kuat", "Sedang", "Agak Tahan", "Mudah", "Rentan"],
                    index=["Tahan", "Toleran", "Kuat", "Sedang", "Agak Tahan", "Mudah", "Rentan"].index(default_kerontokan),
                    help="Ketahanan gabah terhadap kerontokan"
                )
            
            with col2:
                warna_gabah = st.selectbox(
                    "Warna Gabah:", 
                    ["Kuning Bersih", "Kuning Jerami", "Kuning"],
                    index=["Kuning Bersih", "Kuning Jerami", "Kuning"].index(default_warna),
                    help="Warna gabah yang dihasilkan"
                )
                umur_tanaman = st.number_input(
                    "Umur Tanaman (hari):", 
                    min_value=80, 
                    max_value=150, 
                    value=default_umur,
                    help="Lama waktu tanam hingga panen"
                )
                ph_tanah = st.number_input(
                    "pH Tanah:", 
                    min_value=4.0, 
                    max_value=8.0, 
                    value=float(default_ph), 
                    step=0.1,
                    format="%.1f",
                    help="Tingkat keasaman tanah"
                )
                potensi_hasil = st.number_input(
                    "Potensi Hasil:", 
                    min_value=1.0, 
                    max_value=10.0, 
                    value=float(default_potensi),
                    step=0.1,
                    format="%.1f",
                    help="Potensi hasil panen (skala 1-10)"
                )
            
            submitted = st.form_submit_button("🔮 Prediksi Varietas Padi", use_container_width=True)
        
        if submitted:
                # Prepare input data
                input_data = {
                    "Kerebahan": kerebahan,
                    "TeksturNasi": tekstur_nasi,
                    "KetahananTerhadapHama": ketahanan_hama,
                    "Kerontokan": kerontokan,
                    "WarnaGabah": warna_gabah,
                    "UmurTanaman": umur_tanaman,
                    "PHTanah": ph_tanah,
                    "PotensiHasil": potensi_hasil,
                }
                
                # Preprocess input with CORRECTED order
                def preprocess_input(input_dict, feature_cols):
                    """Fixed preprocessing that orders inputs correctly for the scaler"""
                    processed = {}
                    
                    # Apply manual mappings
                    for col, mapper in MANUAL_MAP.items():
                        if col in input_dict:
                            val = input_dict[col]
                            mapped_val = mapper.get(_norm(str(val)), 0)
                            processed[col] = float(mapped_val)
                    
                    # Handle numeric columns
                    numeric_cols = ["UmurTanaman", "PHTanah", "PotensiHasil"]
                    for col in numeric_cols:
                        if col in input_dict:
                            processed[col] = float(input_dict[col])
                    
                    # Create input array in the CORRECT ORDER that scaler expects
                    # feature_cols = ['UmurTanaman', 'Kerebahan', 'TeksturNasi', 'PotensiHasil', 
                    #                 'KetahananTerhadapHama', 'Kerontokan', 'WarnaGabah', 'PHTanah']
                    input_row = []
                    for col in feature_cols:
                        value = processed.get(col, 0.0)
                        input_row.append(value)
                    
                    return np.array(input_row).reshape(1, -1)
                
                try:
                    # Load models
                    fresh_models = {}
                    model_files = ["decision_tree_final.joblib", "gaussian_naive_bayes_final.joblib", "svm_rbf_final.joblib"]
                    for model_file in model_files:
                        if os.path.exists(model_file):
                            fresh_models[model_file.replace('_final.joblib', '')] = joblib.load(model_file)
                    
                    # Load preprocessing
                    fresh_preprocess = joblib.load("preprocess_artifacts.pkl")
                    fresh_scaler = fresh_preprocess["scaler"]
                    
                    X_input = preprocess_input(input_data, feature_cols)
                    X_scaled = fresh_scaler.transform(X_input)
                    
                    # Get predictions from all models
                    predictions = {}
                    all_predictions = {}  # For top 3 predictions
                    
                    st.markdown("### 🎯 Rekomendasi Benih Padi")
                    
                    for model_name, model in fresh_models.items():
                        if model is None:
                            continue
                            
                        # Get prediction
                        prediction = model.predict(X_scaled)[0]
                        variety = idx2name(prediction)
                        predictions[model_name] = variety
                        
                        # Get top 3 predictions with probabilities
                        try:
                            proba = model.predict_proba(X_scaled)[0]
                            # Get indices of top 3 probabilities
                            top_indices = proba.argsort()[-3:][::-1]
                            top_predictions = [(idx2name(idx), proba[idx] * 100) for idx in top_indices]
                            all_predictions[model_name] = top_predictions
                        except:
                            # For models without predict_proba, just show the prediction
                            all_predictions[model_name] = [(variety, 100.0), ("", 0), ("", 0)]
                        
                        # Model-specific styling
                        if "decision_tree" in model_name.lower():
                            color = "#228B22"  # Forest Green
                            icon = "🌳"
                            display_name = "Decision Tree"
                        elif "naive_bayes" in model_name.lower():
                            color = "#4169E1"  # Royal Blue
                            icon = "🧠"
                            display_name = "Naive Bayes"
                        elif "svm" in model_name.lower():
                            color = "#DC143C"  # Crimson
                            icon = "⚡"
                            display_name = "SVM"
                        else:
                            color = "#666666"
                            icon = "🤖"
                            display_name = model_name.title()
                        
                        # Display result with top 3 predictions
                        st.markdown(f"""
                        <div style="background-color: {color}15; border-left: 4px solid {color}; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                            <h4 style="color: {color}; margin: 0;">{icon} {display_name}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show top 3 predictions
                        for i, (pred_variety, confidence) in enumerate(all_predictions[model_name]):
                            if pred_variety:  # Only show non-empty predictions
                                rank = i + 1
                                if rank == 1:
                                    bg_color = f"{color}25"
                                    text_weight = "bold"
                                    rank_icon = "🥇"
                                elif rank == 2:
                                    bg_color = f"{color}15"
                                    text_weight = "normal"
                                    rank_icon = "🥈"
                                else:
                                    bg_color = f"{color}10"
                                    text_weight = "normal"
                                    rank_icon = "🥉"
                                
                                st.markdown(f"""
                                <div style="background-color: {bg_color}; padding: 0.8rem; border-radius: 0.3rem; margin: 0.3rem 0;">
                                    <span style="font-weight: {text_weight}; color: {color};">
                                        {rank_icon} {pred_variety} - {confidence:.1f}%
                                    </span>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Summary of top predictions
                    st.markdown("### 📋 Ringkasan Rekomendasi Utama")
                    summary_data = []
                    for model_name, pred_list in all_predictions.items():
                        if "decision_tree" in model_name.lower():
                            display_name = "Decision Tree"
                        elif "naive_bayes" in model_name.lower():
                            display_name = "Naive Bayes"
                        elif "svm" in model_name.lower():
                            display_name = "SVM"
                        else:
                            display_name = model_name.title()
                            
                            if pred_list[0][0]:  # If there's a prediction
                                summary_data.append({
                                    "Model": display_name,
                                    "Rekomendasi Utama": pred_list[0][0],
                                    "Confidence": f"{pred_list[0][1]:.1f}%"
                                })
                    
                    if summary_data:
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True)
                    
                    # Show diversity
                    main_predictions = [pred_list[0][0] for pred_list in all_predictions.values() if pred_list[0][0]]
                    unique_predictions = set(main_predictions)
                    if len(unique_predictions) > 1:
                        st.success(f"✅ Keragaman tercapai! Ditemukan {len(unique_predictions)} varietas berbeda: {', '.join(sorted(unique_predictions))}")
                    else:
                        st.warning(f"⚠️ Semua model merekomendasikan varietas yang sama: {list(unique_predictions)[0] if unique_predictions else 'Tidak ada'}")
                        
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan dalam prediksi: {str(e)}")
                
                # Consensus voting
                if len(predictions) > 1:
                    st.markdown("### 🗳️ Rekomendasi Semua Model")
                    
                    # Count votes
                    from collections import Counter
                    vote_counts = Counter(predictions.values())
                    most_common = vote_counts.most_common()
                    
                    if len(most_common) > 0:
                        winner, vote_count = most_common[0]
                        
                        # Check if there's a clear winner
                        if vote_count > len(predictions) // 2:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #FFD700, #FFA500); padding: 2rem; border-radius: 1rem; text-align: center; margin: 1rem 0; box-shadow: 0 6px 12px rgba(0,0,0,0.15);">
                                <h3 style="color: #8B4513; margin: 0;">🏆 Rekomendasi Padi</h3>
                                <h1 style="color: #8B0000; margin: 0.5rem 0; font-size: 2.5rem;">{winner}</h1>
                                <p style="color: #8B4513; margin: 0; font-size: 1.1rem;">Dipilih oleh {vote_count} dari {len(predictions)} model</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #87CEEB, #4682B4); padding: 2rem; border-radius: 1rem; text-align: center; margin: 1rem 0;">
                                <h3 style="color: white; margin: 0;">🤝 Rekomendasi Mayoritas</h3>
                                <h1 style="color: #FFD700; margin: 0.5rem 0; font-size: 2.5rem;">{winner}</h1>
                                <p style="color: white; margin: 0; font-size: 1.1rem;">Dipilih oleh {vote_count} dari {len(predictions)} model</p>
                            </div>
                            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🌾 Aplikasi Prediksi Varietas Padi | By Adi Sahrul Ramadhan menggunakan Streamlit</p>
</div>
""", unsafe_allow_html=True)