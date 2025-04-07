# Analisis Pelanggan Ritel Online

Proyek ini melakukan klasterisasi dan klasifikasi pada data pelanggan ritel online.

## Langkah-langkah Pengaturan

### 1. Salin Repository
```sh
git clone <repository_url>
cd <repository_name>
```

### 2. Buat dan Aktifkan Virtual Environment
```sh
python -m venv venv
# Untuk Windows
venv\Scripts\activate
# Untuk macOS/Linux
source venv/bin/activate
```

### 3. Instal Semua Kebutuhan
```sh
pip install -r requirements.txt
```

### 4. Unduh Dataset
Dataset akan diunduh secara otomatis saat menjalankan notebook. Atau, Anda bisa menjalankan perintah berikut:
```sh
python download_dataset.py
```

### 5. Jalankan Notebook
Buka Jupyter Notebook atau Jupyter Lab:
```sh
jupyter notebook  # atau jupyter lab
```

Kemudian buka dan jalankan notebook berikut:
- **`[Clustering]_Submission_Akhir_BMLP_Andakara_Athaya_Sidiq_(Updated).ipynb`** - Untuk analisis segmentasi pelanggan
- **`[Klasifikasi]_Submission_Akhir_BMLP_Andakara_Athaya_Sidiq.ipynb`** - Untuk analisis klasifikasi pelanggan

## Kebutuhan Sistem
- Python 3.12+
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Yellowbrick
- Seaborn