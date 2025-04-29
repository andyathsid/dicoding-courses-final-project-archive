# Analisis Sentimen Ulasan Aplikasi

Proyek ini mengimplementasikan model machine learning dan deep learning untuk mengklasifikasikan sentimen ulasan aplikasi dari Google Play Store.

## Prasyarat

- Python 3.12
- pip dan virtualenv

## Instalasi

1. Clone repositori ini
2. Buat lingkungan virtual Python dan aktifkan:

```bash
python -m venv .dicoding-lmld-venv
source .dicoding-lmld-venv/bin/activate # Linux/Mac
# atau
.\.dicoding-lmld-venv\Scripts\activate # Windows
```

3. Pasang dependensi yang diperlukan:

```bash
pip install -r requirements.txt
```

## Struktur Proyek

- **notebooks/**: Jupyter notebook untuk eksplorasi data dan pemodelan
  - **data_engineering.ipynb**: Proses persiapan data
  - **modelling.ipynb**: Pembuatan dan evaluasi model
- **script/**: Kode Python untuk pengambilan data
  - **play_store_scraper.py**: Script untuk mengambil ulasan dari Google Play Store
- **data/**: Direktori untuk menyimpan data
- **checkpoints/**: Model yang sudah dilatih (BERT dan RoBERTa)

## Penggunaan

### Mengumpulkan Data

Untuk mengambil ulasan dari Google Play Store:

```bash
python script/play_store_scraper.py [APP_ID] --output ulasan --language id --country id
```

Contoh:

```bash
python script/play_store_scraper.py com.gojek.app --output gojek_reviews --language id --country id
```

### Menjalankan Analisis

Buka notebook dengan Jupyter:

```bash
jupyter notebook notebooks/modelling.ipynb
```

Jalankan semua sel untuk melatih model atau menggunakan model yang sudah disimpan di direktori checkpoints.