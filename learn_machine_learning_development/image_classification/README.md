# Proyek Klasifikasi Gambar Berlian

Proyek ini menggunakan deep learning untuk mengklasifikasikan gambar berlian berdasarkan bentuk potongannya (pear, heart, marquise, emerald, cushion, round, oval, princess).

## Ringkasan Proyek

- Klasifikasi gambar menggunakan TensorFlow
- Model yang sudah dilatih tersedia di direktori checkpoints
- Model diekspor dalam format SavedModel, TFLite, dan TensorFlow.js

## Dataset

Proyek ini menggunakan [Diamond Images Dataset](https://www.kaggle.com/datasets/aayushpurswani/diamond-images-dataset/data) yang berisi gambar berlian dengan berbagai bentuk potongan. Dataset diorganisir berdasarkan jenis potongan berlian.

## Library yang Dibutuhkan

- Python 3.11+
- TensorFlow 2.x
- TensorFlow.js
- tflite-runtime
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- kagglehub (opsional - untuk mengunduh dataset)

## Petunjuk Pengaturan

1. **Clone atau unduh repositori ini**

2. **Masuk ke direktori proyek**
   ```bash
   cd image_classification
   ```

3. **Buat virtual environment**
   ```bash
   python -m venv .venv
   ```

4. **Aktifkan virtual environment**
   - Untuk Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - Untuk macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

5. **Pasang library yang diperlukan**
   ```bash
   pip install -r requirements.txt
   ```

## Menjalankan Proyek

### Menggunakan Jupyter Notebook
1. Jalankan Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Buka file `notebook.ipynb` di browser

3. Jalankan cell secara berurutan untuk:
   - Menyiapkan lingkungan
   - Memuat dan memproses data
   - Melatih atau memuat model
   - Mengevaluasi hasil

## Struktur Proyek
```
.
├── notebook.ipynb            # Notebook utama dengan kode dan dokumentasi
├── requirements.txt          # Paket Python yang dibutuhkan
├── checkpoints/              # Checkpoint model tersimpan
├── data/                     # Direktori dataset
│   └── test/                 # Gambar untuk pengujian
└── model/                    # File model yang diekspor
    ├── tflite_model/         # Format TF-Lite
    ├── metadata.json         # Metada model berisi urutan kelas pada training dan resolusi input gambar
    ├── saved_model/          # Format SavedModel
    └── tfjs_model/           # Format TFJS
```

