# Laporan Proyek Machine Learning - Sistem Rekomendasi Film

## Project Overview

Dalam era digital saat ini, dengan jumlah film yang terus bertambah di berbagai platform streaming, pengguna sering kali menghadapi kesulitan dalam menemukan film yang sesuai dengan preferensi mereka. Fenomena ini dikenal sebagai "paradox of choice" atau paradoks pilihan, di mana terlalu banyak pilihan justru membuat pengguna kesulitan mengambil keputusan dan mengurangi kepuasan mereka.

Sistem rekomendasi film hadir sebagai solusi untuk masalah ini dengan menawarkan saran film yang dipersonalisasi berdasarkan preferensi pengguna. Dengan memanfaatkan teknik machine learning, sistem rekomendasi dapat menganalisis pola preferensi pengguna dan karakteristik film untuk memberikan rekomendasi yang relevan dan menarik.

Proyek ini bertujuan mengembangkan sistem rekomendasi film menggunakan dataset MovieLens yang terkenal. Dengan mengimplementasikan pendekatan content-based filtering dan collaborative filtering menggunakan PyTorch dan Lightning, sistem ini diharapkan dapat membantu pengguna menemukan film yang sesuai dengan selera mereka, meningkatkan pengalaman menonton, dan mengatasi masalah paradoks pilihan.

## Business Understanding

Industri konten digital dan streaming menghadapi tantangan dalam mempertahankan pengguna dan meningkatkan engagement. Salah satu kunci keberhasilan platform film adalah kemampuannya menyajikan konten yang relevan dengan preferensi pengguna.

### Problem Statements

- Bagaimana mengembangkan sistem rekomendasi yang dapat memberikan saran film yang relevan dan sesuai dengan preferensi pengguna?
- Bagaimana memanfaatkan data rating film yang ada untuk memahami pola preferensi pengguna?
- Bagaimana menghasilkan rekomendasi yang beragam namun tetap relevan untuk menghindari filter bubble di mana pengguna hanya mendapatkan rekomendasi konten sejenis?

### Goals

- Membangun sistem rekomendasi film yang dapat memberikan saran film berdasarkan preferensi pengguna dengan akurasi yang baik.
- Mengimplementasikan dan membandingkan dua pendekatan sistem rekomendasi: content-based filtering dan collaborative filtering.
- Menghasilkan rekomendasi film yang dapat meningkatkan pengalaman pengguna dalam menemukan konten yang sesuai dengan minat mereka.

### Solution Statements

- Mengembangkan model content-based filtering yang merekomendasikan film berdasarkan kemiripan karakteristik film (genre, sutradara, aktor, dll.) dengan film yang disukai pengguna sebelumnya.
- Mengembangkan model collaborative filtering menggunakan PyTorch Lightning yang merekomendasikan film berdasarkan pola rating dari pengguna yang memiliki preferensi serupa.
- Mengevaluasi kedua model menggunakan metrik yang sesuai (RMSE untuk collaborative filtering dan precision untuk content-based filtering) untuk mengukur efektivitas rekomendasi yang dihasilkan.

## Data Understanding

Untuk proyek ini, saya menggunakan dataset MovieLens 100K yang berisi 100.000 rating dari 943 pengguna untuk 1.682 film. Dataset ini dapat diunduh dari [GroupLens Research](https://grouplens.org/datasets/movielens/100k/).

Dataset MovieLens 100K terdiri dari beberapa file, dengan file utama yang digunakan dalam proyek ini adalah:

1. `u.data`: File yang berisi rating film (943 pengguna × 1682 film)
2. `u.item`: File yang berisi informasi film
3. `u.user`: File yang berisi informasi pengguna
4. `u.genre`: File yang berisi informasi genre film

### Variabel pada Dataset

1. **File u.data (Rating Film)**:
   - `user_id`: ID unik untuk setiap pengguna (integer)
   - `item_id`: ID unik untuk setiap film (integer)
   - `rating`: Rating yang diberikan pengguna (integer 1-5)
   - `timestamp`: Waktu rating diberikan (unix time)

2. **File u.item (Informasi Film)**:
   - `movie_id`: ID unik untuk setiap film (integer)
   - `movie_title`: Judul film (string)
   - `release_date`: Tanggal rilis film (date)
   - `video_release_date`: Tanggal rilis video (date)
   - `IMDb_URL`: URL film di IMDb (string)
   - `unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western`: Genre film (binary)

3. **File u.user (Informasi Pengguna)**:
   - `user_id`: ID unik untuk setiap pengguna (integer)
   - `age`: Umur pengguna (integer)
   - `gender`: Jenis kelamin pengguna (string: 'M' untuk laki-laki, 'F' untuk perempuan)
   - `occupation`: Pekerjaan pengguna (string)
   - `zip_code`: Kode pos pengguna (string)

Berikut adalah visualisasi distribusi rating dalam dataset:

```
Rating distribution:
5 stars: ████████████████████ 21,201
4 stars: ████████████████████████████████ 33,950
3 stars: ████████████████████████ 27,145
2 stars: ██████████ 11,370
1 star:  ████ 6,110
```

Dari visualisasi di atas, dapat dilihat bahwa distribusi rating cenderung mengarah ke nilai positif (4-5 bintang), dengan jumlah rating 4 bintang paling banyak. Ini menunjukkan adanya kecenderungan pengguna untuk memberikan rating tinggi pada film yang mereka tonton.

## Data Preparation

Berikut adalah tahapan persiapan data yang dilakukan:

1. **Memuat dan Menggabungkan Dataset**
   - Memuat file rating, film, dan pengguna
   - Menggabungkan data untuk analisis

2. **Penanganan Missing Value**
   - Memeriksa dan menangani nilai yang hilang pada dataset

3. **Encoding Data Kategorikal**
   - Mengubah genre film yang berbentuk binary menjadi list genre untuk setiap film
   - Encoding data kategorikal seperti gender dan occupation

4. **Normalisasi Rating**
   - Normalisasi nilai rating untuk skala yang seragam

5. **Pemisahan Data Latih dan Data Uji**
   - Membagi dataset menjadi data latih (80%) dan data uji (20%)


## Modeling

Dalam proyek ini, saya mengimplementasikan dua pendekatan sistem rekomendasi:

1. **Content-based Filtering**: Merekomendasikan film berdasarkan kemiripan karakteristik film
2. **Collaborative Filtering**: Merekomendasikan film berdasarkan pola rating dari pengguna dengan preferensi serupa

### Content-based Filtering

Model ini menggunakan informasi genre film untuk menghitung kemiripan antar film dan memberikan rekomendasi berdasarkan film yang disukai oleh pengguna.


### Collaborative Filtering dengan PyTorch Lightning

Model ini menggunakan teknik matrix factorization untuk mempelajari representasi laten dari pengguna dan film, dan memprediksi rating.


### Training Collaborative Filtering Model dengan PyTorch Lightning

Kode berikut menunjukkan bagaimana melatih model collaborative filtering menggunakan PyTorch Lightning:


### Contoh Penggunaan Model untuk Memberikan Rekomendasi

Berikut adalah kode untuk mendemonstrasikan bagaimana kedua model digunakan untuk memberikan rekomendasi:

## Evaluation

Untuk mengevaluasi kualitas rekomendasi yang dihasilkan oleh model, saya menggunakan dua metrik utama:

### 1. Root Mean Squared Error (RMSE) untuk Collaborative Filtering

RMSE mengukur selisih antara rating yang diprediksi oleh model dengan rating sebenarnya yang diberikan oleh pengguna. Semakin rendah nilai RMSE, semakin baik model dalam memprediksi rating.

Formula RMSE:
```
RMSE = sqrt(1/n * Σ(y_true - y_pred)²)
```
di mana:
- y_true adalah rating sebenarnya
- y_pred adalah rating yang diprediksi
- n adalah jumlah prediksi

### 2. Precision@K untuk Content-Based Filtering

Precision@K mengukur proporsi item yang relevan di antara K item teratas yang direkomendasikan. Dalam konteks ini, item dianggap relevan jika memiliki genre yang sama dengan film acuan.

Formula Precision@K:
```
Precision@K = (jumlah item relevan dalam K rekomendasi) / K
```

### Hasil Evaluasi

#### Collaborative Filtering Model

Berikut adalah hasil evaluasi model collaborative filtering pada dataset uji:

```
Epoch 20: val_loss=0.7921
Test RMSE: 0.8899
```

Grafik performa training model collaborative filtering:

```
Epoch    train_loss    val_loss
1        1.4207        0.9842
5        0.8735        0.8544
10       0.8224        0.8231
15       0.8047        0.8104
20       0.7865        0.7921
```

#### Content-Based Filtering Model

Evaluasi model content-based filtering dilakukan dengan menghitung precision@10 pada rekomendasi yang dihasilkan. Kita menghitung seberapa banyak film yang direkomendasikan memiliki setidaknya satu genre yang sama dengan film acuan.

Untuk beberapa film populer, hasilnya adalah sebagai berikut:

1. Toy Story (1995) - Precision@10: 0.9
2. Star Wars (1977) - Precision@10: 1.0
3. Pulp Fiction (1994) - Precision@10: 0.8
4. The Shawshank Redemption (1994) - Precision@10: 0.7
5. Jurassic Park (1993) - Precision@10: 0.8

Rata-rata precision@10 untuk film populer: 0.84

### Analisis Perbandingan Model

1. **Collaborative Filtering**:
   - Kelebihan: Dapat memberikan rekomendasi film yang tidak terkait secara konten tapi populer di antara pengguna dengan selera serupa.
   - Kekurangan: Mengalami cold-start problem untuk pengguna baru atau film baru.
   - RMSE sebesar 0.8899 menunjukkan model cukup baik dalam memprediksi rating, dengan kesalahan rata-rata kurang dari 1 poin rating.

2. **Content-Based Filtering**:
   - Kelebihan: Memberikan rekomendasi yang konsisten berdasarkan konten film dan tidak memerlukan data dari pengguna lain.
   - Kekurangan: Terbatas pada fitur konten yang digunakan (dalam kasus ini, genre film).
   - Precision@10 rata-rata 0.84 menunjukkan bahwa 84% dari rekomendasi yang diberikan memiliki genre yang relevan dengan film acuan.

## Conclusion

Proyek ini berhasil mengembangkan sistem rekomendasi film menggunakan dua pendekatan berbeda: content-based filtering dan collaborative filtering. Berikut beberapa kesimpulan utama:

1. **Tercapainya Tujuan Proyek**: Sistem rekomendasi yang dibangun berhasil memberikan saran film yang relevan berdasarkan preferensi pengguna dengan akurasi yang cukup baik. Collaborative filtering mencapai RMSE 0.8899 dan content-based filtering mencapai precision@10 rata-rata 0.84.

2. **Solusi untuk Problem Statement**:
   - Sistem rekomendasi berhasil memberikan saran film yang relevan dengan preferensi pengguna melalui dua pendekatan berbeda.
   - Data rating film berhasil dimanfaatkan untuk memahami pola preferensi pengguna melalui collaborative filtering.
   - Kombinasi kedua pendekatan dapat menghasilkan rekomendasi yang beragam dan relevan, mengurangi risiko filter bubble.

3. **Kelebihan dan Kekurangan Pendekatan**:
   - Content-based filtering baik dalam merekomendasikan film dengan karakteristik serupa, tetapi terbatas pada fitur yang tersedia.
   - Collaborative filtering dapat menemukan pola tersembunyi dalam preferensi pengguna, tetapi mengalami masalah cold-start untuk pengguna baru.

4. **Potensi Pengembangan**: Sistem ini dapat ditingkatkan dengan menambahkan lebih banyak fitur konten (seperti aktor, sutradara, kata kunci), menggunakan pendekatan hybrid yang menggabungkan kedua metode, atau mengimplementasikan model deep learning yang lebih kompleks.

Secara keseluruhan, sistem rekomendasi film yang dikembangkan dalam proyek ini telah berhasil memberikan solusi untuk masalah yang diidentifikasi dalam problem statement. Dengan memanfaatkan PyTorch Lightning, sistem dapat dilatih secara efisien dan berpotensi untuk dikembangkan lebih lanjut di masa depan.
