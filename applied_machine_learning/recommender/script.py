# %%
# Import library yang dibutuhkan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import r2_score

# Setting untuk visualisasi
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette('Set2')

# Setting random seed untuk reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)

# %%
torch.cuda.is_available()

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
device

# %%
torch.set_float32_matmul_precision('medium')

# %% [markdown]
# # Data Loading & Initial Dataset Preparation

# %% [markdown]
# Pada tahap ini saya akan memuat dan melakukan persiapan dataset awal dengan membersihkan dataset untuk membantu tahap EDA selanjutnya.

# %% [markdown]
# ## Load Data

# %% [markdown]
# Pertama-tama saya akan memuat, memeriksa sampel, dan informasi basic dari dataset.

# %%
base_path = 'data/ml-100k/'  

# %%
# Memuat data rating
ratings = pd.read_csv(f'{base_path}u.data', sep='\t', encoding='latin-1')

# Memuat data film

movies = pd.read_csv(f'{base_path}u.item', sep='|', encoding='latin-1')

# Memuat data pengguna

users = pd.read_csv(f'{base_path}u.user', sep='|', encoding='latin-1')


# %%
print(ratings.head())

# %%
print(movies.head())

# %%
print(users.info())

# %%
print(movies.info())

# %%
print(ratings.info())

# %% [markdown]
# Terlihat bahwa data masih belum memiliki nama-nama kolom dan terdapat juga kolom yang perlu dikonversi ke format yang kompatibel dengan pandas, seperti kolom `release_date` di dataframe movies.

# %% [markdown]
# ## Pemberian Nama Kolom

# %% [markdown]
# Langkah ini akan menambahkan nama-nama kolom tersebut yang bisa didapatkan melalui [README](data\ml-100k\README) dari dataset.

# %%
ratings.columns = ['user_id', 'movie_id', 'rating', 'timestamp']

# %%
movies_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
               'unknown', 'Action', 'Adventure', 'Animation', 'Children',
               'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
               'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
               'War', 'Western']
movies.columns = movies_cols

# %%
users.columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']

# %% [markdown]
# ## Periksa Missing Value dan Duplikat

# %% [markdown]
# Selanjutnya saya perlu memeriksa missing value dan duplikat di dataset

# %%
print("\nMissing values di dataset ratings:")
print(ratings.isnull().sum())

print("\nMissing values di dataset movies:")
print(movies.isnull().sum())

print("\nMissing values di dataset users:")
print(users.isnull().sum())

# %%
# Memeriksa duplikat di masing-masing dataset
print(f"\nJumlah duplikat di dataset ratings: {ratings.duplicated().sum()}")
print(f"Jumlah duplikat di dataset movies: {movies.duplicated(subset='movie_id').sum()}")
print(f"Jumlah duplikat di dataset users: {users.duplicated(subset='user_id').sum()}")

# %% [markdown]
# Terlihat bahwa tidak ada duplikat di dataset, namun masih ada missing values yang pada dua kolom di dataset `movies`, yakni `release_date` dan `video_release_date`

# %% [markdown]
# ## Missing Values Handling

# %% [markdown]
# Untuk menangani missing values pada kolom `release_date`, saya perlu mengubah tipe datanya menjadi format `datetime` Pandas terlebih dahulu.

# %%
# Menangani missing value pada tanggal jika ditemukan
if movies['release_date'].isnull().sum() > 0:
    # Ubah ke datetime, dengan mengabaikan kesalahan
    movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
    print(f"Jumlah tanggal rilis yang tidak valid: {movies['release_date'].isnull().sum()}")
    # Drop baris dengan tanggal rilis yang tidak valid
    movies.dropna(subset=['release_date'], inplace=True)

if movies['video_release_date'].isnull().sum() > 0:
    movies['video_release_date'].fillna('', inplace=True)
    print("Nilai kosong pada video_release_date telah diisi dengan string kosong")
    
if movies['imdb_url'].isnull().sum() > 0:
    movies['imdb_url'].fillna('', inplace=True)
    print("Nilai kosong pada imdb_url telah diisi dengan string kosong")

# %%
# Cek kembali hasil penanganan missing value
print("\nMissing values di dataset movies:")
print(movies.isnull().sum())


# %% [markdown]
# ## Data Merging

# %% [markdown]
# Selanjutnya saya akan menggabungkan ketiga datraframe menjadi satu dataframe untuk membantu EDA nanti. Oleh karena itu, khusus untuk dataframe movies, saya perlu membuat kolom baru berisi genre yang sebelumnya terpisah-pisah.

# %%
# Menciptakan kolom genre sebagai list untuk setiap film
genre_columns = movies_cols[5:]
movies['genres'] = movies.apply(
    lambda row: [genre for genre, is_genre in zip(genre_columns, row[genre_columns]) if is_genre == 1],
    axis=1
)

# Menggabungkan data
data = ratings.merge(movies[['movie_id', 'title', 'genres']], on='movie_id')
data = data.merge(users[['user_id', 'gender', 'age', 'occupation']], on='user_id')


# %%
# Melihat info dataset
print("\nInformasi dataset gabungan:")
data.info()

# %%
print("\nContoh data:")
data.head()

# %% [markdown]
# # Exploratory Data Analysis (EDA)

# %% [markdown]
# Waktunya untuk EDA untuk melihat distribusi dan insight data

# %%
print(users.describe())

# %%
print(movies.describe())

# %%
print(ratings.describe())

# %%
print(data.describe())

# %%
# Melihat distribusi rating
plt.figure(figsize=(10, 6))
rating_counts = data['rating'].value_counts().sort_index()
sns.barplot(x=rating_counts.index, y=rating_counts.values)
plt.title('Distribusi Rating Film', fontsize=16)
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Jumlah', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
for i, count in enumerate(rating_counts.values):
    plt.text(i, count + 500, f'{count:,}', ha='center', fontsize=10)
plt.show()

print("Distribusi rating:")
for rating, count in rating_counts.items():
    print(f"{rating} stars: {count}")

# %%
# Distribusi genre film
all_genres = []
for genres in movies['genres']:
    all_genres.extend(genres)

genre_counts = pd.Series(all_genres).value_counts()

plt.figure(figsize=(12, 8))
sns.barplot(x=genre_counts.values, y=genre_counts.index)
plt.title('Distribusi Genre Film', fontsize=16)
plt.xlabel('Jumlah Film', fontsize=12)
plt.ylabel('Genre', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
for i, count in enumerate(genre_counts.values):
    plt.text(count + 10, i, f'{count}', va='center', fontsize=10)
plt.show()

# %%
# Distribusi jumlah rating per pengguna
user_ratings_count = data.groupby('user_id').size()

plt.figure(figsize=(12, 6))
sns.histplot(user_ratings_count, bins=50)
plt.title('Distribusi Jumlah Rating per Pengguna', fontsize=16)
plt.xlabel('Jumlah Rating', fontsize=12)
plt.ylabel('Jumlah Pengguna', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

print(f"Rata-rata jumlah rating per pengguna: {user_ratings_count.mean():.2f}")
print(f"Median jumlah rating per pengguna: {user_ratings_count.median():.2f}")
print(f"Min jumlah rating per pengguna: {user_ratings_count.min()}")
print(f"Max jumlah rating per pengguna: {user_ratings_count.max()}")

# %%
# Distribusi jumlah rating per film
movie_ratings_count = data.groupby('movie_id').size()

plt.figure(figsize=(12, 6))
sns.histplot(movie_ratings_count, bins=50)
plt.title('Distribusi Jumlah Rating per Film', fontsize=16)
plt.xlabel('Jumlah Rating', fontsize=12)
plt.ylabel('Jumlah Film', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

print(f"Rata-rata jumlah rating per film: {movie_ratings_count.mean():.2f}")
print(f"Median jumlah rating per film: {movie_ratings_count.median():.2f}")
print(f"Min jumlah rating per film: {movie_ratings_count.min()}")
print(f"Max jumlah rating per film: {movie_ratings_count.max()}")

# %%
# Film dengan rating terbanyak
top_rated_movies = movie_ratings_count.sort_values(ascending=False).head(10)
top_movies_df = pd.DataFrame({
    'movie_id': top_rated_movies.index,
    'count': top_rated_movies.values
})
top_movies_df = top_movies_df.merge(movies[['movie_id', 'title']], on='movie_id')

plt.figure(figsize=(14, 8))
sns.barplot(x='count', y='title', data=top_movies_df)
plt.title('10 Film dengan Rating Terbanyak', fontsize=16)
plt.xlabel('Jumlah Rating', fontsize=12)
plt.ylabel('Judul Film', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
for i, count in enumerate(top_movies_df['count']):
    plt.text(count + 10, i, f'{count}', va='center', fontsize=10)
plt.show()

# %%
# Rating rata-rata per film (untuk film dengan minimal 50 rating)
movie_avg_ratings = data.groupby('movie_id')['rating'].agg(['mean', 'count'])
popular_movies = movie_avg_ratings[movie_avg_ratings['count'] >= 50].sort_values('mean', ascending=False).head(10)
popular_movies = popular_movies.reset_index()
popular_movies = popular_movies.merge(movies[['movie_id', 'title']], on='movie_id')

plt.figure(figsize=(14, 8))
sns.barplot(x='mean', y='title', data=popular_movies)
plt.title('10 Film Terbaik (Min. 50 Rating)', fontsize=16)
plt.xlabel('Rating Rata-rata', fontsize=12)
plt.ylabel('Judul Film', fontsize=12)
plt.xlim(4.0, 5.0)  # Menyesuaikan skala untuk melihat perbedaan lebih jelas
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
for i, (mean, count) in enumerate(zip(popular_movies['mean'], popular_movies['count'])):
    plt.text(mean + 0.01, i, f'Rating: {mean:.2f} (dari {count} rating)', va='center', fontsize=10)
plt.show()

# %% [markdown]
# Key Insights EDA:
# - Distribusi rating cenderung positif dengan mayoritas rating 4 (33,950) dan 3 (27,145) bintang, menunjukkan adanya bias positif yang perlu diantisipasi dalam pengembangan model rekomendasi.
# - Genre Drama, Comedy, dan Action mendominasi dataset, yang mengindikasikan ketidakseimbangan genre yang perlu diperhatikan untuk content-based filtering.
# - Terdapat variasi signifikan dalam jumlah rating per pengguna (1-737) dan per film (1-583), mencerminkan masalah sparsity yang umum dalam sistem rekomendasi.
# - Film-film populer mendapatkan jauh lebih banyak rating, yang dapat menciptakan popularity bias dalam collaborative filtering.

# %% [markdown]
# # Data Preparation untuk Modeling

# %% [markdown]
# Sebelum membangun model rekomendasi, perlu dilakukan persiapan data yang sesuai untuk model content-based filtering dan collaborative filtering. Pada tahap ini, saya akan:
# 
# 1. **Membagi dataset** menjadi data training (80%) dan testing (20%) untuk evaluasi model
# 2. **Membuat mapping ID** untuk mengubah user_id dan movie_id menjadi indeks berurutan yang diperlukan model PyTorch
# 3. **Menerapkan mapping** tersebut ke dataset training dan testing
# 
# Untuk content-based filtering, saya akan memanfaatkan fitur genre film yang telah diproses sebelumnya. Sementara untuk collaborative filtering, kita akan menggunakan matriks interaksi user-film yang direpresentasikan oleh indeks dan nilai rating. 

# %%
test_size=0.2
random_state=42

# Pisahkan data untuk training dan testing
train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)

# Untuk collaborative filtering, buat mapping user_id dan movie_id ke indeks
user_ids = data['user_id'].unique()
movie_ids = data['movie_id'].unique()

user2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
movie2idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

# Terapkan mapping
train_data['user_idx'] = train_data['user_id'].map(user2idx)
train_data['movie_idx'] = train_data['movie_id'].map(movie2idx)

test_data['user_idx'] = test_data['user_id'].map(user2idx)
test_data['movie_idx'] = test_data['movie_id'].map(movie2idx)
    
print(f"Total data: {len(data)}")
print(f"Training data: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
print(f"Testing data: {len(test_data)} ({len(test_data)/len(data)*100:.1f}%)")
print(f"Jumlah pengguna: {len(user2idx)}")
print(f"Jumlah film: {len(movie2idx)}")

# %% [markdown]
# # Modeling

# %% [markdown]
# Pada tahap modeling, saya akan mengembangkan tiga pendekatan untuk sistem rekomendasi film:
# 1. **Content-Based Filtering**: Model ini merekomendasikan film berdasarkan kemiripan konten (genre) dengan film yang disukai pengguna. Menggunakan TF-IDF dan cosine similarity untuk mengukur kemiripan antar film.
# 2. **Collaborative Filtering**: Model ini menggunakan pola rating pengguna untuk menemukan kesamaan preferensi antar pengguna atau film, menggunakan matriks user-item yang telah dipersiapkan.
# 
# Mengenai metrik evaluasi, precision@k akan digunakan untuk content-based dan RMSE serta R² akan digunakan untuk collaborative filtering. Tujuannya adalah membandingkan kinerja kedua pendekatan dan mengidentifikasi kelebihan serta keterbatasan masing-masing untuk kasus rekomendasi film. 

# %% [markdown]
# ## Content-Based Filtering

# %%
class ContentBasedRecommender:
    """
    Content-based filtering untuk rekomendasi film berdasarkan kemiripan genre
    """
    def __init__(self, movies_df):
        self.movies_df = movies_df
        self.tfidf = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        
    def fit(self):
        """
        Mempersiapkan model dengan menghitung similarity matrix
        """
        print("Mempersiapkan content-based recommender...")
        
        # Mengubah list genre menjadi string untuk TF-IDF
        self.movies_df['genres_str'] = self.movies_df['genres'].apply(lambda x: ' '.join(x))
        
        # Menghitung TF-IDF matrix
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies_df['genres_str'])
        
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
        # Menghitung cosine similarity antar film
        print("Menghitung cosine similarity...")
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Mapping indeks film
        self.movies_df = self.movies_df.reset_index(drop=True)
        self.indices = pd.Series(self.movies_df.index, index=self.movies_df['title'])
        
        print("Content-based recommender siap digunakan")
        return self
    
    def recommend(self, title, n_recommendations=10):
        """
        Memberikan rekomendasi film berdasarkan kesamaan dengan film yang diberikan
        """
        # Mendapatkan indeks film yang sesuai dengan judul
        try:
            idx = self.indices[title]
        except KeyError:
            return f"Film dengan judul '{title}' tidak ditemukan dalam dataset."
        
        # Mendapatkan skor kesamaan dengan semua film
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Mengurutkan film berdasarkan skor kesamaan
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Mendapatkan n film teratas (kecuali film itu sendiri)
        sim_scores = sim_scores[1:n_recommendations+1]
        
        # Mendapatkan indeks film
        movie_indices = [i[0] for i in sim_scores]
        
        # Mengembalikan informasi film yang direkomendasikan
        recommendations = []
        for idx in movie_indices:
            recommendations.append({
                'title': self.movies_df['title'].iloc[idx],
                'genres': self.movies_df['genres'].iloc[idx],
                'similarity_score': sim_scores[movie_indices.index(idx)][1]
            })
        
        return recommendations
    
    def get_top_n_for_user(self, user_ratings, n_recommendations=10):
        """
        Memberikan rekomendasi film untuk pengguna berdasarkan film yang telah mereka nilai
        """
        # Simpan semua skor kesamaan
        all_scores = {}
        
        # Iterasi pada setiap film yang telah dinilai oleh pengguna
        for title, rating in user_ratings.items():
            try:
                idx = self.indices[title]
                sim_scores = list(enumerate(self.cosine_sim[idx]))
                
                # Sesuaikan skor kesamaan dengan rating pengguna
                weighted_scores = [(i, score * (rating/5.0)) for i, score in sim_scores]
                
                for i, score in weighted_scores:
                    movie_title = self.movies_df['title'].iloc[i]
                    if movie_title not in user_ratings:  # Hanya rekomendasikan film yang belum dinilai
                        all_scores[i] = all_scores.get(i, 0) + score
            except KeyError:
                print(f"Film '{title}' tidak ditemukan dalam dataset.")
                continue
        
        # Urutkan film berdasarkan skor
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        # Mengembalikan informasi film yang direkomendasikan
        recommendations = []
        for idx, score in sorted_scores:
            recommendations.append({
                'title': self.movies_df['title'].iloc[idx],
                'genres': self.movies_df['genres'].iloc[idx],
                'recommendation_score': score
            })
        
        return recommendations



# %% [markdown]
# Pada implementasi content-based filtering, saya mengembangkan kelas ContentBasedRecommender yang memanfaatkan representasi TF-IDF dari genre film untuk menghitung kemiripan antar konten. Model ini mengubah daftar genre menjadi representasi vektor, kemudian menghitung matriks kesamaan kosinus untuk memetakan hubungan kemiripan setiap film dengan film lainnya. Kelas ini menyediakan dua pendekatan rekomendasi: rekomendasi berdasarkan satu film referensi melalui metode recommend() dan rekomendasi berdasarkan profil pengguna melalui get_top_n_for_user() yang mempertimbangkan rating sebagai bobot preferensi.

# %% [markdown]
# ### Latih Model

# %% [markdown]
# Selanjutnya saya tinggal fit data film ke model dan training.

# %%
# Buat dan latih content-based recommender
content_recommender = ContentBasedRecommender(movies)
content_recommender.fit()

# %% [markdown]
# ### Percobaan Inference

# %% [markdown]
# Selanjutnya saya akan mencoba menggunakan model yang sudah dilatih pada contoh sampel data film dan sampel data pengguna 

# %%
# Contoh penggunaan content-based recommender untuk film tertentu
movie_title = "Legends of the Fall (1994)"
recommendations = content_recommender.recommend(movie_title, n_recommendations=10)

if isinstance(recommendations, str):
    print(recommendations)
else:
    print(f"Rekomendasi film yang mirip dengan '{movie_title}':")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']} - Genre: {', '.join(rec['genres'])} - Skor Kemiripan: {rec['similarity_score']:.4f}")

# %%
# Contoh penggunaan content-based recommender untuk pengguna dengan beberapa rating
user_ratings = {
    "Legends of the Fall (1994)": 5,
    "Star Wars (1977)": 5,
    "Pulp Fiction (1994)": 4,
    "Shawshank Redemption, The (1994)": 5,
    "Terminator, The (1984)": 3
}

recommendations = content_recommender.get_top_n_for_user(user_ratings, n_recommendations=10)

print("Rekomendasi film berdasarkan rating pengguna:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec['title']} - Genre: {', '.join(rec['genres'])} - Skor Rekomendasi: {rec['recommendation_score']:.4f}")

# %% [markdown]
# Terlihat bahwa model bisa bekerja dengan baik! 

# %% [markdown]
# ### Evaluasi

# %% [markdown]
# Pada bagian ini, saya akan mengevaluasi kualitas rekomendasi yang dihasilkan oleh model Content-Based Recommender dengan menggunakan metrik precision@k. Metrik ini mengukur proporsi film yang direkomendasikan yang memiliki kesamaan genre dengan film referensi, sehingga dapat menilai relevansi rekomendasi dari perspektif konten. Evaluasi akan dilakukan terhadap beberapa film populer sebagai sampel untuk mendapatkan gambaran umum tentang efektivitas model dalam memberikan rekomendasi yang sesuai dengan preferensi genre pengguna

# %%
# Fungsi untuk menghitung precision@k untuk evaluasi content-based recommender
def calculate_precision_at_k(recommender, test_movie, k=10):
    """
    Menghitung precision@k untuk sebuah film uji
    """
    # Dapatkan genre film uji
    try:
        test_idx = recommender.indices[test_movie]
        test_genres = set(recommender.movies_df['genres'].iloc[test_idx])
    except KeyError:
        return None
    
    # Dapatkan rekomendasi
    recommendations = recommender.recommend(test_movie, n_recommendations=k)
    if isinstance(recommendations, str):
        return None
    
    # Hitung precision
    relevant_count = 0
    for rec in recommendations:
        rec_genres = set(rec['genres'])
        if len(test_genres.intersection(rec_genres)) > 0:  # Jika ada genre yang sama
            relevant_count += 1
    
    precision = relevant_count / k
    return precision

# Evaluasi content-based recommender untuk beberapa film populer
popular_movies_titles = [
    "Toy Story (1995)",
    "Star Wars (1977)",
    "Pulp Fiction (1994)",
    "Shawshank Redemption, The (1994)",
    "Jurassic Park (1993)"
]

print("Evaluasi Content-Based Recommender dengan Precision@10:")
precisions = []
for title in popular_movies_titles:
    precision = calculate_precision_at_k(content_recommender, title, k=10)
    if precision is not None:
        precisions.append(precision)
        print(f"{title} - Precision@10: {precision:.2f}")

print(f"\nRata-rata Precision@10: {np.mean(precisions):.4f}")

# %% [markdown]
# Hasil evaluasi menunjukkan bahwa model Content-Based Recommender mencapai skor Precision@10 sempurna (1.0) pada semua film populer yang diuji, membuktikan bahwa sistem rekomendasi berhasil memberikan rekomendasi yang sangat relevan dengan selalu menyertakan film-film dengan genre yang serupa dengan film referensi.

# %% [markdown]
# ## Collaborative Filtering

# %% [markdown]
# Untuk pembuatan model collaborative filtering, saya akan nembangun model neural network menggunakan Pytorch dibantu dengan PyTorch Lightning. 

# %% [markdown]
# ### Setup DataModule

# %% [markdown]
# Serupa dengan TensorFlow dan Keras, pada PyTorch saya juga perlu mempersiapkan terlebih dahulu sebuah struktur data yang dibutuhkan untuk menangani proses loading, batching, dan transformasi data rating film untuk tahap training dan evaluasi model. 

# %%
class MovieLensDataset(Dataset):
    """Dataset MovieLens untuk collaborative filtering"""
    
    def __init__(self, ratings_df):
        self.ratings = ratings_df
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        user_idx = self.ratings.iloc[idx]['user_idx']
        movie_idx = self.ratings.iloc[idx]['movie_idx']
        rating = self.ratings.iloc[idx]['rating']
        
        return {
            'user_idx': torch.tensor(user_idx, dtype=torch.long).to(device),
            'movie_idx': torch.tensor(movie_idx, dtype=torch.long).to(device),
            'rating': torch.tensor(rating, dtype=torch.float).to(device)
        }


class MovieLensDataModule(pl.LightningDataModule):
    """Lightning DataModule untuk MovieLens dataset"""
    
    def __init__(self, train_df, test_df, batch_size=64):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size
        
    def setup(self, stage=None):
        self.train_dataset = MovieLensDataset(self.train_df)
        self.test_dataset = MovieLensDataset(self.test_df)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,

        )
        
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,

        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0

        )

# %% [markdown]
# Di sini saya mengikuti best practice dengan mendefinisikan kelas `Dataset` dari PyTorch dalam membuat struktur data dan menggunakan library `DataModule` dari Lightning untuk mempersipakan DataLoader.

# %% [markdown]
# ### Build Model

# %%
class CollaborativeFilteringModel(pl.LightningModule):
    """Model collaborative filtering menggunakan matrix factorization"""
    
    def __init__(self, n_users, n_movies, embedding_dim=100, dropout=0.2, 
                 l2_reg=1e-5, lr=1e-3, lr_scheduler=True):
        super().__init__()
        
        # Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)
        
        # Global bias
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Non-linear layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim//2, embedding_dim//4),
            nn.ReLU()
        )
        
        self.final_layer = nn.Linear(embedding_dim//4, 1)
        
        # Hyperparameters
        self.embedding_dim = embedding_dim
        self.l2_reg = l2_reg
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        
        self.test_preds = []
        self.test_targets = []
        
        # Inisialisasi bobot
        self._init_weights()
        self.save_hyperparameters()
        
    def _init_weights(self):
        """Initialize model weights"""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)
        
    def forward(self, user_idx, movie_idx):

        user_emb = self.user_embedding(user_idx)
        movie_emb = self.movie_embedding(movie_idx)
        user_b = self.user_bias(user_idx).squeeze()
        movie_b = self.movie_bias(movie_idx).squeeze()
        
        interaction = user_emb * movie_emb
        
        x = self.fc_layers(interaction)
        x = self.final_layer(x).squeeze()
        pred = x + user_b + movie_b + self.global_bias
        
        return pred
    
    def training_step(self, batch, batch_idx):
        user_idx = batch['user_idx']
        movie_idx = batch['movie_idx']
        rating = batch['rating']
        
        pred = self(user_idx, movie_idx)
        loss = nn.MSELoss()(pred, rating)
        
        # L2 regularization
        if self.l2_reg > 0:
            l2_reg = 0
            for param in self.parameters():
                l2_reg += torch.norm(param)**2
            loss += self.l2_reg * l2_reg
            
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        user_idx = batch['user_idx']
        movie_idx = batch['movie_idx']
        rating = batch['rating']
        
        pred = self(user_idx, movie_idx)
        loss = nn.MSELoss()(pred, rating)
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        user_idx = batch['user_idx']
        movie_idx = batch['movie_idx']
        rating = batch['rating']
        
        pred = self(user_idx, movie_idx)
        loss = nn.MSELoss()(pred, rating)
        
        # Kalkulasi R² 
        self.test_preds.append(pred)
        self.test_targets.append(rating)
        
        # Menghitung RMSE
        rmse = torch.sqrt(loss)
        self.log('test_rmse', rmse, prog_bar=True)
        
        return {'test_loss': loss, 'test_rmse': rmse}
    
    def on_test_epoch_start(self):
        # Reset list saat awal epoch testing
        self.test_preds = []
        self.test_targets = []
    
    def on_test_epoch_end(self):
        # Konkatenasi semua prediksi dan target
        preds = torch.cat(self.test_preds)
        targets = torch.cat(self.test_targets)
        
        # Kalkulasi R²
        targets_mean = torch.mean(targets)
        ss_tot = torch.sum((targets - targets_mean) ** 2)
        ss_res = torch.sum((targets - preds) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        # Log R² 
        self.log('test_r2', r2, prog_bar=True)
        print(f"R² Score: {r2.item():.4f}")
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_reg)
        
        if self.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss"
            }
        return optimizer

    
    def recommend_movies(self, user_idx, movie_data, n_recommendations=10, exclude_rated=True, rated_movies=None):
        """
        Memberikan rekomendasi film untuk pengguna tertentu
        """
        self.eval()  # Set model ke mode evaluasi
        
        # Konversi user_idx ke tensor
        user_tensor = torch.tensor([user_idx] * len(movie_data), dtype=torch.long)
        
        # Mendapatkan seluruh indeks film
        movie_indices = torch.tensor(movie_data.index.values, dtype=torch.long)
        
        with torch.no_grad():
            # Prediksi rating untuk semua film
            predicted_ratings = self(user_tensor, movie_indices)
            
        # Konversi ke NumPy
        predicted_ratings = predicted_ratings.cpu().numpy()
        
        # Masukkan prediksi ke movie_data
        recommendations = movie_data.copy()
        recommendations['predicted_rating'] = predicted_ratings
        
        # Jika perlu, hapus film yang telah dinilai
        if exclude_rated and rated_movies is not None:
            recommendations = recommendations[~recommendations['movie_id'].isin(rated_movies)]
        
        # Sortir berdasarkan prediksi rating tertinggi
        recommendations = recommendations.sort_values(by='predicted_rating', ascending=False)
        
        # Kembalikan n film teratas
        return recommendations.head(n_recommendations)

# %% [markdown]
# - Jadi di sini saya mengimplementasikan arsitektur neural network collaborative filtering berbasis matrix factorization yang merepresentasikan pengguna dan film dalam ruang embedding berdimensi rendah dengan memanfaatkan embedding layer. 
# - Model ini mempelajari vektor embedding untuk setiap pengguna dan film, serta parameter bias untuk masing-masing, dengan prediksi rating dihitung melalui dot product antara embedding pengguna dan film ditambah dengan bias.
# - Implementasi menggunakan MSE loss untuk training dan validasi, serta RMSE dan R² untuk evaluasi, dengan optimizer Adam untuk pembelajaran parameter.
# - Model ini akan memprediksi rating untuk semua film bagi pengguna tertentu dan mengembalikan rekomendasi film teratas yang belum pernah dinilai pengguna.

# %% [markdown]
# ### Latih Model

# %% [markdown]
# Waktunya latih model dengan fit data movies ke model yang dibuat. Dalam melatih saya akan mengikuti best practice dengan mengimplementasikan checkpointing untuk mempermudah menyimpan dan melanjutkan pelatihan model, early stopping untuk menghindari overfitting, serta logging hasil training model ke CSV melalui callback.

# %%
# Siapkan DataModule
data_module = MovieLensDataModule(train_data, test_data, batch_size=512)

# Siapkan model
n_users = len(user2idx)
n_movies = len(movie2idx)
model = CollaborativeFilteringModel(n_users, n_movies, embedding_dim=100, lr = 5e-4)

# %%


# Callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath='./models',
    filename='collaborative_model-{epoch:02d}-{val_loss:.4f}',
    save_top_k=2,
    monitor='val_loss',
    mode='min'
)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=7,          
    mode='min',
    min_delta=0.001      
)


# Logger
logger = CSVLogger(save_dir='./logs', name='collaborative_filtering')

# Trainer
trainer = pl.Trainer(
    max_epochs=50,
    callbacks=[checkpoint_callback, early_stop_callback],
    logger=logger,
    accelerator='gpu',
    devices=1, 
    log_every_n_steps=10,
    num_sanity_val_steps=0,
    gradient_clip_val=1.0,
    deterministic=True 
)

# %%
trainer.fit(model, data_module)

# %%
torch.save(model.state_dict(), './models/collaborative_model_final.pt')

# %% [markdown]
# Model collaborative filtering dilatih dengan konfigurasi maksimum 100 epoch dan early stopping patience 5. Terlihat bahwa training diperhentikan oleh early stopping di epoch 21 dengan training loss dan validation loss menunjukkan penurunan yang stabil selama proses pembelajaran walaupun loss akhir tidak benar-benar kecil. 

# %%
# model_path = './models/collaborative_model_final.pt'    
# n_users = len(user2idx)
# n_movies = len(movie2idx)

# model = CollaborativeFilteringModel(n_users, n_movies)
# # Cek apakah file model tersedia
# try:
#     model.load_state_dict(torch.load(model_path))
#     print(f"Model collaborative filtering berhasil dimuat dari {model_path}")
# except:
#     print(f"Model tidak ditemukan di {model_path}. Menggunakan model yang baru dilatih.")
#     # Jika model file tidak ada, gunakan model yang baru saja dilatih
# model.eval()

# %% [markdown]
# ### Percobaan Inference

# %% [markdown]
# Setelah melatih model collaborative filtering, selanjutnya saya akan menguji kemampuan model dalam memberikan rekomendasi film yang relevan. Pada tahap ini, saya akan mengambil pengguna spesifik (user_id=196) untuk melihat bagaimana model merekomendasikan film berdasarkan pola rating yang telah dipelajari. Proses ini meliputi identifikasi film yang sudah dinilai oleh pengguna untuk kemudian dikeluarkan dari rekomendasi, mempersiapkan data input yang diperlukan model, dan akhirnya memprediksi rating untuk film-film yang belum ditonton untuk menghasilkan daftar rekomendasi personal bagi pengguna tersebut. 

# %%
user_id = 196
data[data['user_id'] == user_id]

# %%
rated_movies = data[data['user_id'] == user_id]['movie_id'].tolist()

rated_movies

# %%
# Siapkan data film untuk rekomendasi
movie_data = movies[['movie_id', 'title', 'genres']].copy()
movie_data['movie_idx'] = movie_data['movie_id'].map(movie2idx)
movie_data.set_index('movie_idx', inplace=True)

# Dapatkan rekomendasi
try:
    recommendations = model.recommend_movies(
        user2idx[user_id],
        movie_data,
        n_recommendations=10,
        exclude_rated=True,
        rated_movies=rated_movies
    )
    
    print(f"Rekomendasi film untuk pengguna {user_id}:")
    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
        print(f"{i}. {row['title']} - Genre: {', '.join(row['genres'])} - Prediksi Rating: {row['predicted_rating']:.2f}")
except Exception as e:
    print(f"Error: {e}")
    print("Pastikan model sudah dilatih dan parameter user_id valid.")


# %% [markdown]
# Percobaan inference berhasil mendemonstrasikan kemampuan model dalam memberikan rekomendasi film personal untuk pengguna spesifik (user_id 196) 

# %% [markdown]
# ### Evaluasi Collaborative Filtering

# %% [markdown]
# Selanjutnya saya akan mencoba mengevaluasi model melalui learning curve dan kedua metrik yang disiapkan.

# %%
try:
    # Baca log training
    log_df = pd.read_csv('./logs/collaborative_filtering/version_0/metrics.csv')
    
    # Pisahkan log train dan validation
    train_logs = log_df[log_df['train_loss'].notna()]
    val_logs = log_df[log_df['val_loss'].notna()]
    
    # Plot loss
    plt.figure(figsize=(12, 6))
    plt.plot(train_logs['epoch'], train_logs['train_loss'], label='Train Loss')
    plt.plot(val_logs['epoch'], val_logs['val_loss'], label='Validation Loss')
    plt.title('Collaborative Filtering Model Training', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()
except Exception as e:
    print(f"Tidak dapat memuat log training: {e}")
    print("Pastikan model sudah dilatih dan log tersedia.")

# %%
test_results = trainer.test(model, data_module)

# %% [markdown]
# Model collaborative filtering menunjukkan pembelajaran dengan penurunan loss yang stabil serta konvergensi yang baik selama proses pelatihan. Terlihat juga dari gap antara training dan validation loss yang sempit mengindikasikan model tidak mengalami overfitting atau underfitting. 
# 
# Hasil evaluasi pada data testing menghasilkan:
# - **RMSE (Root Mean Square Error)**: 0.9307 - menunjukkan rata-rata prediksi model menyimpang sekitar 0.93 bintang dari rating sebenarnya pada skala 1-5.
# - **R² Score**: 0.3165 - mengindikasikan model dapat menjelaskan sekitar 31.6% variasi dalam rating pengguna.
# 
# Walaupun dari pembelajaran yang baik, terlihat dari R² yang rendah bahwa model Collaborative Filtering akan memiliki performa yang kurang optimal dalam merekomendasi.


