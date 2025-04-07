import csv
import time
from typing import List, Dict, Any, Tuple, Optional
from google_play_scraper import Sort, reviews
from tqdm import tqdm
from colorama import Fore, Style, init
import logging
import os
import sys

init()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fetch_reviews(app_id: str, language: str = 'en', country: str = 'us',
                 reviews_per_score: int = 20, balanced: bool = True, 
                 sentiment_balanced: bool = False, verbose: bool = False) -> List[Dict[Any, Any]]:
    """
    Mengambil ulasan untuk aplikasi tertentu dari Google Play Store.
    
    Args:
        app_id (str): ID aplikasi Google Play Store.
        language (str, optional): Kode bahasa untuk ulasan. Default 'en'.
        country (str, optional): Kode negara. Default 'us'.
        reviews_per_score (int, optional): Jumlah ulasan yang diinginkan per skor. Default 20.
        balanced (bool, optional): Jika True, ambil jumlah ulasan yang sama untuk setiap skor.
        sentiment_balanced (bool, optional): Jika True, ambil ulasan yang seimbang untuk analisis sentimen 
                                           (negatif: skor 1-2, netral: skor 3, positif: skor 4-5).
        verbose (bool, optional): Jika True, tampilkan informasi progres secara detail.
        
    Returns:
        List[Dict[Any, Any]]: Daftar kamus ulasan.
    """
    try:
        all_reviews = []
        
        if sentiment_balanced:
            if verbose:
                print(f"{Fore.CYAN}Membuat dataset seimbang untuk analisis sentimen (negatif/netral/positif){Style.RESET_ALL}")
            else:
                print(f"Mengumpulkan ulasan seimbang untuk analisis sentimen...")
            
            # Calculate reviews per sentiment class (negative, neutral, positive)
            total_reviews = reviews_per_score * 5
            reviews_per_sentiment = total_reviews // 3
            
            # Define the target distribution for each score
            reviews_per_rating = {
                1: reviews_per_sentiment // 2,  # Half of negative class
                2: reviews_per_sentiment // 2,  # Half of negative class
                3: reviews_per_sentiment,       # All of neutral class
                4: reviews_per_sentiment // 2,  # Half of positive class
                5: reviews_per_sentiment // 2   # Half of positive class
            }
            
            # Adjust for any rounding issues
            remaining = total_reviews - sum(reviews_per_rating.values())
            if remaining > 0:
                # Distribute remaining evenly
                for score in range(1, 6):
                    if remaining > 0:
                        reviews_per_rating[score] += 1
                        remaining -= 1
                    else:
                        break
            
            if verbose:
                print(f"{Fore.BLUE}Target distribusi ulasan per skor:")
                for score, count in reviews_per_rating.items():
                    print(f"  Skor {score}: {count} ulasan")
                print(f"  Total: {sum(reviews_per_rating.values())} ulasan{Style.RESET_ALL}")
            
            # Fetch reviews for each score based on the sentiment-balanced distribution
            for score in range(1, 6):
                target_count = reviews_per_rating[score]
                if target_count == 0:
                    continue
                
                score_reviews = []
                if verbose:
                    print(f"\n{Fore.BLUE}Memproses ulasan dengan skor {score} (target: {target_count}){Style.RESET_ALL}")
                
                # Get MOST_RELEVANT reviews (70% of total)
                relevant_limit = int(target_count * 0.7)
                relevant_reviews = []
                continuation_token = None
                
                with tqdm(total=relevant_limit, desc=f"Skor {score} - Relevan", 
                         disable=not verbose, ncols=80) as pbar:
                    while len(relevant_reviews) < relevant_limit:
                        batch, continuation_token = reviews(
                            app_id=app_id,
                            lang=language,
                            country=country,
                            sort=Sort.MOST_RELEVANT,
                            count=100,  
                            filter_score_with=score,
                            continuation_token=continuation_token
                        )
                        
                        relevant_reviews.extend(batch)
                        pbar.update(min(len(batch), relevant_limit - pbar.n))
                        
                        if not batch or not continuation_token or len(relevant_reviews) >= relevant_limit:
                            break
                
                if len(relevant_reviews) < relevant_limit and verbose:
                    print(f"{Fore.YELLOW}⚠️ Hanya menemukan {len(relevant_reviews)}/{relevant_limit} ulasan relevan dengan skor {score}{Style.RESET_ALL}")
                
                # Get NEWEST reviews (30% of total)
                newest_limit = target_count - len(relevant_reviews[:relevant_limit])
                newest_reviews = []
                continuation_token = None
                
                with tqdm(total=newest_limit, desc=f"Skor {score} - Terbaru", 
                         disable=not verbose, ncols=80) as pbar:
                    while len(newest_reviews) < newest_limit:
                        batch, continuation_token = reviews(
                            app_id=app_id,
                            lang=language,
                            country=country,
                            sort=Sort.NEWEST,
                            count=100,  
                            filter_score_with=score,
                            continuation_token=continuation_token
                        )
                        
                        newest_reviews.extend(batch)
                        pbar.update(min(len(batch), newest_limit - pbar.n))
                        
                        if not batch or not continuation_token or len(newest_reviews) >= newest_limit:
                            break
            
                if len(newest_reviews) < newest_limit and verbose:
                    print(f"{Fore.YELLOW}⚠️ Hanya menemukan {len(newest_reviews)}/{newest_limit} ulasan terbaru dengan skor {score}{Style.RESET_ALL}")
                
                score_reviews = relevant_reviews[:relevant_limit] + newest_reviews[:newest_limit]
                all_reviews.extend(score_reviews)
                
                if not verbose:
                    sys.stdout.write(f"\rMengumpulkan... Skor {score}/5: {len(score_reviews)} ulasan")
                    sys.stdout.flush()
                else:
                    print(f"{Fore.GREEN}✓ Skor {score}: {len(score_reviews)} ulasan "
                          f"({min(len(relevant_reviews), relevant_limit)} relevan, "
                          f"{min(len(newest_reviews), newest_limit)} terbaru){Style.RESET_ALL}")
                    
            # Print sentiment class distribution
            if verbose:
                negative_reviews = sum(1 for review in all_reviews if review['score'] in [1, 2])
                neutral_reviews = sum(1 for review in all_reviews if review['score'] == 3)
                positive_reviews = sum(1 for review in all_reviews if review['score'] in [4, 5])
                
                print(f"\n{Fore.BLUE}Distribusi sentimen akhir:")
                print(f"  Negatif (skor 1-2): {negative_reviews} ulasan ({negative_reviews/len(all_reviews)*100:.1f}%)")
                print(f"  Netral (skor 3): {neutral_reviews} ulasan ({neutral_reviews/len(all_reviews)*100:.1f}%)")
                print(f"  Positif (skor 4-5): {positive_reviews} ulasan ({positive_reviews/len(all_reviews)*100:.1f}%)")
                print(f"  Total: {len(all_reviews)} ulasan{Style.RESET_ALL}")
        
        elif balanced:
            if verbose:
                print(f"{Fore.CYAN}Membuat dataset seimbang dengan ulasan untuk setiap skor (1-5){Style.RESET_ALL}")
            else:
                print(f"Mengumpulkan ulasan seimbang (target: {reviews_per_score} per skor)...")
            
            for score in range(1, 6):
                score_reviews = []
                if verbose:
                    print(f"\n{Fore.BLUE}Memproses ulasan dengan skor {score}{Style.RESET_ALL}")
                
                # Dapatkan ulasan MOST_RELEVANT (70% dari total)
                relevant_limit = int(reviews_per_score * 0.7)
                relevant_reviews = []
                continuation_token = None
                
                with tqdm(total=relevant_limit, desc=f"Skor {score} - Relevan", 
                         disable=not verbose, ncols=80) as pbar:
                    while len(relevant_reviews) < relevant_limit:
                        batch, continuation_token = reviews(
                            app_id=app_id,
                            lang=language,
                            country=country,
                            sort=Sort.MOST_RELEVANT,
                            count=100,  
                            filter_score_with=score,
                            continuation_token=continuation_token
                        )
                        
                        relevant_reviews.extend(batch)
                        pbar.update(min(len(batch), relevant_limit - pbar.n))
                        
                        if not batch or not continuation_token or len(relevant_reviews) >= relevant_limit:
                            break
                
                if len(relevant_reviews) < relevant_limit and verbose:
                    print(f"{Fore.YELLOW}⚠️ Hanya menemukan {len(relevant_reviews)}/{relevant_limit} ulasan relevan dengan skor {score}{Style.RESET_ALL}")
                
                # Dapatkan ulasan TERBARU (30% dari total)
                newest_limit = reviews_per_score - len(relevant_reviews[:relevant_limit])
                newest_reviews = []
                continuation_token = None
                
                with tqdm(total=newest_limit, desc=f"Skor {score} - Terbaru", 
                         disable=not verbose, ncols=80) as pbar:
                    while len(newest_reviews) < newest_limit:
                        batch, continuation_token = reviews(
                            app_id=app_id,
                            lang=language,
                            country=country,
                            sort=Sort.NEWEST,
                            count=100,  
                            filter_score_with=score,
                            continuation_token=continuation_token
                        )
                        
                        newest_reviews.extend(batch)
                        pbar.update(min(len(batch), newest_limit - pbar.n))
                        
                        if not batch or not continuation_token or len(newest_reviews) >= newest_limit:
                            break
            
                if len(newest_reviews) < newest_limit and verbose:
                    print(f"{Fore.YELLOW}⚠️ Hanya menemukan {len(newest_reviews)}/{newest_limit} ulasan terbaru dengan skor {score}{Style.RESET_ALL}")
                
                score_reviews = relevant_reviews[:relevant_limit] + newest_reviews[:newest_limit]
                all_reviews.extend(score_reviews)
                
                if not verbose:
                    sys.stdout.write(f"\rMengumpulkan... Skor {score}/5: {len(score_reviews)} ulasan")
                    sys.stdout.flush()
                else:
                    print(f"{Fore.GREEN}✓ Skor {score}: {len(score_reviews)} ulasan "
                          f"({min(len(relevant_reviews), relevant_limit)} relevan, "
                          f"{min(len(newest_reviews), newest_limit)} terbaru){Style.RESET_ALL}")
        
        else:
            # Jika tidak seimbang, ambil semua ulasan tanpa menyeimbangkan
            if verbose:
                print(f"{Fore.YELLOW}Mengambil ulasan tanpa menyeimbangkan skor{Style.RESET_ALL}")
            else:
                print(f"Mengumpulkan ulasan tidak seimbang (target: {reviews_per_score*5})...")
                
            continuation_token = None
            
            with tqdm(desc="Mengambil ulasan", unit=" ulasan", 
                     disable=not verbose, ncols=80) as pbar:
                while True:
                    batch, continuation_token = reviews(
                        app_id=app_id,
                        lang=language,
                        country=country,
                        sort=Sort.MOST_RELEVANT,
                        count=100,
                        continuation_token=continuation_token
                    )
                    
                    all_reviews.extend(batch)
                    pbar.update(len(batch))
                    
                    if not verbose:
                        sys.stdout.write(f"\rTelah mengumpulkan {len(all_reviews)} ulasan sejauh ini...")
                        sys.stdout.flush()
                    
                    if not continuation_token or (reviews_per_score * 5 and len(all_reviews) >= reviews_per_score * 5):
                        break
            
            if reviews_per_score:
                all_reviews = all_reviews[:reviews_per_score * 5]
        
        print(f"\n{Fore.GREEN}✓ Berhasil mengambil {len(all_reviews)} ulasan untuk {app_id}{Style.RESET_ALL}")
        return all_reviews
        
    except Exception as e:
        print(f"\n{Fore.RED}✗ Kesalahan saat mengambil ulasan untuk {app_id}: {str(e)}{Style.RESET_ALL}")
        logger.error(f"Kesalahan saat mengambil ulasan untuk {app_id}: {str(e)}")
        return []

def save_to_csv(reviews: List[Dict[Any, Any]], filename: str, overwrite: bool = False,
               verbose: bool = False) -> None:
    """
    Menyimpan ulasan ke file CSV.
    
    Args:
        reviews (List[Dict[Any, Any]]): Daftar kamus ulasan.
        filename (str): Nama file CSV output.
        overwrite (bool, optional): Apakah akan menimpa file yang sudah ada. Default False.
        verbose (bool, optional): Jika True, tampilkan informasi progres secara detail.
    
    Raises:
        FileExistsError: Jika file sudah ada dan overwrite adalah False.
    """
    try:
        if not reviews:
            print(f"{Fore.YELLOW}Tidak ada ulasan untuk disimpan ke {filename}{Style.RESET_ALL}")
            return
            
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            if verbose:
                print(f"{Fore.BLUE}Membuat direktori: {directory}{Style.RESET_ALL}")
        
        if os.path.exists(filename) and not overwrite:
            error_msg = f"File '{filename}' sudah ada. Gunakan nama file yang berbeda untuk mencegah kehilangan data atau atur overwrite=True."
            print(f"{Fore.RED}✗ Kesalahan: {error_msg}{Style.RESET_ALL}")
            raise FileExistsError(error_msg)
            
        fieldnames = reviews[0].keys()
        
        if verbose:
            print(f"\n{Fore.BLUE}Menyimpan {len(reviews)} ulasan ke {filename}{Style.RESET_ALL}")
        else:
            print(f"Menyimpan {len(reviews)} ulasan ke {filename}...")
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for review in tqdm(reviews, desc="Menyimpan ulasan", unit=" ulasan", 
                              disable=not verbose, ncols=80):
                writer.writerow(review)
                    
        print(f"{Fore.GREEN}✓ Berhasil menyimpan semua ulasan ke {filename}{Style.RESET_ALL}")
    except FileExistsError:
        raise
    except Exception as e:
        print(f"{Fore.RED}✗ Kesalahan menyimpan ulasan: {str(e)}{Style.RESET_ALL}")
        logger.error(f"Kesalahan menyimpan ulasan: {str(e)}")
        raise

class PlayStoreReviews:
    """Class interface untuk bekerja dengan ulasan Google Play Store."""
    
    def __init__(self, app_id: str = None, language: str = 'en', country: str = 'us', verbose: bool = False):
        """
        Inisialisasi objek PlayStoreReviews.
        
        Args:
            app_id (str, optional): ID aplikasi Google Play Store. Dapat diatur nanti.
            language (str, optional): Kode bahasa untuk ulasan. Default 'en'.
            country (str, optional): Kode negara. Default 'us'.
            verbose (bool, optional): Jika True, tampilkan informasi progres secara detail.
        """
        self.app_id = app_id
        self.language = language
        self.country = country
        self.verbose = verbose
        self.reviews = []
    
    def fetch(self, app_id: str = None, reviews_per_score: int = 20, 
              balanced: bool = True, sentiment_balanced: bool = False) -> List[Dict[Any, Any]]:
        """
        Ambil ulasan untuk aplikasi.
        
        Args:
            app_id (str, optional): Ganti app_id yang diatur dalam konstruktor.
            reviews_per_score (int): Jumlah ulasan per skor (1-5).
            balanced (bool): Apakah akan menyeimbangkan ulasan di semua skor.
            sentiment_balanced (bool): Apakah akan menyeimbangkan ulasan untuk analisis sentimen
                                      (negatif: skor 1-2, netral: skor 3, positif: skor 4-5).
            
        Returns:
            List[Dict[Any, Any]]: Ulasan yang diambil.
        """
        app_id = app_id or self.app_id
        if not app_id:
            raise ValueError("app_id harus disediakan baik selama inisialisasi atau saat memanggil fetch()")
            
        # If both balanced and sentiment_balanced are True, prioritize sentiment_balanced
        if balanced and sentiment_balanced:
            balanced = False
            
        self.reviews = fetch_reviews(
            app_id=app_id,
            language=self.language,
            country=self.country,
            reviews_per_score=reviews_per_score,
            balanced=balanced,
            sentiment_balanced=sentiment_balanced,
            verbose=self.verbose
        )
        return self.reviews
    
    def save(self, filename: str, overwrite: bool = False) -> None:
        """
        Simpan ulasan yang diambil ke CSV.
        
        Args:
            filename (str): Nama file output.
            overwrite (bool, optional): Apakah akan menimpa file yang sudah ada. Default False.
        """
        if not self.reviews:
            raise ValueError("Tidak ada ulasan untuk disimpan. Panggil fetch() terlebih dahulu.")
        save_to_csv(self.reviews, filename, overwrite=overwrite, verbose=self.verbose)
    
    def get_reviews(self) -> List[Dict[Any, Any]]:
        """
        Dapatkan ulasan yang telah diambil.
        
        Returns:
            List[Dict[Any, Any]]: Ulasan yang telah diambil.
        """
        return self.reviews

def main():
    """Fungsi utama untuk menjalankan pengambil ulasan dari command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ambil ulasan dari Google Play Store')
    parser.add_argument('app_ids', nargs='+', help='ID aplikasi Google Play Store (bisa lebih dari satu)')
    parser.add_argument('--output', '-o', help='Nama file CSV output (tanpa ekstensi untuk multi-app)')
    parser.add_argument('--language', '-l', default='en', help='Kode bahasa (default: en)')
    parser.add_argument('--country', '-ct', default='us', help='Kode negara (default: us)')
    parser.add_argument('--balanced', '-b', action='store_true', default=True, 
                        help='Ambil jumlah ulasan yang sama untuk setiap skor (1-5)')
    parser.add_argument('--sentiment-balanced', '-sb', action='store_true', default=False,
                        help='Ambil ulasan untuk analisis sentimen yang seimbang (negatif/netral/positif)')
    parser.add_argument('--limit-per-score', '-lps', type=int, default=100, 
                        help='Batas total ulasan yang akan dikumpulkan dari semua aplikasi')
    parser.add_argument('--combine', '-co', action='store_true', 
                        help='Gabungkan semua ulasan aplikasi menjadi satu file CSV')
    parser.add_argument('--overwrite', '-ow', action='store_true',
                        help='Timpa file yang sudah ada (default: False)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Tampilkan informasi progres secara detail')
    
    args = parser.parse_args()
    
    try:
        # Hitung ulasan per aplikasi dan per skor
        total_requested = args.limit_per_score
        num_apps = len(args.app_ids)
        
        # If both balanced and sentiment_balanced are True, prioritize sentiment_balanced
        if args.balanced and args.sentiment_balanced:
            args.balanced = False
            
        if args.sentiment_balanced:
            # For sentiment analysis: 3 classes (negative, neutral, positive)
            reviews_per_sentiment_per_app = total_requested // (num_apps * 3)
            
            # Calculate reviews per score per app for sentiment balance
            reviews_per_score_per_app = {
                # Negative class (scores 1-2)
                1: reviews_per_sentiment_per_app // 2,
                2: reviews_per_sentiment_per_app // 2,
                # Neutral class (score 3)
                3: reviews_per_sentiment_per_app,
                # Positive class (scores 4-5)
                4: reviews_per_sentiment_per_app // 2,
                5: reviews_per_sentiment_per_app // 2
            }
            
            # Ensure at least 1 review per score per app
            for score in reviews_per_score_per_app:
                reviews_per_score_per_app[score] = max(1, reviews_per_score_per_app[score])
                
            if args.verbose:
                print(f"{Fore.CYAN}Menargetkan ~{total_requested} ulasan total untuk analisis sentimen:"
                      f"\n  Negatif (skor 1-2): ~{reviews_per_sentiment_per_app * num_apps} ulasan"
                      f"\n  Netral (skor 3): ~{reviews_per_sentiment_per_app * num_apps} ulasan"
                      f"\n  Positif (skor 4-5): ~{reviews_per_sentiment_per_app * num_apps} ulasan{Style.RESET_ALL}")
            else:
                print(f"Menargetkan ~{total_requested} ulasan untuk analisis sentimen dari {num_apps} aplikasi")
                
            # Use a middle value for the command line flow - actual distribution will be handled in fetch_reviews
            reviews_per_score_per_app = reviews_per_sentiment_per_app
            
        elif args.balanced:
            # Distribusi yang sama di semua aplikasi dan semua skor (1-5)
            reviews_per_score_per_app = total_requested // (num_apps * 5)
            # Pastikan minimal 1 ulasan per skor per aplikasi
            reviews_per_score_per_app = max(1, reviews_per_score_per_app)
            if args.verbose:
                print(f"{Fore.CYAN}Menargetkan ~{total_requested} ulasan total: "
                      f"{reviews_per_score_per_app} ulasan per skor per aplikasi "
                      f"({reviews_per_score_per_app * 5} per aplikasi, {num_apps} aplikasi){Style.RESET_ALL}")
            else:
                print(f"Menargetkan ~{total_requested} ulasan dari {num_apps} aplikasi")
        else:
            # Distribusi yang sama hanya di aplikasi
            reviews_per_app = total_requested // num_apps
            reviews_per_score_per_app = reviews_per_app // 5
            # Pastikan minimal 1 ulasan per aplikasi
            reviews_per_score_per_app = max(1, reviews_per_score_per_app)
            if args.verbose:
                print(f"{Fore.CYAN}Menargetkan ~{total_requested} ulasan total: "
                      f"{reviews_per_app} ulasan per aplikasi ({num_apps} aplikasi){Style.RESET_ALL}")
            else:
                print(f"Menargetkan ~{total_requested} ulasan tidak seimbang dari {num_apps} aplikasi")
        
        all_reviews = []
        total_collected = 0
        max_reviews = total_requested
        
        for app_idx, app_id in enumerate(args.app_ids):
            # Periksa apakah kita telah mencapai batas total
            if total_collected >= total_requested:
                print(f"{Fore.YELLOW}Mencapai batas total {total_requested} ulasan. Menghentikan pengumpulan.{Style.RESET_ALL}")
                break
                
            # Hitung ulasan yang tersisa yang dapat kita kumpulkan
            remaining = max_reviews - total_collected
            
            # Jika kita telah mengumpulkan lebih dari yang diharapkan dari aplikasi sebelumnya, sesuaikan untuk aplikasi yang tersisa
            if app_idx < len(args.app_ids) - 1:  # Bukan aplikasi terakhir
                remaining_apps = len(args.app_ids) - app_idx
                if args.sentiment_balanced:
                    reviews_this_app = min(reviews_per_score_per_app * 3, remaining // remaining_apps)
                else:
                    reviews_this_app = min(reviews_per_score_per_app * 5, remaining // remaining_apps)
                reviews_per_score = reviews_this_app // 5
            else:  # Aplikasi terakhir - ambil semua yang tersisa
                reviews_per_score = max(1, remaining // 5)
            
            if args.verbose:
                print(f"\n{Fore.CYAN}=== Memproses aplikasi {app_idx+1}/{num_apps}: {app_id} "
                      f"(mengumpulkan ~{reviews_per_score * 5} ulasan) ==={Style.RESET_ALL}")
            else:
                print(f"\nAplikasi {app_idx+1}/{num_apps}: {app_id} (target: {reviews_per_score * 5} ulasan)")
            
            app_reviews = fetch_reviews(
                app_id,
                language=args.language,
                country=args.country,
                balanced=args.balanced,
                sentiment_balanced=args.sentiment_balanced,
                reviews_per_score=reviews_per_score,
                verbose=args.verbose
            )
            
            # Tambahkan field app_id ke setiap ulasan
            for review in app_reviews:
                review['app_id'] = app_id
            
            total_collected += len(app_reviews)
            
            # Hanya tampilkan progres detail dalam mode verbose
            if args.verbose:
                print(f"{Fore.BLUE}Progres: {total_collected}/{total_requested} ulasan terkumpul "
                      f"({(total_collected/max(1, total_requested)*100):.1f}%){Style.RESET_ALL}")
            else:
                print(f"Progres: {total_collected}/{total_requested} ulasan terkumpul")
            
            # Simpan ulasan aplikasi individual
            if not args.combine:
                output_file = f"{app_id}_reviews.csv" if not args.output else f"{args.output}_{app_id}.csv"
                save_to_csv(app_reviews, output_file, overwrite=args.overwrite, verbose=args.verbose)
            else:
                all_reviews.extend(app_reviews)
        
        # Simpan ulasan gabungan jika diminta
        if args.combine and all_reviews:
            output_file = "combined_reviews.csv" if not args.output else f"{args.output}.csv"
            save_to_csv(all_reviews, output_file, overwrite=args.overwrite, verbose=args.verbose)
            
        print(f"\n{Fore.GREEN}✓ Pengumpulan selesai: {total_collected} ulasan terkumpul "
              f"({min(total_collected/max(1, total_requested)*100, 100):.1f}% dari jumlah yang diminta){Style.RESET_ALL}")
            
    except FileExistsError as e:
        print(f"{Fore.RED}✗ {str(e)}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Tip: Gunakan flag --overwrite atau nama file yang berbeda untuk melanjutkan.{Style.RESET_ALL}")
        sys.exit(1)
    except Exception as e:
        print(f"{Fore.RED}✗ Kesalahan selama pengumpulan ulasan: {str(e)}{Style.RESET_ALL}")
        logger.error(f"Kesalahan selama pengumpulan ulasan: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()