import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base URL for scraping
BASE_URL = "https://fashion-studio.dicoding.dev"
MAX_PAGES = 50
MAX_PRODUCTS = 1000

# Dollar to Rupiah conversion rate
USD_TO_IDR = 16000

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL")

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = "fashion_products"  # Table name in Supabase

# Google Sheets configuration
GOOGLE_SHEETS_CREDS_FILE = "google-sheets-api.json"
GOOGLE_SHEETS_ID = os.getenv("GOOGLE_SHEETS_ID")