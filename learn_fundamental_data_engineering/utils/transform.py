import pandas as pd
import re
from datetime import datetime
import sys

sys.path.append('.')
from config import USD_TO_IDR
from utils.helpers import setup_logging

# Get logger
logger = setup_logging()

def convert_price(price_str):
    price_match = re.search(r'\$(\d+\.\d+)', price_str)
    if price_match:
        price_usd = float(price_match.group(1))
        price_idr = price_usd * USD_TO_IDR
        return float(price_idr) 
    return None

def transform_data(df):
    """
    Transform the DataFrame by removing invalid rows and converting data
    
    Args:
        df: DataFrame with raw data
        
    Returns:
        Cleaned DataFrame
    """
    # Make an explicit copy of the DataFrame to avoid the SettingWithCopyWarning
    df = df.copy()
    
    # Remove rows with missing or invalid titles
    initial_count = len(df)
    df = df[df['Title'].notna() & (df['Title'] != '')]
    removed_titles = initial_count - len(df)
    logger.info(f"Removed {removed_titles} rows with invalid titles")
    
    # Remove rows with unavailable prices
    initial_count = len(df)
    df = df[~df['Price'].str.contains('Unavailable', na=False, case=False)]
    removed_prices = initial_count - len(df)
    logger.info(f"Removed {removed_prices} rows with unavailable prices")
    
    # Convert price from string to numeric
    # Using loc to avoid the SettingWithCopyWarning
    df.loc[:, 'Price'] = df['Price'].apply(convert_price)
    
    # Make sure Price is explicitly converted to float type
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce').astype(float)
    
    # Clean Rating column to extract numeric ratings
    def extract_rating(rating_str):
        rating_match = re.search(r'(\d+\.\d+)', str(rating_str))
        if rating_match:
            return float(rating_match.group(1))
        return None
    
    df['Rating'] = df['Rating'].apply(extract_rating)
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').astype(float)
    initial_count = len(df)
    df = df.dropna(subset=['Rating'])
    removed_ratings = initial_count - len(df)
    logger.info(f"Removed {removed_ratings} rows with invalid ratings")
    
    # Clean Colors column to extract the number of colors
    def extract_colors(colors_str):
        color_match = re.search(r'(\d+)', str(colors_str))
        if color_match:
            return int(color_match.group(1))
        return None
    
    df['Colors'] = df['Colors'].apply(extract_colors)
    # Changed from Int64 to regular int type to match test expectations
    df['Colors'] = pd.to_numeric(df['Colors'], errors='coerce').astype(int)
    initial_count = len(df)
    df = df.dropna(subset=['Colors'])
    removed_colors = initial_count - len(df)
    logger.info(f"Removed {removed_colors} rows with invalid colors")
    
    # Clean Size column to extract size values
    def extract_size(size_str):
        size_match = re.search(r'Size: (\w+)', str(size_str))
        if size_match:
            return size_match.group(1)
        return None
    
    df['Size'] = df['Size'].apply(extract_size)
    initial_count = len(df)
    df = df.dropna(subset=['Size'])
    removed_sizes = initial_count - len(df)
    logger.info(f"Removed {removed_sizes} rows with invalid sizes")
    
    # Clean Gender column to extract gender values
    def extract_gender(gender_str):
        gender_match = re.search(r'Gender: (\w+)', str(gender_str))
        if gender_match:
            return gender_match.group(1)
        return None
    
    df['Gender'] = df['Gender'].apply(extract_gender)
    initial_count = len(df)
    df = df.dropna(subset=['Gender'])
    removed_genders = initial_count - len(df)
    logger.info(f"Removed {removed_genders} rows with invalid genders")
    
    # Remove duplicate products
    initial_count = len(df)
    df = df.drop_duplicates(subset=['Title', 'Price', 'Rating', 'Colors', 'Size', 'Gender'])
    removed_duplicates = initial_count - len(df)
    logger.info(f"Removed {removed_duplicates} duplicate rows")
    
    return df

if __name__ == "__main__":
    sample_data = {
        "Title": ["T-shirt", "Unknown Product", "Hoodie"],
        "Price": ["$100.00", "Price Unavailable", "$150.50"],
        "Rating": ["Rating: ⭐ 4.5 / 5", "Rating: Invalid Rating", "Rating: ⭐ 3.9 / 5"],
        "Colors": ["3 Colors", "5 Colors", "2 Colors"],
        "Size": ["Size: M", "Size: L", "Size: XL"],
        "Gender": ["Gender: Men", "Gender: Women", "Gender: Unisex"],
        "Timestamp": [datetime.now(), datetime.now(), datetime.now()]
    }
    
    sample_df = pd.DataFrame(sample_data)
    transformed_df = transform_data(sample_df)
    print(transformed_df)