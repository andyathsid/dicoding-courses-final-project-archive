import logging
from typing import Dict
import pandas as pd
from colorama import Fore, Style
from google.oauth2.service_account import Credentials
import os
import sys

sys.path.append('.')
from config import USD_TO_IDR, GOOGLE_SHEETS_CREDS_FILE

logger = logging.getLogger(__name__)

# Existing functions
def count_by_category(df: pd.DataFrame, column_name: str) -> Dict:
    """
    Count items per category in a specific column
    
    Args:
        df: DataFrame to analyze
        column_name: Column to group by
        
    Returns:
        Dictionary with counts by category
    """
    try:
        return df[column_name].value_counts().to_dict()
    except Exception as e:
        logger.error(f"Error counting categories for {column_name}: {e}")
        return {}

def generate_summary(df: pd.DataFrame) -> str:
    """
    Generate a summary of the dataset
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        String containing the summary text
    """
    try:
        total_products = len(df)
        avg_price = df['Price'].mean()
        avg_rating = df['Rating'].mean()
        
        gender_counts = count_by_category(df, 'Gender')
        gender_summary = ", ".join([f"{k}: {v}" for k, v in gender_counts.items()])
        
        size_counts = count_by_category(df, 'Size')
        size_summary = ", ".join([f"{k}: {v}" for k, v in size_counts.items()])
        
        summary = f"""
ETL Pipeline Results Summary
===========================
Total products processed: {total_products}
Average price: Rp{avg_price:,.2f}
Average rating: {avg_rating:.2f}/5.0

Gender distribution:
{gender_summary}

Size distribution:
{size_summary}
        """
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return "Error generating summary statistics"

def setup_logging(log_file="etl_pipeline.log", verbose=False):
    """Configure logging with optional verbosity level"""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def print_colored(message, color=Fore.WHITE, is_success=None):
    """Print colored message based on success/failure status"""
    if is_success is True:
        print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")
    elif is_success is False:
        print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")
    else:
        print(f"{color}{message}{Style.RESET_ALL}")
        
def print_column_dtypes(df):
    """Print the data types of each column in the DataFrame"""
    print_colored("\nColumn Datatypes:", Fore.CYAN)
    print_colored("-" * 40, Fore.CYAN)
    
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        print_colored(f"{col:<15}: {dtype_str}", Fore.WHITE)
    
    print_colored("-" * 40, Fore.CYAN)

