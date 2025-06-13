import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
from typing import Dict, Any, Optional
import sys
from tqdm.auto import tqdm
from colorama import Fore, Style

# Import configuration and helpers
sys.path.append('.')
from config import BASE_URL, MAX_PAGES, MAX_PRODUCTS
from utils.helpers import setup_logging

# Get logger
logger = setup_logging()

def get_page(page_number: int) -> Optional[BeautifulSoup]:
    """
    Fetch a single page from the Fashion Studio website
    
    Args:
        page_number: The page number to fetch
        
    Returns:
        BeautifulSoup object of the page or None if request fails
    """
    url = f"{BASE_URL}/page{page_number}" if page_number > 1 else BASE_URL
    
    try:
        logger.info(f"Fetching page {page_number} from {url}")
        response = requests.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')
    except requests.RequestException as e:
        logger.error(f"Error fetching page {page_number}: {e}")
        return None

def extract_product_data(product_card: BeautifulSoup) -> Dict[str, Any]:
    """
    Extract product information from a product card element
    
    Args:
        product_card: BeautifulSoup object of the product card HTML element
        
    Returns:
        Dictionary with product data
    """
    try:
        # Extract title
        title_elem = product_card.select_one('.product-title')
        title = title_elem.text if title_elem else "Unknown"
        
        # Extract price
        price_elem = product_card.select_one('.price')
        price = price_elem.text if price_elem else "Price Unavailable"
        
        # Extract rating
        rating_text = product_card.select_one('p:-soup-contains("Rating:")').text if product_card.select_one('p:-soup-contains("Rating:")') else ""
        
        # Extract colors
        colors_text = product_card.select_one('p:-soup-contains("Colors")').text if product_card.select_one('p:-soup-contains("Colors")') else ""
        
        # Extract size
        size_text = product_card.select_one('p:-soup-contains("Size:")').text if product_card.select_one('p:-soup-contains("Size:")') else ""
        
        # Extract gender
        gender_text = product_card.select_one('p:-soup-contains("Gender:")').text if product_card.select_one('p:-soup-contains("Gender:")') else ""

        # Current timestamp
        timestamp = datetime.now()

        return {
            "Title": title,
            "Price": price,
            "Rating": rating_text,
            "Colors": colors_text,
            "Size": size_text,
            "Gender": gender_text,
            "Timestamp": timestamp
        }
    
    except Exception as e:
        logger.error(f"Error extracting product data: {e}")
        return {
            "Title": "Error",
            "Price": "Error",
            "Rating": "Error",
            "Colors": "Error", 
            "Size": "Error",
            "Gender": "Error",
            "Timestamp": datetime.now()
        }

def scrape_all_products(max_pages: int = MAX_PAGES, max_products: int = MAX_PRODUCTS) -> pd.DataFrame:
    """
    Scrape products from all pages up to max_pages or max_products
    
    Args:
        max_pages: Maximum number of pages to scrape
        max_products: Maximum number of products to scrape
        
    Returns:
        DataFrame with all scraped product data
    """
    all_products = []
    current_page = 1
    
    try:
        # Create colored progress bar for pages
        pbar = tqdm(
            total=min(max_pages, max_products//20), 
            desc=f"{Fore.BLUE}Scraping pages{Style.RESET_ALL}",
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Style.RESET_ALL)
        )
        
        while current_page <= max_pages and len(all_products) < max_products:
            soup = get_page(current_page)
            
            if not soup:
                logger.warning(f"Couldn't fetch page {current_page}, stopping.")
                break
                
            # Find product cards on the page
            product_cards = soup.select('.collection-card')
            
            if not product_cards:
                logger.warning(f"No product cards found on page {current_page}, stopping.")
                break
                
            # Extract data from each product card
            for card in product_cards:
                if len(all_products) >= max_products:
                    break
                    
                product_data = extract_product_data(card)
                all_products.append(product_data)
                
            logger.info(f"Scraped {len(all_products)} products so far")
            
            # Update progress bar
            pbar.update(1)
            
            # Go to next page
            current_page += 1
            
            # Add a small delay to be respectful to the server
            time.sleep(1)
        
        # Close the progress bar
        pbar.close()
    
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
    
    return pd.DataFrame(all_products)

def extract_data(max_pages: int = MAX_PAGES, max_products: int = MAX_PRODUCTS) -> pd.DataFrame:
    """
    Main function to extract data from Fashion Studio website
    
    Args:
        max_pages: Maximum number of pages to scrape
        max_products: Maximum number of products to scrape
    
    Returns:
        DataFrame with raw extracted data
    """
    logger.info("Starting data extraction process")
    start_time = time.time()
    
    try:
        products_df = scrape_all_products(max_pages=max_pages, max_products=max_products)
        
        logger.info(f"Extraction completed. {len(products_df)} products extracted.")
        logger.info(f"Extraction took {time.time() - start_time:.2f} seconds.")
        
        return products_df
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        # Return empty DataFrame in case of failure
        return pd.DataFrame()

if __name__ == "__main__":
    # Test the extraction function
    df = extract_data(max_pages=2, max_products=10)  
    print(f"{Fore.GREEN}Extracted {len(df)} products{Style.RESET_ALL}")
    print(df.head())