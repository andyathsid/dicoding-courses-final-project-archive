import pytest
import pandas as pd
import sys
from bs4 import BeautifulSoup
from unittest.mock import patch, MagicMock, call
from requests.exceptions import RequestException

sys.path.append('.')
from utils.extract import get_page, extract_product_data, extract_data, scrape_all_products

@pytest.fixture
def sample_product_html():
    return """
    <div class="collection-card">
        <h2 class="product-title">Test Product</h2>
        <span class="price">$99.99</span>
        <p>Rating: 4.5/5.0</p>
        <p>Colors: 3</p>
        <p>Size: M</p>
        <p>Gender: Unisex</p>
    </div>
    """

@pytest.fixture
def sample_page_html():
    return """
    <html>
        <body>
            <div class="collection-card">
                <h2 class="product-title">Product 1</h2>
                <span class="price">$99.99</span>
                <p>Rating: 4.5/5.0</p>
                <p>Colors: 3</p>
                <p>Size: M</p>
                <p>Gender: Men</p>
            </div>
            <div class="collection-card">
                <h2 class="product-title">Product 2</h2>
                <span class="price">$149.99</span>
                <p>Rating: 4.2/5.0</p>
                <p>Colors: 2</p>
                <p>Size: L</p>
                <p>Gender: Women</p>
            </div>
        </body>
    </html>
    """

def test_get_page():
    with patch('utils.extract.requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.content = "<html></html>"
        mock_get.return_value = mock_response
        
        result = get_page(1)
        assert isinstance(result, BeautifulSoup)
        mock_get.assert_called_once()

def test_get_page_request_exception():
    with patch('utils.extract.requests.get') as mock_get:
        mock_get.side_effect = RequestException("Connection error")
        result = get_page(1)
        assert result is None

def test_extract_product_data(sample_product_html):
    soup = BeautifulSoup(sample_product_html, 'html.parser')
    result = extract_product_data(soup)
    
    assert isinstance(result, dict)
    assert result['Title'] == "Test Product"
    assert result['Price'] == "$99.99"
    assert "4.5" in result['Rating']

def test_extract_product_data_exception():
    broken_soup = BeautifulSoup("<div></div>", 'html.parser')
    result = extract_product_data(broken_soup)
    
    assert isinstance(result, dict)
    assert result['Title'] == "Unknown"
    assert result['Price'] == "Price Unavailable"

def test_scrape_all_products(sample_page_html):
    with patch('utils.extract.get_page') as mock_get_page:
        # Set up the mock to return our sample page for page 1, and None for page 2
        mock_get_page.side_effect = [
            BeautifulSoup(sample_page_html, 'html.parser'),
            None  # Simulates failure on page 2
        ]
        
        with patch('utils.extract.time.sleep'):  # Don't actually sleep during tests
            result = scrape_all_products(max_pages=2, max_products=10)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            mock_get_page.assert_has_calls([call(1), call(2)])

def test_scrape_all_products_no_cards():
    with patch('utils.extract.get_page') as mock_get_page:
        # Return a page with no product cards
        mock_get_page.return_value = BeautifulSoup("<html><body></body></html>", 'html.parser')
        
        with patch('utils.extract.time.sleep'):
            result = scrape_all_products(max_pages=1, max_products=10)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

def test_scrape_all_products_max_products(sample_page_html):
    with patch('utils.extract.get_page') as mock_get_page:
        # Return a page with two product cards but set max_products to 1
        mock_get_page.return_value = BeautifulSoup(sample_page_html, 'html.parser')
        
        with patch('utils.extract.time.sleep'):
            result = scrape_all_products(max_pages=2, max_products=1)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1 

def test_extract_data():
    with patch('utils.extract.scrape_all_products') as mock_scrape:
        mock_df = pd.DataFrame({
            'Title': ['Product 1', 'Product 2'],
            'Price': ['$99.99', '$149.99']
        })
        mock_scrape.return_value = mock_df
        
        result = extract_data(max_pages=1, max_products=2)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

def test_extract_data_exception():
    with patch('utils.extract.scrape_all_products') as mock_scrape:
        mock_scrape.side_effect = Exception("Scraping failed")
        
        result = extract_data(max_pages=1, max_products=2)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0  # Should return empty DataFrame on failure