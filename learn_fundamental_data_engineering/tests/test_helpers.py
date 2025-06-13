import pytest
import pandas as pd
import sys
import logging
from unittest.mock import patch, MagicMock
from io import StringIO

sys.path.append('.')
from utils.helpers import (
    count_by_category, 
    generate_summary, 
    setup_logging,
    print_colored,
    print_column_dtypes
)

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'Title': ['Shirt', 'Pants', 'Jacket'],
        'Price': [100000.0, 150000.0, 200000.0],
        'Rating': [4.5, 4.2, 4.8],
        'Colors': [3, 2, 4],
        'Size': ['M', 'L', 'XL'],
        'Gender': ['Men', 'Women', 'Men']
    })

def test_count_by_category(sample_dataframe):
    result = count_by_category(sample_dataframe, 'Gender')
    assert result == {'Men': 2, 'Women': 1}
    
    result = count_by_category(sample_dataframe, 'Size')
    assert result == {'M': 1, 'L': 1, 'XL': 1}

def test_count_by_category_exception():
    df = pd.DataFrame({'A': [1, 2, 3]})
    # Test with non-existent column
    result = count_by_category(df, 'NonExistentColumn')
    assert result == {}

def test_generate_summary(sample_dataframe):
    result = generate_summary(sample_dataframe)
    
    assert isinstance(result, str)
    assert "Total products processed: 3" in result
    assert "Average price: Rp150,000.00" in result
    assert "Average rating: 4.50" in result
    assert "Gender distribution" in result
    assert "Size distribution" in result

def test_generate_summary_exception():
    # Create a DataFrame that will cause an exception in generate_summary
    df = pd.DataFrame({'A': [1, 2, 3]})  # Missing required columns
    
    result = generate_summary(df)
    assert "Error generating summary" in result

def test_setup_logging():
    with patch('logging.basicConfig') as mock_config:
        logger = setup_logging()
        mock_config.assert_called_once()
        args, kwargs = mock_config.call_args
        assert kwargs['level'] == logging.INFO
    
    # Test with verbose=True
    with patch('logging.basicConfig') as mock_config:
        logger = setup_logging(verbose=True)
        mock_config.assert_called_once()
        args, kwargs = mock_config.call_args
        assert kwargs['level'] == logging.DEBUG

@patch('builtins.print')
def test_print_colored(mock_print):
    # Test regular message
    print_colored("Test message")
    mock_print.assert_called_once()
    
    # Test success message
    print_colored("Success message", is_success=True)
    assert mock_print.call_count == 2
    
    # Test failure message
    print_colored("Failure message", is_success=False)
    assert mock_print.call_count == 3

@patch('builtins.print')
def test_print_column_dtypes(mock_print, sample_dataframe):
    print_column_dtypes(sample_dataframe)
    # Check that print was called at least for the header and each column (at least 7 times)
    assert mock_print.call_count >= 7