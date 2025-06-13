import pytest
import pandas as pd
import sys
from datetime import datetime

sys.path.append('.')
from utils.transform import transform_data, convert_price

@pytest.fixture
def sample_raw_data():
    return pd.DataFrame({
        'Title': ['T-shirt', 'Hoodie', 'Invalid'],
        'Price': ['$99.99', '$149.99', 'Price Unavailable'],
        'Rating': ['Rating: 4.5/5.0', 'Rating: 3.9/5.0', 'Invalid'],
        'Colors': ['3 Colors', '2 Colors', 'Invalid'],
        'Size': ['Size: M', 'Size: L', 'Invalid'],
        'Gender': ['Gender: Men', 'Gender: Women', 'Invalid'],
        'Timestamp': [datetime.now(), datetime.now(), datetime.now()]
    })

def test_convert_price():
    assert isinstance(convert_price('$99.99'), float)
    assert convert_price('Invalid') is None

def test_transform_data(sample_raw_data):
    result = transform_data(sample_raw_data)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2  # Invalid row should be removed
    assert 'Price' in result.columns
    assert result['Price'].dtype == float
    assert result['Rating'].dtype == float
    assert result['Colors'].dtype == int