import pytest
import pandas as pd
import sys
import os
from unittest.mock import patch, MagicMock, call
from datetime import datetime
import gspread
from sqlalchemy.exc import SQLAlchemyError

sys.path.append('.')
from utils.load import (
    save_to_csv, 
    save_to_google_sheets, 
    save_to_postgresql,
    load_data
)

@pytest.fixture
def sample_transformed_data():
    return pd.DataFrame({
        'Title': ['Product 1', 'Product 2'],
        'Price': [1500000.0, 2000000.0],
        'Rating': [4.5, 4.0],
        'Colors': [3, 2],
        'Size': ['M', 'L'],
        'Gender': ['Men', 'Women'],
        'Timestamp': [datetime.now(), datetime.now()]
    })

def test_save_to_csv(sample_transformed_data, tmp_path):
    filepath = tmp_path / "test_output.csv"
    result = save_to_csv(sample_transformed_data, filename=str(filepath))
    
    assert result is True
    assert filepath.exists()
    loaded_df = pd.read_csv(filepath)
    assert len(loaded_df) == len(sample_transformed_data)

def test_save_to_csv_with_new_directory(sample_transformed_data, tmp_path):
    # Test creating a new directory
    filepath = tmp_path / "new_dir" / "test_output.csv"
    result = save_to_csv(sample_transformed_data, filename=str(filepath))
    
    assert result is True
    assert filepath.exists()

def test_save_to_csv_exception(sample_transformed_data):
    # Test exception handling
    with patch('pandas.DataFrame.to_csv', side_effect=Exception("CSV write error")):
        result = save_to_csv(sample_transformed_data, filename="test_output.csv")
        assert result is False

def test_save_to_google_sheets(sample_transformed_data):
    with patch('utils.load.gspread.authorize') as mock_auth:
        mock_sheet = MagicMock()
        mock_worksheet = MagicMock()
        mock_auth.return_value.open_by_key.return_value = mock_sheet
        mock_sheet.worksheet.return_value = mock_worksheet
        
        result = save_to_google_sheets(sample_transformed_data)
        assert result is True
        mock_auth.assert_called_once()

def test_save_to_google_sheets_no_credentials_file(sample_transformed_data):
    with patch('os.path.exists', return_value=False):
        result = save_to_google_sheets(sample_transformed_data)
        assert result is False

def test_save_to_google_sheets_create_new_worksheet(sample_transformed_data):
    with patch('utils.load.gspread.authorize') as mock_auth:
        mock_sheet = MagicMock()
        mock_auth.return_value.open_by_key.return_value = mock_sheet
        
        # Simulate worksheet not found, requiring a new one
        mock_sheet.worksheet.side_effect = gspread.WorksheetNotFound("Not found")
        
        result = save_to_google_sheets(sample_transformed_data)
        assert result is True
        mock_sheet.add_worksheet.assert_called_once()

def test_save_to_google_sheets_spreadsheet_not_found(sample_transformed_data):
    with patch('utils.load.gspread.authorize') as mock_auth:
        mock_auth.return_value.open_by_key.side_effect = gspread.SpreadsheetNotFound("Not found")
        
        result = save_to_google_sheets(sample_transformed_data)
        assert result is False

def test_save_to_postgresql(sample_transformed_data):
    with patch('utils.load.create_engine') as mock_engine:
        mock_conn = MagicMock()
        mock_engine.return_value.begin.return_value.__enter__.return_value = mock_conn
        
        result = save_to_postgresql(sample_transformed_data)
        assert result is True

def test_save_to_postgresql_with_clear_table(sample_transformed_data):
    with patch('utils.load.create_engine') as mock_engine:
        mock_conn = MagicMock()
        mock_engine.return_value.begin.return_value.__enter__.return_value = mock_conn
        
        result = save_to_postgresql(sample_transformed_data, clear_table=True)
        assert result is True
        # Verify the delete query was executed
        assert mock_conn.execute.call_count >= 1

def test_save_to_postgresql_no_database_url(sample_transformed_data):
    with patch('utils.load.DATABASE_URL', ''):
        result = save_to_postgresql(sample_transformed_data)
        assert result is False

def test_save_to_postgresql_exception(sample_transformed_data):
    with patch('utils.load.create_engine') as mock_engine:
        mock_engine.side_effect = SQLAlchemyError("Database error")
        
        result = save_to_postgresql(sample_transformed_data)
        assert result is False

def test_load_data(sample_transformed_data):
    with patch('utils.load.save_to_csv', return_value=True) as mock_csv, \
         patch('utils.load.save_to_google_sheets', return_value=True) as mock_gs, \
         patch('utils.load.save_to_postgresql', return_value=True) as mock_pg:
        
        result = load_data(sample_transformed_data)
        assert result is True
        mock_csv.assert_called_once()
        mock_gs.assert_called_once()
        mock_pg.assert_called_once()

def test_load_data_empty_dataframe():
    empty_df = pd.DataFrame()
    result = load_data(empty_df)
    assert result is False

def test_load_data_with_clear_table(sample_transformed_data):
    with patch('utils.load.save_to_csv', return_value=True), \
         patch('utils.load.save_to_google_sheets', return_value=True), \
         patch('utils.load.save_to_postgresql', return_value=True) as mock_pg:
        
        result = load_data(sample_transformed_data, clear_table=True)
        assert result is True
        mock_pg.assert_called_once_with(sample_transformed_data, clear_table=True)

def test_load_data_csv_failure(sample_transformed_data):
    with patch('utils.load.save_to_csv', return_value=False), \
         patch('utils.load.save_to_google_sheets', return_value=True), \
         patch('utils.load.save_to_postgresql', return_value=True):
        
        result = load_data(sample_transformed_data)
        assert result is False

def test_load_data_google_sheets_failure(sample_transformed_data):
    with patch('utils.load.save_to_csv', return_value=True), \
         patch('utils.load.save_to_google_sheets', return_value=False), \
         patch('utils.load.save_to_postgresql', return_value=True):
        
        result = load_data(sample_transformed_data)
        assert result is False

def test_load_data_postgresql_failure(sample_transformed_data):
    with patch('utils.load.save_to_csv', return_value=True), \
         patch('utils.load.save_to_google_sheets', return_value=True), \
         patch('utils.load.save_to_postgresql', return_value=False):
        
        result = load_data(sample_transformed_data)
        assert result is False

def test_load_data_exception(sample_transformed_data):
    with patch('utils.load.save_to_csv', side_effect=Exception("Unexpected error")):
        result = load_data(sample_transformed_data)
        assert result is False