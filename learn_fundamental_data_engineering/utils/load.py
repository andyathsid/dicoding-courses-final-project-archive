import pandas as pd
import os
import gspread
from sqlalchemy import create_engine, text
from google.oauth2.service_account import Credentials
import sys

sys.path.append('.')
from config import DATABASE_URL, GOOGLE_SHEETS_CREDS_FILE, GOOGLE_SHEETS_ID, SUPABASE_URL, SUPABASE_KEY, SUPABASE_TABLE
from utils.helpers import setup_logging

# Get logger
logger = setup_logging()
 
def save_to_csv(df: pd.DataFrame, filename: str = "products.csv") -> bool:
    """
    Save data to a CSV file, creating directory if it doesn't exist
    
    Args:
        df: DataFrame to save
        filename: Name of the output file
        
    Returns:
        Boolean indicating success or failure
    """
    try:
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            logger.info(f"Creating directory: {directory}")
            os.makedirs(directory)
            
        df.to_csv(filename, index=False)
        logger.info(f"Successfully saved {len(df)} records to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving data to CSV: {e}")
        return False

def save_to_google_sheets(df: pd.DataFrame, spreadsheet_id: str = GOOGLE_SHEETS_ID) -> bool:
    """
    Save data to Google Sheets
    
    Args:
        df: DataFrame to save
        spreadsheet_id: Google Sheets document ID
        
    Returns:
        Boolean indicating success or failure
    """
    try:
        if not os.path.exists(GOOGLE_SHEETS_CREDS_FILE):
            logger.warning(f"Google Sheets credentials file not found: {GOOGLE_SHEETS_CREDS_FILE}")
            return False
            
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        
        credentials = Credentials.from_service_account_file(
            GOOGLE_SHEETS_CREDS_FILE, 
            scopes=scopes
        )
        
        gc = gspread.authorize(credentials)
        
        df_copy = df.copy()
        
        for col in df_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].astype(str)
        
        try:
            spreadsheet = gc.open_by_key(spreadsheet_id)
            
            try:
                worksheet = spreadsheet.worksheet('Fashion Products')
                worksheet.clear()
            except gspread.WorksheetNotFound:
                worksheet = spreadsheet.add_worksheet('Fashion Products', rows=len(df_copy)+1, cols=len(df_copy.columns))
        
        except gspread.SpreadsheetNotFound:
            logger.error(f"Spreadsheet with ID {spreadsheet_id} not found")
            return False
            
        data_to_upload = [df_copy.columns.tolist()] + df_copy.values.tolist()
        
        worksheet.update(data_to_upload, value_input_option='USER_ENTERED')
        
        logger.info(f"Successfully saved {len(df_copy)} records to Google Sheets")
        return True
        
    except Exception as e:
        logger.error(f"Error saving to Google Sheets: {e}")
        return False

def save_to_postgresql(df: pd.DataFrame, table_name: str = "fashion_products", clear_table: bool = False) -> bool:
    """
    Save data to PostgreSQL database using SQLAlchemy with column name case correction
    
    Args:
        df: DataFrame to save
        table_name: Name of the database table
        clear_table: Whether to clear existing data before inserting new data
        
    Returns:
        Boolean indicating success or failure
    """
    try:
        if not DATABASE_URL:
            logger.error("DATABASE_URL environment variable not found")
            return False
            
        df_copy = df.copy()
        
        df_copy.columns = [col.lower() for col in df_copy.columns]
        
        if 'timestamp' in df_copy.columns and not df_copy.empty:
            df_copy['timestamp'] = df_copy['timestamp'].apply(
                lambda x: x.isoformat() if hasattr(x, 'isoformat') else str(x)
            )
        
        # Create a proper SQLAlchemy engine
        engine = create_engine(DATABASE_URL)
        
        # Create the table if it doesn't exist
        with engine.begin() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    title VARCHAR(255) NOT NULL,
                    price DECIMAL(10, 2),
                    rating DECIMAL(3, 1),
                    colors INTEGER,
                    size VARCHAR(10),
                    gender VARCHAR(10),
                    timestamp TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Create useful indexes if they don't exist
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_title ON {table_name}(title)
            """))
            
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_gender ON {table_name}(gender)
            """))
            
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_size ON {table_name}(size)
            """))
            
            logger.info(f"Ensured table {table_name} exists with proper schema")
        
        # Use SQLAlchemy to create and manage connection
        temp_table = f"{table_name}_temp"
        
        with engine.begin() as conn:
            df_copy.to_sql(temp_table, con=conn, if_exists='replace', index=False)
            
            if clear_table:
                logger.info(f"Deleting all existing data from table {table_name}")
                conn.execute(text(f"DELETE FROM {table_name}"))
            else:
                titles = df_copy['title'].unique().tolist()
                
                if titles:
                    titles_str = "', '".join(titles)
                    delete_query = text(f"DELETE FROM {table_name} WHERE title IN ('{titles_str}')")
                    logger.info(f"Deleting only records with matching titles from table {table_name}")
                    conn.execute(delete_query)
            
            # Insert from temp table to main table
            conn.execute(text(f"""
                INSERT INTO {table_name} (title, price, rating, colors, size, gender, timestamp)
                SELECT t.title, t.price, t.rating, t.colors, t.size, t.gender, 
                       t.timestamp::timestamp with time zone
                FROM {temp_table} t
                LEFT JOIN {table_name} f ON t.title = f.title
                WHERE f.title IS NULL
            """))
            
            # Drop the temporary table
            conn.execute(text(f"DROP TABLE {temp_table}"))
            
        logger.info(f"Successfully saved {len(df)} records to PostgreSQL table '{table_name}'")
        return True
        
    except Exception as e:
        logger.error(f"Error saving to PostgreSQL with SQLAlchemy: {e}")
        return False

def load_data(df: pd.DataFrame, clear_table: bool = False) -> bool:
    """
    Main function to load transformed data into storage systems
    
    Args:
        df: DataFrame with transformed data
        
    Returns:
        Boolean indicating overall success or failure
    """
    if df.empty:
        logger.error("No data to load. Empty DataFrame provided.")
        return False
    
    logger.info("Starting data loading process")
    
    try:
        # Save to CSV 
        csv_result = save_to_csv(df)
        
        # Save to Google Sheets 
        gs_result = save_to_google_sheets(df)
        
        # Save to PostgreSQL with Supabase
        if clear_table:
            pg_result = save_to_postgresql(df, clear_table=clear_table)
        else:
            pg_result = save_to_postgresql(df)
        
        # Check if required storage methods succeeded
        if not csv_result:
            logger.error("Failed to save data to CSV")
            return False
        
        if not gs_result:
            logger.error("Failed to save data to Google Sheets")
            return False
        
        if not pg_result:
            logger.error("Failed to save data to PostgreSQL")
            return False
            
        logger.info("Data loading completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return False