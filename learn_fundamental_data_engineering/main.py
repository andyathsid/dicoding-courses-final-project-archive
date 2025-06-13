import pandas as pd
import time
import os
import argparse
from colorama import init, Fore

# Initialize colorama for cross-platform color support
init()

# Import ETL modules
from utils.extract import extract_data
from utils.transform import transform_data
from utils.load import save_to_csv, save_to_google_sheets, save_to_postgresql
from utils.helpers import generate_summary, setup_logging, print_colored, print_column_dtypes

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="ETL Pipeline for Fashion Studio products")
    
    # General options
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--output", "-o", type=str, default="products.csv", help="Output CSV filename")
    parser.add_argument("--log", type=str, default="etl_pipeline.log", help="Log filename")
    
    # ETL process control
    parser.add_argument("--extract-only", action="store_true", help="Run only the extract phase")
    parser.add_argument("--transform-only", action="store_true", help="Run only the transform phase (requires input file)")
    parser.add_argument("--load-only", action="store_true", help="Run only the load phase (requires input file)")
    parser.add_argument("--input", "-i", type=str, help="Input CSV file for transform/load phases")
    
    # Extract options
    parser.add_argument("--max-pages", type=int, default=50, help="Maximum number of pages to scrape")
    parser.add_argument("--max-products", type=int, default=1000, help="Maximum number of products to extract")
    
    # Load options
    parser.add_argument("--no-csv", action="store_true", help="Skip saving to CSV")
    parser.add_argument("--no-sheets", action="store_true", help="Skip saving to Google Sheets")
    parser.add_argument("--no-db", action="store_true", help="Skip saving to database")
    parser.add_argument("--clear-db", action="store_true", help="Clear database table before loading")
    parser.add_argument("--show-dtypes", action="store_true", help="Show datatypes of columns after transformation")
    
    return parser.parse_args()



def run_etl_pipeline(args):
    """
    Main function to run the complete ETL pipeline with options from command line arguments
    """
    logger = setup_logging(args.log, args.verbose)
    
    start_time = time.time()
    print_colored("=" * 60, Fore.CYAN)
    print_colored(f"Starting ETL Pipeline for Fashion Studio products", Fore.CYAN)
    print_colored("=" * 60, Fore.CYAN)
    
    extracted_data = pd.DataFrame()
    transformed_data = pd.DataFrame()
    
    try:
        # EXTRACT phase
        if not args.transform_only and not args.load_only:
            print_colored("\n[1/3] Starting Extract phase...", Fore.BLUE)
            extract_start = time.time()
            
            extracted_data = extract_data(max_pages=args.max_pages, max_products=args.max_products)
            
            if extracted_data.empty:
                print_colored("Extract phase produced no data. Stopping pipeline.", is_success=False)
                return
                
            extract_time = time.time() - extract_start
            print_colored(f"Extract phase completed in {extract_time:.2f} seconds.", is_success=True)
            print_colored(f"Extracted {len(extracted_data)} products.", is_success=True)
            
            if args.verbose or args.show_dtypes:
                print_colored("\nExtracted Data Column Types:", Fore.YELLOW)
                print_column_dtypes(extracted_data)
            
            # Save raw data if extract_only is specified
            if args.extract_only:
                raw_filename = f"raw_{args.output}"
                if save_to_csv(extracted_data, raw_filename):
                    print_colored(f"Raw data saved to {raw_filename}", is_success=True)
                return
        
        # Load input file if transform_only or load_only is specified
        if (args.transform_only or args.load_only) and args.input:
            if not os.path.exists(args.input):
                print_colored(f"Input file {args.input} does not exist!", is_success=False)
                return
                
            print_colored(f"Loading data from {args.input}...", Fore.BLUE)
            try:
                if args.load_only:
                    transformed_data = pd.read_csv(args.input)
                else:
                    extracted_data = pd.read_csv(args.input)
                print_colored(f"Successfully loaded {args.input}", is_success=True)
            except Exception as e:
                print_colored(f"Failed to load {args.input}: {str(e)}", is_success=False)
                return
        
        # TRANSFORM phase
        if not args.load_only:
            print_colored("\n[2/3] Starting Transform phase...", Fore.BLUE)
            transform_start = time.time()
            
            transformed_data = transform_data(extracted_data)
            
            if transformed_data.empty:
                print_colored("Transform phase produced no data. Stopping pipeline.", is_success=False)
                return
                
            transform_time = time.time() - transform_start
            print_colored(f"Transform phase completed in {transform_time:.2f} seconds.", is_success=True)
            print_colored(f"Transformed data contains {len(transformed_data)} products after cleaning.", is_success=True)
            
            # Always show datatypes after transformation when requested
            if args.show_dtypes:
                print_colored("\nTransformed Data Column Types:", Fore.YELLOW)
                print_column_dtypes(transformed_data)
            
            # Save transformed data if transform_only is specified
            if args.transform_only:  
                if save_to_csv(transformed_data, args.output):
                    print_colored(f"Transformed data saved to {args.output}", is_success=True)
                return  
        
        # LOAD phase
        if not args.extract_only and not args.transform_only:
            print_colored("\n[3/3] Starting Load phase...", Fore.BLUE)
            load_start = time.time()
            
            load_results = {}
            
            # Selective loading based on arguments
            if not args.no_csv:
                load_results["CSV"] = save_to_csv(transformed_data, args.output)
                print_colored(f"Save to CSV: {'Success' if load_results['CSV'] else 'Failed'}", 
                             is_success=load_results['CSV'])
                
            if not args.no_sheets:
                load_results["Google Sheets"] = save_to_google_sheets(transformed_data)
                print_colored(f"Save to Google Sheets: {'Success' if load_results['Google Sheets'] else 'Failed'}", 
                             is_success=load_results['Google Sheets'])
                
            if not args.no_db:
                load_results["PostgreSQL"] = save_to_postgresql(transformed_data, clear_table=args.clear_db)
                print_colored(f"Save to PostgreSQL: {'Success' if load_results['PostgreSQL'] else 'Failed'}", 
                             is_success=load_results['PostgreSQL'])
            
            load_success = all(result for result in load_results.values()) if load_results else False
            load_time = time.time() - load_start
            
            if load_success:
                print_colored(f"Load phase completed in {load_time:.2f} seconds.", is_success=True)
            else:
                print_colored(f"Load phase completed with errors in {load_time:.2f} seconds.", is_success=False)
        
        summary = generate_summary(transformed_data)

        total_time = time.time() - start_time
        print_colored("\nSummary:", Fore.CYAN)
        print_colored("=" * 60, Fore.CYAN)
        print_colored(f"Total ETL process completed in {total_time:.2f} seconds.", is_success=True)
        print(summary)
        
    except Exception as e:
        print_colored(f"ETL pipeline failed: {str(e)}", is_success=False)
        logger.exception("Detailed error information:")
        
if __name__ == "__main__":
    args = parse_arguments()
    run_etl_pipeline(args)