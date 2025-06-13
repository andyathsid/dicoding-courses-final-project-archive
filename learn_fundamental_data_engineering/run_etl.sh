#!/bin/bash
# ETL Pipeline runner script

# Default values
VERBOSE=false
MAX_PAGES=50
MAX_PRODUCTS=1000
OUTPUT="products.csv"
EXTRACT_ONLY=false
TRANSFORM_ONLY=false
LOAD_ONLY=false
INPUT=""
SHOW_DTYPES=false

# Print usage information
function show_help {
    echo "Fashion Studio ETL Pipeline"
    echo "Usage: ./run_etl.sh [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -v, --verbose           Enable verbose mode"
    echo "  -p, --max-pages NUM     Maximum number of pages to scrape (default: 50)"
    echo "  -n, --max-products NUM  Maximum number of products to extract (default: 1000)"
    echo "  -o, --output FILE       Output CSV filename (default: products.csv)"
    echo "  -e, --extract-only      Run only the extract phase"
    echo "  -t, --transform-only    Run only the transform phase"
    echo "  -l, --load-only         Run only the load phase"
    echo "  -i, --input FILE        Input file for transform/load phases"
    echo "  --no-csv                Skip saving to CSV"
    echo "  --no-sheets             Skip saving to Google Sheets"
    echo "  --no-db                 Skip saving to database"
    echo "  --clear-db              Clear database table before loading"
    echo "  --show-dtypes           Show datatypes of columns after transformation"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -p|--max-pages)
            MAX_PAGES=$2
            shift 2
            ;;
        -n|--max-products)
            MAX_PRODUCTS=$2
            shift 2
            ;;
        -o|--output)
            OUTPUT=$2
            shift 2
            ;;
        -e|--extract-only)
            EXTRACT_ONLY=true
            shift
            ;;
        -t|--transform-only)
            TRANSFORM_ONLY=true
            shift
            ;;
        -l|--load-only)
            LOAD_ONLY=true
            shift
            ;;
        -i|--input)
            INPUT=$2
            shift 2
            ;;
        --no-csv)
            NO_CSV="--no-csv"
            shift
            ;;
        --no-sheets)
            NO_SHEETS="--no-sheets"
            shift
            ;;
        --no-db)
            NO_DB="--no-db"
            shift
            ;;
        --clear-db)
            CLEAR_DB="--clear-db"
            shift
            ;;
        --show-dtypes)
            SHOW_DTYPES=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Build command with options
CMD="python main.py"

if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

if [ "$EXTRACT_ONLY" = true ]; then
    CMD="$CMD --extract-only"
fi

if [ "$TRANSFORM_ONLY" = true ]; then
    CMD="$CMD --transform-only"
fi

if [ "$LOAD_ONLY" = true ]; then
    CMD="$CMD --load-only"
fi

if [ -n "$INPUT" ]; then
    CMD="$CMD --input $INPUT"
fi

if [ -n "$NO_CSV" ]; then
    CMD="$CMD $NO_CSV"
fi

if [ -n "$NO_SHEETS" ]; then
    CMD="$CMD $NO_SHEETS"
fi

if [ -n "$NO_DB" ]; then
    CMD="$CMD $NO_DB"
fi

if [ -n "$CLEAR_DB" ]; then
    CMD="$CMD $CLEAR_DB"
fi

if [ "$SHOW_DTYPES" = true ]; then
    CMD="$CMD --show-dtypes"
fi

CMD="$CMD --max-pages $MAX_PAGES --max-products $MAX_PRODUCTS --output $OUTPUT"

# Execute command
echo "Running: $CMD"
eval "$CMD"