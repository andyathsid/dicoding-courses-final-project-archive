# ETL Pipeline Project: Submission Pemda

## Overview
This project implements an ETL (Extract, Transform, Load) pipeline to scrape, clean, and store fashion product data from the website [Fashion Studio Dicoding](https://fashion-studio.dicoding.dev/). The pipeline extracts data from multiple pages, transforms it to ensure quality, and loads it into various formats including CSV, Google Sheets, and PostgreSQL.

## Project Structure
```
├── tests
│   ├── test_extract.py
│   ├── test_transform.py
│   ├── test_load.py
│   └── conftest.py
├── utils
│   ├── extract.py
│   ├── transform.py
│   ├── load.py
│   └── helpers.py
├── main.py
├── config.py
├── requirements.txt
├── submission.txt
├── products.csv
├── google-sheets-api.json
├── .env.example
└── README.md
```

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd submission-pemda
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables by copying `.env.example` to `.env` and filling in the necessary values.

## Usage
To run the ETL pipeline, execute the following command:
```
python main.py
```
This will initiate the extraction of data from the website, transform it according to the specified criteria, and load the cleaned data into the output formats.

### You can also use various command line options:
```bash
# Run with verbose logging
python main.py --verbose

# Run only the extract phase
python main.py --extract-only

# Limit the number of pages/products to scrape
python main.py --max-pages 10 --max-products 100

# Run with a custom output filename
python main.py --output custom_output.csv

```

Alternatively, you can use the provided shell script:
```bash
./run_etl.sh
```

## Testing
To run the tests using pytest, ensure you are in the project directory and execute:
```bash
pytest
```

This will discover and run all the test cases defined in the tests directory.

### Testing with Coverage
To run tests with coverage reporting, use:
```bash
pytest --cov=utils
```

## Data Processing Details
- **Extraction**: Data is scraped from 50 pages of the website, collecting information such as Title, Price, Rating, Colors, Size, and Gender.
- **Transformation**: The data is cleaned by:
  - Converting prices from USD to IDR (using a conversion rate of Rp16,000).
  - Removing duplicates and null values.
  - Ensuring valid data formats for all fields.
  - Adding a timestamp for the extraction process.
- **Loading**: The cleaned data is saved in CSV format, uploaded to Google Sheets, and stored in a PostgreSQL database.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.