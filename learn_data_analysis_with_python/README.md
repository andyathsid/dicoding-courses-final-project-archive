# Bike Sharing Analysis Dashboard

This is final project as part of Dicoding Data Analysis Course to analyzes bike sharing patterns from the [Capital Bikeshare system in Washington D.C. (2011-2012) ](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), exploring user behavior, weather impacts, and temporal trends.

## Live Demo
Access the live dashboard at: [Bike Sharing Analysis Dashboard](https://andakara-dicoding-final-project-bike-sharing-analysis.streamlit.app)

## Features
1. Peak Period Analysis
   - Hourly rental patterns
   - Daily usage trends
   
2. Weather Impact Analysis
   - Effect of weather conditions on rentals
   - User type sensitivity to weather

3. Monthly Trends
   - Year-over-year growth
   - Seasonal patterns

4. Weather Recovery Analysis
   - Post-bad weather recovery patterns
   - User behavior differences

## Setup Instructions

### Prerequisites
- Python 3.12 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/bike-sharing-analysis.git
   cd bike-sharing-analysis
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv .dicoding-ldawp-final-project-venv
   source .dicoding-ldawp-final-project-venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Dashboard

1. **Start the Streamlit server:**
   ```bash
   cd dashboard
   streamlit run dashboard.py
   ```

2. **Access the dashboard:**
   - Open your browser
   - Visit `http://localhost:8501`

## 📁 Project Structure
```
.
├── dashboard/
│   └── dashboard.py          # Main dashboard application
├── data/
│   └── bike-sharing-dataset/
│       ├── hour.csv         # Hourly rental data
│       └── day.csv          # Daily rental data
├── notebook.ipynb           # Analysis notebook
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```
