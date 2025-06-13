# %% [markdown]
# This notebook performs comprehensive exploratory data analysis on the mining dataset to understand data patterns, distributions, and relationships between features.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import probscale
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

# %% [markdown]
# ## Data Loading and Initial Exploration

# %% [markdown]
# We start by loading the mining dataset and examining its structure, dimensions, and basic statistics.

# %%
df = pd.read_csv('../data/raw/silica_data.csv')

# %%
df.shape

# %%
df.info()

# %%
df.head()

# %%
print(df.head())

# %%
df.describe()

# %%
missing_data = df.isnull().sum()
missing_percentage = (missing_data / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing Percentage': missing_percentage
}).sort_values('Missing Percentage', ascending=False)

missing_df[missing_df['Missing Count'] > 0]

# %%
df.duplicated().sum()

# %% [markdown]
# ## Initial Data Preprocessing

# %% [markdown]
# Before analysis, we perform necessary initial preprocessing steps to help with EDA by:
# - Removing duplicate records to prevent bias in our analysis
# - Converting the date column to proper datetime format for time-based analysis
# - Standardizing column names by converting to lowercase and replacing spaces with underscores
# - Converting numeric columns from string format to proper floating point values
# - Identifying feature columns and the target variable for subsequent analysis

# %%
# Drop duplicates
df.drop_duplicates(inplace=True)

# %%
# Parse date column to datetime format
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')

# %%
df.rename(columns={'date': 'datetime'}, inplace=True)

# %%
# Lower columns name and replace space with underscores
df.columns = [clm.replace(' ', '_').lower() for clm in df.columns]

# %%
# Convert numeric columns to float
for column in df.columns[1:]:
    df[column]=df[column].str.replace(',','.').astype(float)

# %%
df.info()

# %%
NUM_FEATURES = df.columns[1:-1]

# %%
NUM_FEATURES

# %%
TARGET = df.columns[-1]
TARGET

# %%
df[NUM_FEATURES].describe().T

# %%
df[TARGET].describe()

# %% [markdown]
# ## Exploratory Data Analysis

# %% [markdown]
# ### Data Distribution Analysis

# %% [markdown]
# We visualize the distribution of all numeric variables using histograms with KDE curves to understand their shapes, central tendencies, and spread. This helps identify potential outliers, skewness, and overall data patterns.

# %%
numeric_columns = df.columns[1:]
n_cols = 3
n_rows = (len(numeric_columns) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes

for i, col in enumerate(numeric_columns):
    if i < len(axes):
        sns.histplot(data=df, x=col, kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
        axes[i].tick_params(axis='x', rotation=45)

# Hide extra subplots
for i in range(len(numeric_columns), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Target Variable Analysis

# %% [markdown]
# We conduct an in-depth analysis of the silica concentration (target variable) using multiple visualization techniques:
# - Histogram to show the overall distribution
# - Boxplot to identify potential outliers and quartile distribution
# - ECDF plot to visualize the cumulative distribution
# - Probability plot to assess normality of the target variable
# - Time series plot to observe temporal patterns in silica concentration
# - Monthly aggregation to detect seasonal patterns or shifts over time

# %%
fig,ax=plt.subplots(4, 1, figsize=(14,10))

sns.histplot(df[TARGET], ax=ax[0])
sns.boxplot(x=df[TARGET], color='lightblue', saturation=0.8, ax=ax[1])
ax[1].axvline(np.percentile(df[TARGET],.1), label='.1%', c='orange', linestyle=':', linewidth=3)
ax[1].axvline(np.percentile(df[TARGET],.5), label='.5%', c='darkblue', linestyle=':', linewidth=3)
ax[1].legend()
sns.ecdfplot(df[TARGET], ax=ax[2])
ax[0].set_xticks(np.arange(0, df[TARGET].max(), .5))
ax[1].set_xticks(np.arange(0, df[TARGET].max(), .5))
ax[2].set_xticks(np.arange(0, df[TARGET].max(), .5))
fig = probscale.probplot(
    df[TARGET],
    ax=ax[3],
    plottype='qq',
    bestfit=True
)
plt.show()
plt.show()

# %%
plt.figure(figsize=(22,10))
plt.plot(df.datetime, df[TARGET])
plt.title(f"{TARGET}")
plt.xlabel("Date")
plt.ylabel(f"{TARGET}")
plt.show()

# %%
print(f"Skew of target: {df[TARGET].skew():.3f}")

# %%
df[TARGET].groupby(df.datetime.dt.to_period('M')).agg(['mean', 'median', 'min', 'max'])

# %% [markdown]
# ### Outliers Analysis

# %% [markdown]
# We perform a detailed analysis of outliers using the IQR method

# %%
def get_distribution_metrics(data, title='', prec=2):
    ''' Calculate distribution metrics without plotting '''
    
    # Quartiles and IQR
    q1, q3 = np.percentile(data.dropna(), [25, 75])
    iqr = q3 - q1
    
    # Outlier boundaries (1.5 * IQR rule)
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Actual whisker values (data points within bounds)
    valid_data = data.dropna()
    whisker_high = valid_data[valid_data <= upper_bound]
    whisker_low = valid_data[valid_data >= lower_bound]
    actual_upper_whisker = np.max(whisker_high) if len(whisker_high) > 0 else upper_bound
    actual_lower_whisker = np.min(whisker_low) if len(whisker_low) > 0 else lower_bound
    
    # Count outliers
    outliers = valid_data[(valid_data < actual_lower_whisker) | (valid_data > actual_upper_whisker)]
    
    metrics = {
        'Variable': title if title else 'Data',
        'Q1 (25th percentile)': round(q1, prec),
        'Q3 (75th percentile)': round(q3, prec),
        'IQR': round(iqr, prec),
        'Lower Bound (Q1 - 1.5*IQR)': round(lower_bound, prec),
        'Upper Bound (Q3 + 1.5*IQR)': round(upper_bound, prec),
        'Actual Lower Whisker': round(actual_lower_whisker, prec),
        'Actual Upper Whisker': round(actual_upper_whisker, prec),
        'Number of Outliers': len(outliers),
        'Outlier Percentage': round((len(outliers) / len(valid_data)) * 100, prec),
    }
    
    return metrics

# %%
target_metrics = get_distribution_metrics(df[TARGET], TARGET)
print(f"\nTarget Variable ({TARGET}) Distribution Summary:")
print("=" * 60)
for key, value in target_metrics.items():
    print(f"{key}: {value}")

# %% [markdown]
# ### Correlation Analysis

# %% [markdown]
# We compute and visualize the correlation matrix between all numeric variables using a heatmap

# %%
corrMatrix = df[df.columns[1:]].corr(method='pearson', min_periods=1)
plt.figure(figsize=(15,10))
mask = np.triu(np.ones_like(corrMatrix, dtype=bool))
ax = sns.heatmap(corrMatrix, mask=mask, annot=True, cbar_kws={"shrink": .8}, cmap='coolwarm')
plt.show()

# %%
# import os
# if not os.path.exists('../data/interim'):
#     os.makedirs('../data/interim')
# df.to_csv('../data/interim/silica_data_cleaned.csv', index=False)


