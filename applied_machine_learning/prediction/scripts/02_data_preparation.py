# %% [markdown]
# This notebook focuses on preparing the cleaned mining dataset for modeling by handling missing values, removing outliers, engineering new features, and performing data splitting and scaling.

# %%
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


# %% [markdown]
# ## Loading and Exploring the Cleaned Dataset

# %% [markdown]
# Let's load the cleaned data from the previous step.

# %%
# Load the cleaned data
df = pd.read_csv('../data/interim/silica_data_cleaned.csv')

# Display basic information
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
df.head()

# %%
# Check data types and missing values
df.info()

# %%
# Verify datetime column is in correct format
df['datetime'] = pd.to_datetime(df['datetime'])

# %%
target_column = '%_silica_concentrate'

# %%

print(f"Target variable: {target_column}")
print(f"Target variable statistics: \n{df[target_column].describe()}")
print("\n")

# %% [markdown]
# ## Missing Values Handling

# %% [markdown]
# This step addresses missing data through two strategic approaches:
# - Rows with missing target values are removed as they cannot contribute to supervised learning
# - Missing values in feature columns are imputed with median values to preserve as much data as possible

# %%
# Remove rows with missing values in the target column
print(f"Rows before cleaning target: {df.shape[0]}")
df = df.dropna(subset=[target_column])
print(f"Rows after cleaning target: {df.shape[0]}")

# For remaining columns, impute missing values with median
numeric_columns = df.select_dtypes(include=['number']).columns
for column in numeric_columns:
    if df[column].isnull().sum() > 0:
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)
        print(f"Imputed {df[column].isnull().sum()} values in {column} with median: {median_value}")

# %% [markdown]
# ## Outliers Handling

# %% [markdown]
# Statistical outliers in the target variable are identified and removed using the Interquartile Range (IQR) method:
# - Calculate quartiles (Q1, Q3) and compute IQR = Q3-Q1
# - Define outlier boundaries at Q1-1.5×IQR and Q3+1.5×IQR
# - Visualize the distribution with boundary markers
# - Filter out observations that fall outside these boundaries

# %%
# Check for outliers in target variable using IQR method
Q1 = df[target_column].quantile(0.25)
Q3 = df[target_column].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df[target_column] < lower_bound) | (df[target_column] > upper_bound)]
print(f"Number of outliers in target variable: {len(outliers)}")
print(f"Outlier boundaries: Lower = {lower_bound}, Upper = {upper_bound}")

# Visualize target variable with outlier boundaries
plt.figure(figsize=(10, 6))
sns.histplot(df[target_column], bins=30, kde=True)
plt.axvline(lower_bound, color='red', linestyle='--', label='Lower Bound')
plt.axvline(upper_bound, color='red', linestyle='--', label='Upper Bound')
plt.title('Target Variable Distribution with Outlier Boundaries')
plt.xlabel(target_column)
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Remove outliers from target variable
df = df[(df[target_column] >= lower_bound) & (df[target_column] <= upper_bound)]
print(f"Dataset shape after removing outliers: {df.shape}")

# %% [markdown]
# ## Feature Engineering 

# %% [markdown]
# This critical step creates new informative features that can help uncover complex relationships:
# - Identify features with strong correlations to the target variable
# - Generate ratio features to capture proportional relationships between variables
# - Create interaction features (products) to model combined effects
# - Add polynomial terms to capture non-linear relationships
# - Compute statistical aggregations (mean, std, range) for related sensor groups

# %%
# Calculate correlations with target variable
correlations = df.select_dtypes(include=['number']).corr()[target_column].sort_values(ascending=False)
print("Top correlations with target variable:")
print(correlations.head(10))
print("\nBottom correlations with target variable:")
print(correlations.tail(10))
print("\n")

# Plot correlation heatmap
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', 
            linewidths=0.5, vmax=1, vmin=-1)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Identify top correlated features (positive and negative)
high_corr_features = abs(correlations).sort_values(ascending=False).index[1:6]
print(f"Using top correlated features for engineering: {list(high_corr_features)}")

# Visualize relationships between top correlated features and target
top_positive = correlations.head(3).index.tolist()
top_negative = correlations.tail(3).index.tolist()

# Create scatter plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, col in enumerate(top_positive):
    sns.scatterplot(x=df[col], y=df[target_column], ax=axes[0, i])
    axes[0, i].set_title(f'{target_column} vs {col}')

for i, col in enumerate(top_negative):
    sns.scatterplot(x=df[col], y=df[target_column], ax=axes[1, i])
    axes[1, i].set_title(f'{target_column} vs {col}')

plt.tight_layout()
plt.show()

# %%
# Create ratios and interactions between important features
for i, feat1 in enumerate(high_corr_features):
    for feat2 in high_corr_features[i+1:]:
        # Create ratio feature
        ratio_name = f'{feat1}_to_{feat2}_ratio'
        df[ratio_name] = df[feat1] / df[feat2].replace(0, np.nan)
        df[ratio_name] = df[ratio_name].replace([np.inf, -np.inf], np.nan).fillna(df[ratio_name].median())
        
        # Create product feature
        product_name = f'{feat1}_x_{feat2}'
        df[product_name] = df[feat1] * df[feat2]

# Create polynomial features for top correlated variables
for feature in high_corr_features:
    df[f'{feature}_squared'] = df[feature] ** 2

print(f"Dataset shape after feature engineering: {df.shape}")
print(f"New features added: {df.shape[1] - len(numeric_columns)}")

# %%
# Calculate statistical features for air flow columns if they exist
air_flow_cols = [col for col in df.columns if 'air_flow' in col.lower()]
if len(air_flow_cols) > 0:
    df['mean_air_flow'] = df[air_flow_cols].mean(axis=1)
    df['std_air_flow'] = df[air_flow_cols].std(axis=1)
    df['max_min_ratio_air_flow'] = df[air_flow_cols].max(axis=1) / df[air_flow_cols].min(axis=1).replace(0, np.nan)
    df['max_min_ratio_air_flow'] = df['max_min_ratio_air_flow'].replace([np.inf, -np.inf], np.nan).fillna(df['max_min_ratio_air_flow'].median())
    print(f"Created air flow statistical features: mean_air_flow, std_air_flow, max_min_ratio_air_flow")

# Calculate statistical features for level columns if they exist
level_cols = [col for col in df.columns if 'level' in col.lower()]
if len(level_cols) > 0:
    df['mean_level'] = df[level_cols].mean(axis=1)
    df['std_level'] = df[level_cols].std(axis=1)
    df['level_range'] = df[level_cols].max(axis=1) - df[level_cols].min(axis=1)
    print(f"Created level statistical features: mean_level, std_level, level_range")

# %% [markdown]
# ## Data Split Handling

# %% [markdown]
# The data is partitioned to enable proper model training and evaluation:
# - Features (X) are separated from the target variable (y)
# - Non-numeric columns are removed
# - Any remaining missing values are addressed
# - Data is split into training (80%) and testing (20%) sets

# %%
# Define features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Remove any non-numeric columns (like date)
X = X.select_dtypes(include=['number'])

# Final check for any remaining NaN values
X = X.fillna(X.median())

# Split data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# %% [markdown]
# ## Feature Scaling

# %% [markdown]
# This step transforms all features to have zero mean and unit variance using standardization

# %%
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to dataframes for better handling (optional)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# %% [markdown]
# ## Save Processed Data

# %%
import os
import joblib

# Create directory if it doesn't exist
output_dir = '../data/processed'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save processed data
np.save(f'{output_dir}/X_train_scaled.npy', X_train_scaled)
np.save(f'{output_dir}/X_test_scaled.npy', X_test_scaled)
np.save(f'{output_dir}/y_train.npy', y_train.values)
np.save(f'{output_dir}/y_test.npy', y_test.values)

# Save feature names
feature_columns = X_train.columns.tolist()
with open(f'{output_dir}/feature_columns.txt', 'w') as f:
    f.write('\n'.join(feature_columns))

# Save scaler for future use
joblib.dump(scaler, f'{output_dir}/scaler.joblib')

# %%
print("\nData Preparation Summary:")
print("-----------------------")
print(f"Original dataset shape: {df.shape}")
print(f"Features after preprocessing: {X.shape[1]}")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

print("\nFeature Engineering:")
engineered_features = [col for col in X.columns if ('_to_' in col or '_x_' in col or '_squared' in col or
                                                   'mean_' in col or 'std_' in col or '_range' in col)]
print(f"- Created {len(engineered_features)} engineered features")
print("- Types of engineered features:")
print("  - Ratio features between highly correlated variables")
print("  - Interaction (product) features")
print("  - Polynomial features (squared terms)")
print("  - Statistical aggregations (mean, std, range) for similar sensors")

print("\nPrepared data saved to:", output_dir)


