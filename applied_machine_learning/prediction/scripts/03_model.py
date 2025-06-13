# %% [markdown]
# This notebook develops predictive models for silica concentration in mining processes, comparing baseline algorithms before optimizing the best performer.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, validation_curve, learning_curve
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.terminator import TerminatorCallback
import lightgbm as lgb
import xgboost as xgb
import joblib
import warnings
import time
from tqdm.notebook import tqdm

warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")


# %% [markdown]
# ## Loading Processed Data

# %% [markdown]
# Load the preprocessed data from our data preparation phase

# %%
# Define data directory
data_dir = '../data/processed'

# Load training and testing data
X_train_scaled = np.load(f'{data_dir}/X_train_scaled.npy')
X_test_scaled = np.load(f'{data_dir}/X_test_scaled.npy')
y_train = np.load(f'{data_dir}/y_train.npy')
y_test = np.load(f'{data_dir}/y_test.npy')



# %%
# Load feature column names and scaler
with open(f'{data_dir}/feature_columns.txt', 'r') as f:
    feature_columns = f.read().splitlines()
    
scaler = joblib.load(f'{data_dir}/scaler.joblib')

# %%
# Verify data shapes
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"Number of features: {len(feature_columns)}")

# %% [markdown]
# ## Define Evaluation Metrics

# %% [markdown]
# This step establishes a comprehensive evaluation framework with multiple complementary metrics:
# - R² score measures the proportion of variance explained by the model
# - Root Mean Squared Error (RMSE) quantifies prediction errors with sensitivity to larger deviations
# - Mean Absolute Error (MAE) provides an intuitive average error magnitude
# - Mean Absolute Percentage Error (MAPE) offers scale-independent error measurement

# %%
# Define comprehensive evaluation metrics
def evaluate_model(y_true, y_pred):
    """
    Calculate multiple regression evaluation metrics
    
    Parameters:
    -----------
    y_true : array-like
        Actual target values
    y_pred : array-like
        Predicted target values
        
    Returns:
    --------
    dict
        Dictionary containing multiple evaluation metrics
    """
    # R² score (coefficient of determination)
    r2 = r2_score(y_true, y_pred)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    # Add small epsilon to avoid division by zero
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }

def print_metrics(metrics):
    """Print metrics in a formatted way"""
    print("Model Performance:")
    print(f"  R² Score:   {metrics['r2']:.4f}")
    print(f"  RMSE:       {metrics['rmse']:.4f}")
    print(f"  MAE:        {metrics['mae']:.4f}")
    print(f"  MAPE:       {metrics['mape']:.2f}%")

# %% [markdown]
# ## Baseline Modelling

# %% [markdown]
# This step evaluates multiple regression algorithms to establish performance benchmarks:
# - Linear models: Linear Regression, ElasticNet
# - Tree-based ensembles: Random Forest, LightGBM, XGBoost

# %%
models = {
    'Linear Regression': LinearRegression(),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000),
    'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10, max_features='sqrt', n_jobs=-1, ),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
}


# %%
results = []

print("Evaluating baseline models...")
# Add outer progress bar for models
for name, model in tqdm(models.items(), desc="Models", position=0):
    print(f"\nRunning baseline model: {name}")
    
    # Set verbose for supported models to show native progress
    if name == 'Random Forest':
        model.set_params(verbose=1)
    elif name in ['LightGBM', 'XGBoost']:
        model.set_params(verbose=0)  # Set to non-zero for progress
    
    # Modified cross-validation with progress bar
    cv_scores = []
    start_time = time.time()
    
    # Create cross-validation folds manually to add progress bar
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, val_idx in tqdm(kf.split(X_train_scaled), 
                                   total=5, 
                                   desc="CV Folds", 
                                   position=1,
                                   leave=False):
        X_train_fold, X_val_fold = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        model.fit(X_train_fold, y_train_fold)
        y_val_pred = model.predict(X_val_fold)
        cv_scores.append(r2_score(y_val_fold, y_val_pred))
    
    cv_r2 = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    cv_time = time.time() - start_time
    
    # Train on full training data with progress display
    print(f"Training full model for {name}...")
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    
    # Make predictions and evaluate
    y_pred = model.predict(X_test_scaled)
    metrics = evaluate_model(y_test, y_pred)
    
    # Store results
    results.append({
        'Model': name,
        'CV R²': cv_r2,
        'CV Std': cv_std,
        'Test R²': metrics['r2'],
        'Test RMSE': metrics['rmse'],
        'Test MAE': metrics['mae'],
        'Test MAPE': metrics['mape'],
        'CV Time (s)': cv_time,
        'Train Time (s)': train_time
    })
    
    # Print results
    print(f"Cross-validation R² score: {cv_r2:.4f} ± {cv_std:.4f}")
    print(f"Cross-validation time: {cv_time:.2f} seconds")
    print(f"Training time on full dataset: {train_time:.2f} seconds")
    print_metrics(metrics)

# Convert to DataFrame and sort by test R²
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Test R²', ascending=False).reset_index(drop=True)

# %%
results_df

# %% [markdown]
# ## Optuna Objective Function

# %% [markdown]
# This step defines the optimization target for hyperparameter tuning:
# - Creates a parameterized objective function for XGBoost models
# - Specifies the search space for each hyperparameter
# - Uses cross-validation R² score as the optimization metric

# %%
def objective(trial):
  
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0001, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0001, 1.0, log=True),
        'random_state': 42
    }
    
    # Create XGBoost model with trial parameters
    model = xgb.XGBRegressor(**params)
    
    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=-1)
    
    return cv_scores.mean()

# %% [markdown]
# ## Hyperparameter Optimization with Optuna

# %% [markdown]
# This step systematically explores the hyperparameter space to maximize model performance:
# - Creates an optimization study with Tree-structured Parzen Estimator (TPE) sampler
# - Implements early stopping via a median pruner to avoid wasting computation
# - Executes multiple trials with progress tracking
# - Records the best parameter combination discovered during optimization

# %%
# Create Optuna study
terminator_callback = TerminatorCallback()
study = optuna.create_study(
    direction="maximize",
    study_name="xgboost_optimization",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10),
    sampler=optuna.samplers.TPESampler(seed=42)
)
# Optimize
n_trials = 25

with tqdm(total=n_trials, desc="XGBoost Trials") as pbar:
    def callback(study, trial):
        pbar.update(1)
        pbar.set_postfix({"Best R²": study.best_value})
        terminator_callback 
    
    study.optimize(objective, n_trials=n_trials, callbacks=[callback], timeout=3600)

# %%
print(f"\nBest XGBoost R² score: {study.best_value:.6f}")
print("Best hyperparameters:")
for param, value in study.best_params.items():
    print(f"  {param}: {value}")

# %%
print(f"Number of trials: {len(study.trials)}")
print(f"Best R² score: {study.best_value:.4f}")
print(f"Best parameters: {study.best_params}")

# %% [markdown]
# ## Evaluate the Tuned Best Model

# %% [markdown]
# This step assesses the optimized model's performance:
# - Trains the best and optimized model with the best hyperparameters on the full training set
# - Generates predictions on the held-out test data
# - Calculates comprehensive performance metrics
# - Compares results to the baseline XGBoost model to quantify improvements

# %%
best_xgb = xgb.XGBRegressor(**study.best_params)

# %%
best_xgb.fit(X_train_scaled, y_train)

# %%
y_pred = best_xgb.predict(X_test_scaled)
best_metrics = evaluate_model(y_test, y_pred)

# %%
print_metrics(best_metrics)

# %%
baseline_xgb_metrics = results_df[results_df['Model'] == 'XGBoost'].iloc[0]
print("\nImprovement over baseline:")
print(f"  R² Score:   {best_metrics['r2'] - baseline_xgb_metrics['Test R²']:.4f}")
print(f"  RMSE:       {baseline_xgb_metrics['Test RMSE'] - best_metrics['rmse']:.4f}")
print(f"  MAE:        {baseline_xgb_metrics['Test MAE'] - best_metrics['mae']:.4f}")
print(f"  MAPE:       {baseline_xgb_metrics['Test MAPE'] - best_metrics['mape']:.2f}%")

# %% [markdown]
# ### Learning Curves

# %% [markdown]
# This step visualizes model performance as a function of training set size to provide insights into the model's learning capacity and generalization ability.

# %%
def plot_learning_curve(estimator, X, y, title="Learning Curve", ylim=None, cv=5,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if ylim is not None:
        ax.set_ylim(*ylim)
        
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.set_title(title)
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring="r2")
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    ax.grid()
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    ax.legend(loc="best")
    
    return fig, ax

# %%
plot_learning_curve(
    best_xgb, X_train_scaled, y_train,
    title=f"Learning Curve - Tuned XGBoost (Best R²: {study.best_value:.4f})"
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Feature Importance

# %% [markdown]
# This step identifies which variables most strongly influence silica concentration predictions

# %%
plt.figure(figsize=(12, 8))
xgb_importance = best_xgb.feature_importances_
sorted_idx = np.argsort(xgb_importance)
plt.barh(np.array(feature_columns)[sorted_idx][-20:], xgb_importance[sorted_idx][-20:])
plt.xlabel('Feature Importance')
plt.title('Top 20 Features - XGBoost')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Predictions and Residuals Analysis

# %% [markdown]
# This step examines prediction quality and error patterns

# %%
# Plot predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted - Tuned XGBoost')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot - Tuned XGBoost')
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
# Plot residual distribution
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.title('Distribution of Residuals - Tuned XGBoost')
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Save the Best Model

# %%
model_dir = '../models'
import os
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

joblib.dump(best_xgb, f'{model_dir}/best_xgb_model.joblib')
print(f"\nBest XGBoost model saved to {model_dir}/best_xgb_model.joblib")

# Save hyperparameters for reference
import json
with open(f'{model_dir}/best_xgb_params.json', 'w') as f:
    json.dump(study.best_params, f, indent=4)


