import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from feature_selector import FeatureSelector
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create two synthetic DataFrames with overlapping features, identical targets, and no NaNs
def create_two_synthetic_dataframes(n_samples=50):
    # Generate base classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,  # Matches feature1 to feature8
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    # DataFrame 1: All 8 features + target + TARGET
    df1 = pd.DataFrame(X, columns=[f"feature{i+1}" for i in range(8)])
    df1["target"] = y
    df1["TARGET"] = y  # Duplicate target column
    
    # DataFrame 2: First 6 features + target
    df2 = pd.DataFrame(X[:, :6], columns=[f"feature{i+1}" for i in range(6)])
    df2["target"] = y
    
    # Verify no NaNs
    if df1.isna().any().any() or df2.isna().any().any():
        raise ValueError("NaNs found in input DataFrames")
    
    return [df1, df2]

# Generate two DataFrames
dfs = create_two_synthetic_dataframes(n_samples=50)

# Print info about input DataFrames
print("Input DataFrames:")
for i, df in enumerate(dfs):
    print(f"\nDataFrame {i}:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"NaN count: {df.isna().sum().sum()}")
    print(df.head())

# Apply FeatureSelector with save_to_csv=True
try:
    result = FeatureSelector(
        dataframes=dfs,
        top_n=3,  # Select top 3 features
        problem_type="classification",
        save_to_csv=True
    )
    
    # Print results
    print("\nFeature Selection Results:")
    for method, selected_df in result.items():
        print(f"\n{method}:")
        print(f"Shape: {selected_df.shape}")
        print(f"NaN count: {selected_df.isna().sum().sum()}")
        print("Selected Features:", [col for col in selected_df.columns if col != "target"])
        print(selected_df.head())
        
except Exception as e:
    print(f"Error: {e}")
