import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from chekml.MetaheuristicOptimization.CIntegration_cffi1.wrapper import Wrapper
from MhF import MetaheuristicFeaturizer
import torch
import torch.nn as nn

# Define PyTorch neural network module for regression
class SimpleNN(nn.Module):
    def __init__(self, input_dim=10, hidden_dim1=50, hidden_dim2=30, output_dim=1):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.output(x)
        return x

# Generate sample data
X, y = make_regression(n_samples=200, n_features=10, n_informative=5, noise=0.1, random_state=42)

# Create two DataFrames with overlapping features but same target
df1 = pd.DataFrame(X[:, :6], columns=[f'feature{i+1}' for i in range(6)])
df1['target'] = y
df2 = pd.DataFrame(X[:, 4:], columns=[f'feature{i+5}' for i in range(6)])
df2['target'] = y
dataframes = [df1, df2]

# Verify shapes and check for NaNs
print(f"df1 shape: {df1.shape}")
print(f"df2 shape: {df2.shape}")
print(f"NaNs in df1: {df1.isna().sum().sum()}")
print(f"NaNs in df2: {df2.isna().sum().sum()}")

# Custom scorer
custom_scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Define models
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=5, random_state=42),
    "MLPRegressor": MLPRegressor(
        hidden_layer_sizes=(50, 30),
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=200,
        tol=1e-3,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    ),
    "SimpleNN": SimpleNN(input_dim=10)  # PyTorch model
}

# Run MetaheuristicFeaturizer for each model
for model_name, model in models.items():
    print(f"\nRunning MetaheuristicFeaturizer with {model_name}...\n")
    selected_dfs = MetaheuristicFeaturizer(
        dataframes=dataframes,
        top_n=3,
        problem_type="regression",
        model=model,
        scorer=custom_scorer,
        wrapper_class=Wrapper,
        wrapper_method="DISO",
        wrapper_population_size=10,
        wrapper_max_iter=20,
    )
    
    # Print results
    for method, df in selected_dfs.items():
        print(f"{model_name} - {method} selected features (shape: {df.shape}):\n{df.head()}\n")
