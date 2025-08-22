import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from chekml.MetaheuristicOptimization.CIntegration_cffi1.wrapper import Wrapper  # Your specified import
# Assume MhF.py is in your working directory or path
from MhF import MetaheuristicFeaturizer  # Or copy the function code into your script

from sklearn.datasets import make_regression

# Generate sample data (adjust as needed)
X, y = make_regression(n_samples=200, n_features=10, n_informative=5, noise=0.1, random_state=42)
df1 = pd.DataFrame(X[:, :6], columns=[f'feature{i+1}' for i in range(6)])
df1['target'] = y[:100]  # Split for multiple DFs
df2 = pd.DataFrame(X[:, 4:], columns=[f'feature{i+5}' for i in range(6)])
df2['target'] = y[100:]  # Overlapping features

dataframes = [df1, df2]

custom_scorer = make_scorer(mean_squared_error, greater_is_better=False)  # Equivalent to 'neg_mean_squared_error'

linear_model = LinearRegression()

selected_dfs_linear = MetaheuristicFeaturizer(
    dataframes=dataframes,
    top_n=3,  # Select top 3 features per method
    problem_type="regression",
    model=linear_model,
    scorer=custom_scorer,  # Or None for default
    wrapper_method="DISO",  # Or any method supported by your Wrapper
    wrapper_population_size=30,  # Customize population size
    wrapper_max_iter=100,  # Customize iterations
    # Additional Wrapper kwargs if needed, e.g., verbose=1
)

# View results (dict of DataFrames keyed by method, e.g., 'RFE', 'Metaheuristic')
for method, df in selected_dfs_linear.items():
    print(f"{method} selected features:\n{df.head()}\n")

