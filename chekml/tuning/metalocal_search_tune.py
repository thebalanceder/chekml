import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from Metegrey_wolf_optimizer import GreyWolfOptimizer

# Step 1: Extract Hyperparameters
def extract_hyperparameter_schema(model):
    """Extract hyperparameter schema for optimization."""
    categorical_options = {
        'copy_X': [True, False],
        'fit_intercept': [True, False],
        'positive': [True, False]
    }
    
    schema = {}
    params = model.get_params()
    
    for param, value in params.items():
        if param in categorical_options:
            schema[param] = categorical_options[param]
        else:
            schema[param] = [value]
    
    return schema

# Instantiate model and extract hyperparameters
model = LinearRegression()
hyperparams = extract_hyperparameter_schema(model)
param_keys = list(hyperparams.keys())  # Store parameter names
print("Extracted Hyperparameters:", hyperparams)

# Step 2: Define Decode Solution Function (Outside GWO)
def decode_solution(wolf):
    """Convert numerical values back to categorical values for the model."""
    decoded_solution = []
    for i, param in enumerate(param_keys):
        if isinstance(hyperparams[param][0], bool):  # Convert float to strictly True/False
            decoded_solution.append(bool(int(round(wolf[i]))))  # ✅ FIXED HERE!
        else:  # Categorical
            decoded_solution.append(hyperparams[param][int(round(wolf[i]))])
    return decoded_solution

# Step 3: Define Objective Function
def objective_function(wolf):
    """ Trains a Linear Regression model and returns MSE as the loss. """
    # ✅ Apply decode_solution() here before using parameters
    solution = decode_solution(wolf)  
    param_values = dict(zip(param_keys, solution))
    
    # Generate synthetic dataset
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression(**param_values)
    model.fit(X_train, y_train)
    
    # Compute MSE
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return mse  # Lower is better

# Step 4: Bounds Preparation for GWO
bounds = []
for param, values in hyperparams.items():
    if isinstance(values[0], bool):  # Convert boolean to numeric (0=False, 1=True)
        bounds.append((0, 1))
    else:  # Categorical
        bounds.append((0, len(values) - 1))

# Step 6: Run GWO to Find Best Hyperparameters
gwo = GreyWolfOptimizer(objective_function, dim=len(param_keys), bounds=bounds, population_size=10, max_iter=20)
best_wolf, best_mse, _ = gwo.optimize()

# Convert best solution to a dictionary
best_hyperparams_dict = dict(zip(param_keys, decode_solution(best_wolf)))
print("Best Hyperparameters:", best_hyperparams_dict)
print("Best MSE:", best_mse)

