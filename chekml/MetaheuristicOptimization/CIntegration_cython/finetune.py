import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression
from dujiangyan_optimizer import DujiangyanIrrigationOptimizer

# Step 1: Extract Hyperparameters
def extract_hyperparameter_schema(model):
    """Extract hyperparameter schema for optimization."""
    categorical_options = {
        'copy_X': [True, False],
        'fit_intercept': [True, False],
        'positive': [True, False],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # For SVM
        'splitter': ['best', 'random'],  # For DecisionTree
    }
    
    schema = {}
    params = model.get_params()
    
    for param, value in params.items():
        if param in categorical_options:
            schema[param] = categorical_options[param]
        else:
            schema[param] = [value]
    
    return schema

# Define models
models = {
    "LinearRegression": LinearRegression(),
    "SVM": SVR(),
    "DecisionTree": DecisionTreeRegressor(random_state=42)
}

# Step 2: Define Decode Solution Function
def decode_solution(wolf, model):
    """Convert numerical values back to categorical values for the model."""
    hyperparams = extract_hyperparameter_schema(model)
    param_keys = list(hyperparams.keys())
    decoded_solution = {}
    
    for i, param in enumerate(param_keys):
        if isinstance(hyperparams[param][0], bool):
            decoded_solution[param] = bool(int(round(min(max(wolf[i], 0), 1))))
        else:
            decoded_solution[param] = hyperparams[param][int(round(min(max(wolf[i], 0), len(hyperparams[param]) - 1)))]
    
    return decoded_solution

# Step 3: Define Objective Function
def objective_function(wolf, model):
    """Trains a model and returns MSE as the loss."""
    solution = decode_solution(wolf, model)
    
    # Generate synthetic dataset
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model.set_params(**solution)
    model.fit(X_train, y_train)
    
    # Compute MSE
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return mse

# Step 4: Optimize for Each Model
optimized_results = {}
initial_performance = {}

start_total = time.time()  # Start total execution time

for model_name, model in models.items():
    start_model = time.time()  # Start time for this model
    
    hyperparams = extract_hyperparameter_schema(model)
    param_keys = list(hyperparams.keys())
    
    bounds = [(0, 1) if isinstance(values[0], bool) else (0, len(values) - 1) for param, values in hyperparams.items()]
    
    # Evaluate initial model performance before optimization
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    initial_mse = mean_squared_error(y_test, y_pred)
    initial_r2 = r2_score(y_test, y_pred)
    initial_performance[model_name] = {"MSE": initial_mse, "R² Score": initial_r2}
    
    diso = DujiangyanIrrigationOptimizer(
        objective_function=lambda wolf: objective_function(wolf, model),
        dim=len(param_keys),
        bounds=bounds,
        population_size=10,
        max_iter=20
    )
    
    best_wolf, best_mse, _ = diso.optimize()
    best_hyperparams_dict = decode_solution(best_wolf, model)
    
    optimized_results[model_name] = {
        "Best Hyperparameters": best_hyperparams_dict,
        "Best MSE": best_mse
    }

    end_model = time.time()
    print(f"⏱ Optimization Time for {model_name}: {end_model - start_model:.4f} sec")

# Step 5: Performance Evaluation
def evaluate_model(model, best_params):
    """Train and evaluate a model with optimized hyperparameters."""
    start_eval = time.time()
    
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.set_params(**best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    end_eval = time.time()
    print(f"⏱ Evaluation Time for {model.__class__.__name__}: {end_eval - start_eval:.4f} sec")

    return {"MSE": mse, "R² Score": r2}

performance_results = {}
for model_name, model in models.items():
    performance_results[model_name] = evaluate_model(model, optimized_results[model_name]["Best Hyperparameters"])

end_total = time.time()

# Step 6: Print Summary Results
print("\n⏱ Total Execution Time:", round(end_total - start_total, 4), "sec\n")

for model_name in models.keys():
    print(f"{model_name} Performance Before Optimization: {initial_performance[model_name]}")
    print(f"{model_name} Best Hyperparameters: {optimized_results[model_name]['Best Hyperparameters']}")
    print(f"{model_name} Performance After Optimization: {performance_results[model_name]}\n")

