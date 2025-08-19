import numpy as np
import time
import logging
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression
from wrapper import Wrapper  # ✅ Use our wrapper
import ctypes

# Configure logging
logging.basicConfig(level=logging.INFO)

# Extract hyperparameter schema dynamically
def extract_hyperparameter_schema(model):
    categorical_options = {
        'copy_X': [True, False],
        'fit_intercept': [True, False],
        'positive': [True, False],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'splitter': ['best', 'random'],
    }

    schema = {}
    params = model.get_params()
    for param, value in params.items():
        if value is None:
            continue
        schema[param] = categorical_options[param] if param in categorical_options else [value]
    
    return schema

def decode_solution(solution, model):
    if isinstance(solution, tuple):
        solution = list(solution)

    hyperparams = extract_hyperparameter_schema(model)
    param_keys = list(hyperparams.keys())

    if len(solution) != len(param_keys):
        logging.warning(f"Solution size {len(solution)} doesn't match hyperparameter count {len(param_keys)}")
        return {}

    decoded_solution = {}
    for i, param in enumerate(param_keys):
        if i >= len(solution):
            continue

        value = solution[i]
        if np.isnan(value):
            logging.warning(f"{param} is NaN, using NaN as hyperparameter")
            decoded_solution[param] = np.nan
            continue

        if param not in hyperparams or not hyperparams[param]:
            logging.warning(f"No valid options found for {param}, skipping")
            continue

        options = hyperparams[param]
        if len(options) == 1:
            decoded_solution[param] = options[0]
            continue

        index = int(round(min(max(value, 0), len(options) - 1)))
        decoded_solution[param] = options[index]

    return decoded_solution

# Objective function
def objective_function(solution, model):
    logging.info(f"Checking solution: {solution} (type: {type(solution)})")

    if hasattr(solution, "contents"):  # Handle ctypes pointer
        dim = len(extract_hyperparameter_schema(model))
        solution = np.ctypeslib.as_array(solution, shape=(dim,))

    if isinstance(solution, np.ndarray):
        solution = solution.tolist()

    if not isinstance(solution, list):
        logging.error("Solution is not a valid list!")
        return float('inf')

    decoded_params = decode_solution(solution, model)
    if not decoded_params:
        return float('inf')

    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        model.set_params(**decoded_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return mean_squared_error(y_test, y_pred)
    except Exception as e:
        logging.error(f"Error during training: {e}")
        return float('inf')

# Models
models = {
    "LinearRegression": LinearRegression(),
    "SVM": SVR(),
    "DecisionTree": DecisionTreeRegressor(random_state=42)
}

execution_times = {}
optimized_results = {}
initial_performance = {}
performance_results = {}

start_total = time.time()

# Fine-tuning and visualization
for model_name, model in models.items():
    logging.info(f"\n🔍 Optimizing {model_name}...")

    hyperparams = extract_hyperparameter_schema(model)
    param_keys = list(hyperparams.keys())

    if not param_keys:
        logging.warning(f"No optimizable hyperparameters for {model_name}, skipping...")
        continue

    bounds = [(0, 1) if isinstance(values[0], bool) else (0, len(values) - 1) for values in hyperparams.values()]
    
    # Measure initial performance
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    initial_performance[model_name] = {"MSE": mean_squared_error(y_test, y_pred), "R²": r2_score(y_test, y_pred)}

    # ✅ Use our wrapper
    optimizer = Wrapper(dim=len(bounds), bounds=bounds, population_size=10, max_iter=20, method="DISO")

    try:
        start_time = time.time()
        optimizer.optimize(lambda sol: objective_function(sol, model))
        end_time = time.time()
        execution_times[model_name] = end_time - start_time

        best_solution, best_fitness = optimizer.get_best_solution()
        best_hyperparams_dict = decode_solution(best_solution, model)

        optimized_results[model_name] = {"Best Hyperparameters": best_hyperparams_dict}

        # Evaluate optimized model
        model.set_params(**best_hyperparams_dict)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        performance_results[model_name] = {"MSE": mean_squared_error(y_test, y_pred), "R²": r2_score(y_test, y_pred)}

    except Exception as e:
        logging.error(f"Error during optimization for {model_name}: {e}")

    finally:
        optimizer.free()  # ✅ Ensure memory cleanup
        
end_total = time.time()
logging.info(f"\n⏱ Total Execution Time: {round(end_total - start_total, 4)} sec\n")

# Print summary
for model_name in models.keys():
    logging.info(f"\n📌 {model_name} Initial Performance: {initial_performance.get(model_name, 'N/A')}")
    logging.info(f"🔹 {model_name} Best Hyperparameters: {optimized_results.get(model_name, {}).get('Best Hyperparameters', 'N/A')}")
    logging.info(f"✅ {model_name} Optimized Performance: {performance_results.get(model_name, 'N/A')}")
    logging.info(f"⏱ Execution Time: {execution_times.get(model_name, 0):.4f} sec\n")
