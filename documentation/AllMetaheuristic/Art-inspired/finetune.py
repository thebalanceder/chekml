import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression
from stochastic_paint_optimizer import StochasticPaintOptimizer  # Assuming SPO is implemented here

# Step 1: Extract Hyperparameters
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

# Step 2: Decode Hyperparameters
def decode_solution(wolf, model):
    hyperparams = extract_hyperparameter_schema(model)
    param_keys = list(hyperparams.keys())
    decoded = {}
    for i, param in enumerate(param_keys):
        values = hyperparams[param]
        if isinstance(values[0], bool):
            decoded[param] = bool(int(round(min(max(wolf[i], 0), 1))))
        else:
            index = int(round(min(max(wolf[i], 0), len(values) - 1)))
            decoded[param] = values[index]
    return decoded

# Step 3: Define Objective Function
def objective_function(wolf, model):
    solution = decode_solution(wolf, model)
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.set_params(**solution)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

# Step 4: Run Optimization
optimized_results = {}
initial_performance = {}

start_total = time.time()

for model_name, model in models.items():
    start_model = time.time()
    
    hyperparams = extract_hyperparameter_schema(model)
    param_keys = list(hyperparams.keys())
    bounds = [(0, 1) if isinstance(v[0], bool) else (0, len(v) - 1) for v in hyperparams.values()]
    
    # Initial performance
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    initial_performance[model_name] = {
        "MSE": mean_squared_error(y_test, y_pred),
        "R² Score": r2_score(y_test, y_pred)
    }

    spo = StochasticPaintOptimizer(
        objective_function=lambda w: objective_function(w, model),
        dim=len(param_keys),
        bounds=bounds,
        population_size=10,
        max_iter=20
    )
    
    best_wolf, best_mse, _ = spo.optimize()
    best_hyperparams = decode_solution(best_wolf, model)
    
    optimized_results[model_name] = {
        "Best Hyperparameters": best_hyperparams,
        "Best MSE": best_mse
    }

    print(f"⏱ Optimization Time for {model_name}: {time.time() - start_model:.4f} sec")

# Step 5: Evaluation
def evaluate_model(model, best_params):
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.set_params(**best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "MSE": mean_squared_error(y_test, y_pred),
        "R² Score": r2_score(y_test, y_pred)
    }

performance_results = {
    name: evaluate_model(models[name], result["Best Hyperparameters"])
    for name, result in optimized_results.items()
}

end_total = time.time()

# Step 6: Summary
print("\n⏱ Total Execution Time:", round(end_total - start_total, 4), "sec\n")
for name in models.keys():
    print(f"{name} Initial Performance: {initial_performance[name]}")
    print(f"{name} Best Hyperparameters: {optimized_results[name]['Best Hyperparameters']}")
    print(f"{name} Performance After Optimization: {performance_results[name]}\n")

