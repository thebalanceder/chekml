import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from metalocal_search_tune import extract_hyperparameter_schema
from Metegrey_wolf_optimizer import GreyWolfOptimizer

# Instantiate model and extract hyperparameters
model = LinearRegression()
hyperparams = extract_hyperparameter_schema(model)
param_keys = list(hyperparams.keys())

# Decode function
def decode_solution(wolf):
    decoded_solution = []
    for i, param in enumerate(param_keys):
        if isinstance(hyperparams[param][0], bool):
            decoded_solution.append(bool(int(round(wolf[i]))))
        else:
            decoded_solution.append(hyperparams[param][int(round(wolf[i]))])
    return decoded_solution

# Objective function
def objective_function(wolf):
    solution = decode_solution(wolf)
    param_values = dict(zip(param_keys, solution))
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression(**param_values)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

# Bounds preparation
bounds = []
for param, values in hyperparams.items():
    if isinstance(values[0], bool):
        bounds.append((0, 1))
    else:
        bounds.append((0, len(values) - 1))

# Run GWO
gwo = GreyWolfOptimizer(objective_function, dim=len(param_keys), bounds=bounds, population_size=10, max_iter=20)
best_wolf, best_mse, _ = gwo.optimize()

# Decode best solution
best_hyperparams = dict(zip(param_keys, decode_solution(best_wolf)))
print("Best Hyperparameters:", best_hyperparams)
print("Best MSE:", best_mse)
```

## Metaheuristic Search Tuning (Grey Wolf Optimizer)

### Algorithm Explanation
The **Grey Wolf Optimizer (GWO)** is a metaheuristic inspired by the leadership hierarchy and hunting mechanism of grey wolves.  
It is adapted here for **hyperparameter tuning** in mixed search spaces.

1. **Schema Extraction** – extract model hyperparameters.  
2. **Bounds Preparation** – map each hyperparameter to numerical ranges (e.g., `0–1` for booleans, index ranges for categoricals).  
3. **Objective Function** – decode wolf positions into parameter values, train the model, and return the loss (e.g., MSE).  
4. **Optimization** – wolves update positions based on the best three wolves (**alpha**, **beta**, and **delta**) to balance exploration and exploitation.  
5. **Decoding** – convert wolf positions back into valid hyperparameter values.  
6. **Output** – best hyperparameter configuration and corresponding performance.  

**Advantages:**  
- Handles **mixed data types** and **non-convex spaces**.  
- Naturally balances exploration and exploitation.  

**Limitations:**  
- Requires a proper GWO implementation.  
- May need many iterations for convergence.  

---

### Parameters
| Parameter | Description |
|-----------|-------------|
| **objective_function** | Function to minimize (e.g., MSE). |
| **dim** | Number of hyperparameters. |
| **bounds** | Parameter ranges for optimization. |
| **population_size** | Number of wolves in the pack (default: `10`). |
| **max_iter** | Maximum number of iterations (default: `20`). |

---

### Example: Hyperparameter Tuning with Linear Regression
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from metalocal_search_tune import extract_hyperparameter_schema
from Metegrey_wolf_optimizer import GreyWolfOptimizer

# Instantiate model and extract hyperparameters
model = LinearRegression()
hyperparams = extract_hyperparameter_schema(model)
param_keys = list(hyperparams.keys())

# Decode function
def decode_solution(wolf):
    decoded_solution = []
    for i, param in enumerate(param_keys):
        if isinstance(hyperparams[param][0], bool):
            decoded_solution.append(bool(int(round(wolf[i]))))
        else:
            decoded_solution.append(hyperparams[param][int(round(wolf[i]))])
    return decoded_solution

# Objective function
def objective_function(wolf):
    solution = decode_solution(wolf)
    param_values = dict(zip(param_keys, solution))
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression(**param_values)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

# Bounds preparation
bounds = []
for param, values in hyperparams.items():
    if isinstance(values[0], bool):
        bounds.append((0, 1))
    else:
        bounds.append((0, len(values) - 1))

# Run Grey Wolf Optimizer
gwo = GreyWolfOptimizer(
    objective_function,
    dim=len(param_keys),
    bounds=bounds,
    population_size=10,
    max_iter=20
)
best_wolf, best_mse, _ = gwo.optimize()

# Decode best solution
best_hyperparams = dict(zip(param_keys, decode_solution(best_wolf)))
print("Best Hyperparameters:", best_hyperparams)
print("Best MSE:", best_mse)
```
