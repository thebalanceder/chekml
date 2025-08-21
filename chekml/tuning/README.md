# Hyperparameter Tuning Framework

This framework provides two complementary methods for **hyperparameter optimization**:

1. **Surrogate-Based Tuning** – efficient approximation of the search space with surrogate models.  
2. **Metaheuristic Search Tuning (Grey Wolf Optimizer)** – population-based search for global optimization in mixed parameter spaces.

---

## General Hyperparameter Schema Extraction

The function `extract_hyperparameter_schema` (in `extract_params.py` and its variants) generates a schema based on the model’s `get_params()`. Parameters are categorized into:

| Category | Examples |
|----------|----------|
| **Boolean** | `[True, False]` |
| **String / Categorical** | `kernel: ['linear', 'poly', 'rbf', 'sigmoid']`<br>`gamma: ['scale', 'auto']`<br>`solver: ['sag', 'saga', 'lsqr', 'lbfgs', 'cholesky', 'auto', 'sparse_cg', 'svd']` |
| **Integer** | `[1, 2, 3, 4, 5]` (or model-specific, e.g., `max_iter: [1, 2, 3, 4, 5, 1000]`) |
| **Float** | `[0, 0.5, 0.9, 1]` (or model-specific, e.g., `alpha`, `tol`) |
| **Special Cases** | `n_jobs: [-1]`, `random_state: [42]`, `positive: [False]` |

### Model-Specific Overrides

- **Ridge (in `surrogate_tune.py`)**  
  - `alpha: [0, 0.5, 0.9, 1]`  
  - `fit_intercept: [True, False]`  
  - `copy_X: [True, False]`  
  - `max_iter: [1, 2, 3, 4, 5, 1000]`  
  - `tol: [0.0001, 0.001, 0.01]`  
  - `solver: ['sag', 'saga', 'lsqr', 'lbfgs', 'cholesky', 'auto', 'sparse_cg', 'svd']`  
  - `positive: [False]`  
  - `random_state: [42]`  

- **LinearRegression (in `metalocal_search_tune.py`)**  
  - `copy_X: [True, False]`  
  - `fit_intercept: [True, False]`  
  - `positive: [True, False]`  

---

## Surrogate-Based Tuning

### Algorithm Explanation
This method approximates the hyperparameter search space using a **surrogate model** to reduce expensive evaluations.

1. **Schema Extraction** – extract hyperparameters and values.  
2. **Data Collection** – randomly sample configurations (`surrogate_iters` times), train the model, and record losses.  
3. **Surrogate Training** – train a `RandomForestRegressor` on the collected data. Handle categoricals with one-hot encoding.  
4. **Coarse Optimization** – sample many candidates (`num_candidates`), predict losses with the surrogate, and pick the best.  
5. **Fine-Tuning** – refine selected hyperparameters:  
   - **Integers:** exponential search around best.  
   - **Floats:** coarse exponential then fine-grained search, with Joblib parallelism.  
   - Skip booleans and categoricals.  
6. **Output** – final tuned config and loss.  

**Advantages:** Efficient in high dimensions, reduces real evaluations.  
**Limitations:** Depends on sampling quality; float fine-tuning can be costly.

---

### Parameters
| Parameter | Description |
|-----------|-------------|
| **model** | Scikit-learn model instance (e.g., `Ridge()`). |
| **loss_fn** | Loss function (e.g., `sklearn.metrics.mean_squared_error`). |
| **X_train, y_train** | Training data. |
| **X_val, y_val** | Validation data. |
| **max_iter** | Maximum fine-tuning iterations (default: `10`). |
| **surrogate_iters** | Number of random samples for surrogate training (default: `30`). |
| **num_candidates** | Number of candidates for surrogate evaluation (default: `1000`). |

---

### Example: Surrogate Tune with Ridge Regression
```python
from sklearn.linear_model import Ridge
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from surrogate_tune import custom_hyperparameter_tuning

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define loss function
def loss_fn(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

# Initialize model
model = Ridge(random_state=42)

# Run surrogate tuning
final_config, final_loss = custom_hyperparameter_tuning(
    model, loss_fn, X_train, y_train, X_val, y_val,
    max_iter=10, surrogate_iters=30, num_candidates=1000
)

print("Final Tuned Config:", final_config)
print("Final Loss:", final_loss)
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
