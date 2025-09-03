## Wrapper Class

The **Wrapper class** provides a unified interface for running optimization algorithms (e.g., DISO).  
It allows flexible configuration of **dimensionality, population size, iterations, and search bounds**, making it suitable for tasks like function minimization, hyperparameter tuning, and robustness testing.

---

### Parameters

| Parameter           | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| **dim** (int)       | Dimensionality of the solution space (number of variables). Must be > 0.    |
| **population_size** (int) | Number of candidate solutions per iteration. Typically 10–50.              |
| **max_iter** (int)  | Maximum number of iterations. Higher values improve accuracy but cost more. |
| **bounds** (list)   | Search space boundaries. Must match `dim`. Each entry can be `(min, max)` or uniform. |
| **method** (str)    | Optimization algorithm (e.g., `"DISO"`). Must be a valid supported method.  |

---

### Submodule Performance Comparison

| Submodule           | Accuracy | Speed | Description |
|---------------------|----------|-------|-------------|
| CIntegration_cython | Precision accuracy | Baseline Fast | Minor optimization towards speed but ensure everything safe and sound |
| CIntegration_ctypes | Precision accuracy | Optimized Fast | Optimize using C coding techniques, with ctypes for python wrapper
| CIntegration_cffi   | Precision accuracy | Optimized Fast | Optimize by C with cffi to parse to python
| CIntegration_cffi1  | Very High Accuracy | High Speed | Integrated with AVX techniques, branchless and other techniques with highly optimized for CPU utilization
| CIntegration_cffi2  | High Accuracy | Very Fast | CPU + GPU hybrid using opencl
| CIntegration_cffi3  | High Accuracy | Maximmum speed | Major operations like matrix-multiplication carry out in GPU with opencl

---

### Key Methods

- if using CIntegration_ctypes, use:
```python
# Define the objective function for ctypes
@ctypes.CFUNCTYPE(ctypes.c_double, ctypes.POINTER(ctypes.c_double))
def objective_function(x):
    """Compute the sum of squares for a C double array."""
    # Convert ctypes pointer to numpy array (assuming dim=10)
    x_array = np.ctypeslib.as_array(x, shape=(10,))
    fitness = np.sum(x_array**2)
    logging.debug(f"Objective function called with x={x_array}, fitness={fitness}")
    return ctypes.c_double(fitness)
```

- **`optimize(objective_function)`**  
  Runs the optimization on the given objective function.  
  Input: A callable that takes a vector `x` and returns a scalar fitness.

- **`get_best_solution()`**  
  Returns `(best_solution, best_fitness)` after optimization.

- **`free()`**  
  Releases allocated resources. Always call after optimization.

---

### Examples

#### Example 1: General Optimization (Quadratic Function)

Optimizes a quadratic function `sum(x**2)` in 10 dimensions.

```python
import numpy as np
import logging
from chekml.MetaheuristicOptimization.CIntegration_cffi import wrapper  # Adjust import

logging.basicConfig(level=logging.INFO)

def objective_function(x, dim=10):
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=np.float64)
    return np.sum(x**2)

def main():
    dim, population_size, max_iter = 10, 20, 50
    bounds, method = [-5.0, 5.0] * dim, "DISO"

    optimizer = wrapper.Wrapper(dim, population_size, max_iter, bounds, method=method)
    optimizer.optimize(lambda x: objective_function(x, dim=dim))

    best_solution, best_fitness = optimizer.get_best_solution()
    logging.info(f"Best solution: {best_solution}, Fitness: {best_fitness}")

    optimizer.free()

if __name__ == "__main__":
    main()
```

### Example 2: Hyperparameter Tuning (SVR)
Optimizes `kernel`, `C`, and `epsilon` for an SVR model to minimize MSE.

```python
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
import logging
from chekml.MetaheuristicOptimization.CIntegration_cffi import wrapper

logging.basicConfig(level=logging.INFO)

def extract_svm_hyperparameter_schema():
    return {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': (0.1, 10.0),
        'epsilon': (0.01, 1.0),
    }

def decode_svm_solution(solution, schema):
    if isinstance(solution, np.ndarray):
        solution = solution.tolist()
    decoded = {}
    for i, param in enumerate(schema.keys()):
        value, options = solution[i], schema[param]
        if isinstance(options, tuple):
            min_val, max_val = options
            scaled = min_val + (max_val - min_val) * (value / (len(schema) - 1))
            decoded[param] = max(min_val, min(max_val, scaled))
        else:
            index = int(round(min(max(value, 0), len(options) - 1)))
            decoded[param] = options[index]
    return decoded

def svm_objective_function(solution, dim=3):
    schema = extract_svm_hyperparameter_schema()
    params = decode_svm_solution(solution, schema)
    if not params:
        return float('inf')
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    try:
        model = SVR(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return mean_squared_error(y_test, y_pred)
    except Exception as e:
        logging.error(f"Error: {e}")
        return float('inf')

def main():
    dim, population_size, max_iter = 3, 20, 50
    bounds, method = [(0, 3), (0, 3), (0, 3)], "DISO"

    optimizer = wrapper.Wrapper(dim, population_size, max_iter, bounds, method=method)
    optimizer.optimize(svm_objective_function)

    best_solution, best_fitness = optimizer.get_best_solution()
    best_params = decode_svm_solution(best_solution, extract_svm_hyperparameter_schema())
    logging.info(f"Best params: {best_params}, MSE: {best_fitness}")

    optimizer.free()

if __name__ == "__main__":
    main()
```

 ### Example 3: Robustness to Noise
 Tests optimization under noisy quadratic objectives.

```python
import numpy as np
import logging
from chekml.MetaheuristicOptimization.CIntegration_cffi import wrapper

logging.basicConfig(level=logging.INFO)

def noisy_objective_function(x, dim=10, noise_level=0.1):
    true_value = np.sum(np.array(x)**2)
    noise = np.random.normal(0, noise_level * (true_value + 1e-10))
    return true_value + noise

def main():
    dim, population_size, max_iter = 10, 20, 50
    bounds, method = [-5.0, 5.0] * dim, "DISO"

    optimizer = wrapper.Wrapper(dim, population_size, max_iter, bounds, method=method)
    optimizer.optimize(lambda x: noisy_objective_function(x, dim=dim))

    best_solution, best_fitness = optimizer.get_best_solution()
    logging.info(f"Best solution: {best_solution}, Fitness: {best_fitness}")

    optimizer.free()

if __name__ == "__main__":
    main()
```

 ### Example 4: Scalability Test (20D)
 Evaluates performance in higher dimensions.

```python
import numpy as np
import logging
import time
from chekml.MetaheuristicOptimization.CIntegration_cffi import wrapper

logging.basicConfig(level=logging.INFO)

def objective_function(x, dim=20):
    return np.sum(np.array(x)**2)

def main():
    dim, population_size, max_iter = 20, 20, 50
    bounds, method = [-5.0, 5.0] * dim, "DISO"

    optimizer = wrapper.Wrapper(dim, population_size, max_iter, bounds, method=method)

    start_time = time.time()
    optimizer.optimize(lambda x: objective_function(x, dim=dim))
    elapsed = time.time() - start_time

    best_solution, best_fitness = optimizer.get_best_solution()
    logging.info(f"Best fitness: {best_fitness}, Time: {elapsed:.2f}s")

    optimizer.free()

if __name__ == "__main__":
    main()
```
