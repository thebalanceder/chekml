# Ensemble Algorithms Implementation Guide

This guide provides an overview of several custom ensemble algorithms, their implementations, and usage examples.  
All algorithms are compatible with **scikit-learn** estimators for both regression and classification tasks.

## 1. GreedyEnsembleSelection

### Description
Greedy ensemble selection that iteratively selects and weights base models to minimize validation error.  
Optimizes weights using constrained minimization. Works for both regression (MSE) and classification (accuracy).

### Algorithm Steps
1. Fit base models on training data and predict on validation data.  
2. Start with an empty ensemble.  
3. For each remaining model:
   - Temporarily add it to the ensemble.  
   - Optimize weights to minimize error (MSE or negative accuracy).  
4. Select the model that yields the best improvement and update weights.  
5. Repeat until all models are added or no improvement is found.  
6. Final predictions are **weighted averages** based on optimized weights.  

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models`  | `dict` | - | Dictionary of `{name: model}`, e.g. `{"Linear": LinearRegression()}`. Required. |
| `task`    | `str`  | `"regression"` | Task type: `"regression"` or `"classification"`. Determines error metric. |

### Usage Example
```python
from chekml.ensemble.greedy_ensemble import GreedyEnsembleSelection
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Sample data
np.random.seed(42)
X_reg = np.random.rand(100, 2) * 10
y_reg = 2 * X_reg[:, 0] + 3 * X_reg[:, 1] + np.random.randn(100)
X_train_reg, X_test_reg = X_reg[:80], X_reg[80:]
y_train_reg, y_test_reg = y_reg[:80], y_reg[80:]

# Base models
reg_models_dict = {
    "Linear": LinearRegression(),
    "SVR": SVR(),
    "Tree": DecisionTreeRegressor()
}

gens = GreedyEnsembleSelection(reg_models_dict, task="regression")
gens.fit(X_train_reg, y_train_reg, X_test_reg, y_test_reg)
preds_gens = gens.predict(X_test_reg)
print("Greedy Ensemble MSE:", mean_squared_error(y_test_reg, preds_gens))
```

## 2. BayesianModelCombination

### Description
Bayesian-inspired ensemble that assigns probabilistic weights to base models based on validation likelihoods.  
Weights are computed as the exponential of negative MSE, favoring models with lower errors.

### Algorithm Steps
1. Train each base model on training data.  
2. Predict on validation data and compute MSE for each.  
3. Compute likelihood as `exp(-MSE)` for each model.  
4. Normalize likelihoods → probabilistic weights.  
5. Final predictions = **weighted average** using these weights.  

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models`  | `dict` | - | Dictionary of `{name: model}`. Required. |

### Usage Example
```python
from chekml.ensemble.bayesian_model_combination import BayesianModelCombination
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Sample data
np.random.seed(42)
X_reg = np.random.rand(100, 2) * 10
y_reg = 2 * X_reg[:, 0] + 3 * X_reg[:, 1] + np.random.randn(100)
X_train_reg, X_test_reg = X_reg[:80], X_reg[80:]
y_train_reg, y_test_reg = y_reg[:80], y_reg[80:]

reg_models_dict = {
    "Linear": LinearRegression(),
    "SVR": SVR(),
    "Tree": DecisionTreeRegressor()
}

bmc = BayesianModelCombination(reg_models_dict)
bmc.fit(X_train_reg, y_train_reg, X_test_reg, y_test_reg)
preds_bmc = bmc.predict(X_test_reg)
print("Bayesian MC MSE:", mean_squared_error(y_test_reg, preds_bmc))
```

## 3. NCLEnsemble (Negative Correlation Learning)

### Description
Promotes diversity by penalizing correlated predictions among base models.  
Encourages **negative correlation** so that models complement each other rather than agree.

### Algorithm Steps
1. Scale features and train each base model.  
2. Predict with all models and compute ensemble mean.  
3. Calculate penalty = average squared deviation from the mean (encourages diversity).  
4. Final prediction =  
   - Regression → ensemble mean  
   - Classification → argmax / threshold  

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models`  | `list` | - | List of scikit-learn estimators. Required. |
| `task`    | `str`  | `"regression"` | Task type: `"regression"` or `"classification"`. |

### Usage Example
```python
from chekml.ensemble.negative_correlation_ensemble import NCLEnsemble
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)
X_clf = np.random.rand(100, 2) * 10
y_clf = (X_clf[:, 0] + X_clf[:, 1] > 10).astype(int)
X_train, X_test = X_clf[:80], X_clf[80:]
y_train, y_test = y_clf[:80], y_clf[80:]

models = [SVC(probability=True), DecisionTreeClassifier()]

ncl = NCLEnsemble(models, task="classification")
ncl.fit(X_train, y_train)
preds = ncl.predict(X_test)
print("NCL Accuracy:", accuracy_score(y_test, preds))
```

## 4. RLEnsemble (Reinforcement Learning Ensemble)

### Description
Applies **Q-learning** to dynamically select the best base model.  
Learns through **rewards** (−MSE) over multiple episodes, balancing **exploration vs. exploitation**.

### Algorithm Steps
1. Scale features and train base models.  
2. Initialize Q-table (model → value).  
3. For each episode:  
   - Choose model via epsilon-greedy strategy.  
   - Predict on validation → compute reward (−MSE).  
   - Update Q-table with Bellman equation.  
   - Decay epsilon to favor exploitation over time.  
4. Final model = one with **highest Q-value**.  

### Parameters
| Parameter     | Type | Default | Description |
|---------------|------|---------|-------------|
| `models`      | `dict` | - | Dictionary `{name: model}`. Required. |
| `num_episodes`| `int`  | `100` | Number of Q-learning episodes. |

### Usage Example
```python
from chekml.ensemble.rl_ensemble import RLEnsemble
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Sample data
np.random.seed(42)
X_reg = np.random.rand(100, 2) * 10
y_reg = 2 * X_reg[:, 0] + 3 * X_reg[:, 1] + np.random.randn(100)
X_train_reg, X_test_reg = X_reg[:80], X_reg[80:]
y_train_reg, y_test_reg = y_reg[:80], y_reg[80:]

models = {
    "Linear": LinearRegression(),
    "SVR": SVR(),
    "Tree": DecisionTreeRegressor()
}

rl = RLEnsemble(models, num_episodes=50)
rl.fit(X_train_reg, y_train_reg, X_test_reg, y_test_reg)
preds = rl.predict(X_test_reg)
print("RL Ensemble MSE:", mean_squared_error(y_test_reg, preds))
```

## 5. HierarchicalStacking

### Description
**Hierarchical Stacking** is a stacking ensemble method where predictions from base models are used as input features for a meta-learner.  
It performs an internal train-validation split to create meta-features, enabling hierarchical learning and improved generalization.

### Algorithm Steps
1. Split data into **train** and **validation** sets.  
2. Train each base model on the train split.  
3. Predict on the validation set to generate **meta-features** (stacked predictions).  
4. Train a meta-learner using these meta-features and validation labels:  
   - Regression → **LinearRegression**  
   - Classification → **LogisticRegression**  
5. For new test data:  
   - Generate meta-features from base models.  
   - Predict with the meta-learner.

### Parameters
| Parameter      | Type  | Default        | Description |
|----------------|-------|----------------|-------------|
| `base_models`  | `list` | -              | List of `(name, model)` tuples. Required. |
| `task`         | `str`  | `"regression"` | Task type: `"regression"` or `"classification"`. |

### Usage Example
```python
from chekml.ensemble.hierarchical_stacking import HierarchicalStacking
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Sample classification data
np.random.seed(42)
X_clf = np.random.rand(100, 2) * 10
y_clf = (X_clf[:, 0] + X_clf[:, 1] > 10).astype(int)
X_train_clf, X_test_clf = X_clf[:80], X_clf[80:]
y_train_clf, y_test_clf = y_clf[:80], y_clf[80:]

# Define base models
base_models = [("SVC", SVC()), ("Tree", DecisionTreeClassifier())]

# Hierarchical Stacking
hs = HierarchicalStacking(base_models, task="classification")
hs.fit(X_clf, y_clf)  # internal split used
preds = hs.predict(X_test_clf)
print("Hierarchical Stacking Accuracy:", accuracy_score(y_test_clf, preds.round()))
```
