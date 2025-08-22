# All Available Supervise Learning Methods

## 1.FuzzySVM

## Overview
**Fuzzy Support Vector Machine (FSVM)** is an extension of the standard SVM that handles noisy or outlier data by assigning **fuzzy membership weights** to samples. This reduces the influence of unreliable points and improves robustness, especially in real-world datasets.

## Available Parameters

| Parameter     | Type      | Default   | Description                                                                 |
|---------------|----------|-----------|-----------------------------------------------------------------------------|
| `kernel`      | str      | `'rbf'`   | Kernel type (`'rbf'`, `'linear'`, `'poly'`, etc.).                          |
| `C`           | float    | `1.0`     | Regularization parameter.                                                   |
| `gamma`       | str/float| `'scale'` | Kernel coefficient for `'rbf'`/`'poly'`.                                    |
| `fuzzy_method`| str      | `'distance'` | Method for computing fuzzy memberships (e.g., `'distance'`).               |

## Algorithm Explanation
The **Fuzzy SVM** assigns a **membership value** to each training sample based on its relevance or proximity to the class center. For example, points closer to the class mean receive higher weights, while noisy or outlier points receive lower weights.  

This transforms the standard SVM optimization into a **weighted SVM problem**, where the margin is maximized while accounting for sample fuzziness. The result is a more robust classifier that generalizes better on noisy data.

## Example Implementation

```python
import numpy as np
from sklearn.datasets import make_classification
from chekml.supervise.FuSVM import FuzzySVM  # Assuming the file is named FuSVM.py

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Initialize and train model
fuzzy_svm = FuzzySVM(kernel='rbf', C=1.0, gamma='scale', fuzzy_method='distance')
fuzzy_svm.fit(X, y)

# Predict on new data
predictions = fuzzy_svm.predict(X[:5])
print("Predictions:", predictions)

# Optional: Hyperparameter optimization
fuzzy_svm.optimize_hyperparameters(X, y)
```

## 2.TwinSVM & MultiClassTwinSVM

## Overview
**Twin Support Vector Machine (TWSVM)** is a faster variant of SVM that constructs **two non-parallel hyperplanes** instead of one. Each hyperplane is closer to one class while being farther from the other.  

This approach reduces the optimization problem into **two smaller quadratic programming problems (QPPs)**, making it more efficient than traditional SVM.  

The **MultiClassTwinSVM** extends the binary TWSVM using a **one-vs-one strategy**, training binary classifiers for each pair of classes and using majority voting for final predictions.

## Available Parameters

| Parameter       | Type      | Default     | Description                                                                 |
|-----------------|----------|-------------|-----------------------------------------------------------------------------|
| `C1`            | float    | `1.0`       | Regularization parameter for **class +1**.                                  |
| `C2`            | float    | `1.0`       | Regularization parameter for **class -1**.                                  |
| `kernel`        | str      | `'linear'`  | Kernel type (`'linear'`, `'rbf'`, `'poly'`, `'custom'`).                    |
| `gamma`         | float    | `1.0`       | RBF kernel coefficient.                                                     |
| `degree`        | int      | `3`         | Degree for polynomial kernel.                                               |
| `custom_kernel` | callable | `None`      | Custom kernel function (if provided).                                       |

## Algorithm Explanation
Unlike standard SVM which finds a single optimal separating hyperplane, **TwinSVM** constructs **two hyperplanes**:
- One is closer to the positive class but far from the negative class.  
- The other is closer to the negative class but far from the positive class.  

This results in **faster training** since each hyperplane is solved using a smaller QPP.  

For **multi-class problems**, TwinSVM uses a **one-vs-one scheme**, where multiple binary classifiers are trained for each class pair, and predictions are made via **majority voting**.

## Example Implementation

### Binary TwinSVM
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from chekml.supervise.TwinSVM import TwinSVM

# Generate binary dataset (labels -1 and +1)
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
y = np.where(y == 0, -1, 1)  # Convert labels to -1 and +1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train TwinSVM
model = TwinSVM(C1=1, C2=1, kernel='rbf', gamma=1)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Multi-class TwinSVM
```python
from chekml.supervise.TwinSVM import MultiClassTwinSVM
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Generate multi-class dataset
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,   # all features informative
    n_redundant=0,
    n_repeated=0,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train MultiClassTwinSVM
model = MultiClassTwinSVM(C1=1, C2=1, kernel='linear')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

## 3.HellingerDecisionTree

## Overview
**Hellinger Distance Decision Tree (HDDT)** is a decision tree variant designed for **imbalanced datasets**.  
Instead of using traditional split criteria like **Gini impurity** or **entropy**, HDDT uses the **Hellinger distance**, which is insensitive to skewed class distributions.  

This makes HDDT especially effective when one class heavily outnumbers the other(s).

## Available Parameters

| Parameter           | Type        | Default | Description                                                         |
|---------------------|------------|---------|---------------------------------------------------------------------|
| `max_depth`         | int or None| `None`  | Maximum depth of the tree (prevents overfitting).                   |
| `min_samples_split` | int        | `2`     | Minimum number of samples required to split an internal node.       |

## Algorithm Explanation
- HDDT chooses the split that **maximizes the Hellinger distance** between the class probability distributions in the left and right child nodes.  
- The **Hellinger distance** is given by:  

\[
H(P, Q) = \sqrt{ \sum_i \left( \sqrt{P_i} - \sqrt{Q_i} \right)^2 }
\]

- Unlike entropy or Gini, the Hellinger distance is **skew-insensitive**, which means it remains robust even when the dataset is highly imbalanced.  
- The tree grows recursively until reaching stopping conditions (max depth, node purity, or minimum samples).

## Example Implementation

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from chekml.supervise.HDT import HellingerDecisionTree

# Generate imbalanced dataset
X, y = make_classification(
    n_samples=5000,
    n_features=10,
    weights=[0.9, 0.1],  # 90% negative, 10% positive
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train HDDT
hddt = HellingerDecisionTree(max_depth=10, min_samples_split=5)
hddt.fit(X_train, y_train)

# Predict and evaluate
preds = hddt.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print("F1-Score:", f1_score(y_test, preds, average='weighted'))
```
## 4.ProximalSVM

## Overview
**Proximal Support Vector Machine (PSVM)** is an efficient variant of SVM that reformulates the optimization problem into a **regularized least-squares problem**.  
Instead of solving a quadratic programming (QP) problem (as in standard SVM), PSVM uses **matrix inversion** or **LU decomposition** for much faster training.  

This makes it particularly well-suited for **large-scale datasets**.

## Available Parameters

| Parameter   | Type     | Default | Description                                                        |
|-------------|----------|---------|--------------------------------------------------------------------|
| `C`         | float    | `1.0`   | Regularization parameter controlling margin vs. error trade-off.   |
| `use_gpu`   | bool     | `False` | Enable GPU acceleration if **CuPy** is available.                  |
| `kernel`    | callable or None | `None` | Custom kernel function (default: `None`, uses linear kernel). |

## Algorithm Explanation
- Standard SVM requires solving a **quadratic programming problem (QPP)**, which is computationally expensive for large datasets.  
- ProximalSVM avoids QPP by reformulating the optimization as:  

\[
\min_w \| Xw - y \|^2 + \frac{1}{C} \| w \|^2
\]

- This reduces to solving a **regularized least-squares system**:  

\[
(X^T X + \frac{I}{C}) w = X^T y
\]

- Solved efficiently using **matrix inversion** or **LU decomposition**.  
- Classification is done by assigning each point to the **closest of two parallel hyperplanes**.  

## Example Implementation

```python
import numpy as np
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from chekml.supervise.PSVC import ProximalSVM

# Generate dataset (labels -1/1)
X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
y = 2 * y - 1  # Convert {0,1} → {-1,1}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train PSVM
model = ProximalSVM(C=1.0, use_gpu=True, kernel=None)  # Linear kernel
start = time.time()
model.fit(X_train, y_train)
print("Training Time:", time.time() - start)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```
## 5.SupportMatrixMachine (SMM)

## Overview
**Support Matrix Machine (SMM)** is an extension of the traditional Support Vector Machine (SVM) that directly operates on **matrix-structured inputs** (e.g., images, correlation matrices).  

Instead of flattening the data, SMM preserves the **matrix structure** by incorporating a **spectral elastic net regularizer**, which combines the Frobenius and nuclear norms.  
This makes it more effective for structured data and high-dimensional problems.

## Available Parameters

| Parameter        | Type   | Default | Description                                                                 |
|------------------|--------|---------|-----------------------------------------------------------------------------|
| `learning_rate`  | float  | `0.01`  | Gradient descent learning rate.                                             |
| `epochs`         | int    | `1000`  | Number of training epochs.                                                  |
| `reg_param`      | float  | `0.1`   | Regularization strength (spectral elastic net).                             |

## Algorithm Explanation
- Standard SVM flattens data into vectors, which can lose important structural information.  
- **SMM** instead keeps the input as a **matrix** and uses a specialized regularizer:  

\[
\Omega(W) = \alpha \| W \|_F^2 + (1-\alpha)\| W \|_*
\]

Where:
- \(\| W \|_F\) is the Frobenius norm (L2-like regularization).  
- \(\| W \|_*\) is the nuclear norm (sum of singular values, promoting low-rank solutions).  

- The model minimizes hinge loss with this regularization using **parallel mini-batch gradient descent**.  
- Training is distributed across CPU cores, and weights are averaged for the final model.  

## Example Implementation

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from chekml.supervise.SMM import SupportMatrixMachine

# Generate synthetic matrix data (labels -1/1)
np.random.seed(42)
X = np.random.randn(1000, 20)  # Can also be reshaped matrices (e.g., images)
y = np.random.choice([-1, 1], size=1000)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# Train Support Matrix Machine
smm = SupportMatrixMachine(learning_rate=0.01, epochs=1000, reg_param=0.1)
smm.fit(X_train, y_train)

# Predict and evaluate
preds = smm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print("Precision:", precision_score(y_test, preds))
print("Recall:", recall_score(y_test, preds))
print("F1 Score:", f1_score(y_test, preds))
```

## 6.Fast Feature Linear Regression (FFLR)

## Overview
**Fast Feature Linear Regression (FFLR)** is an optimized linear regression solver designed for **large-scale datasets**.  
It uses advanced numerical methods (LU decomposition), GPU acceleration (via CuPy), parallelism (via Joblib), and mini-batch stochastic gradient descent (Numba-JIT) for handling **streaming and massive data efficiently**.

---

## Available Parameters

| Parameter        | Type   | Default   | Description                                                                 |
|------------------|--------|-----------|-----------------------------------------------------------------------------|
| `use_gpu`        | bool   | `False`   | Enable GPU acceleration via CuPy.                                           |
| `use_parallel`   | bool   | `False`   | Enable parallel processing with Joblib.                                     |
| `use_mini_batch` | bool   | `False`   | Use mini-batch gradient descent instead of closed-form solver.              |
| `alpha`          | float  | `0.1`     | Ridge regularization parameter.                                             |
| `lr`             | float  | `0.01`    | Learning rate for mini-batch gradient descent.                              |
| `epochs`         | int    | `10`      | Number of epochs for mini-batch training.                                   |
| `n_jobs`         | int    | `-1`      | Number of parallel jobs (`-1` uses all CPU cores).                          |
| `num_threads`    | int    | `12`      | CPU threads for BLAS backend operations.                                    |
| `batch_size`     | int    | `10000`   | Batch size for mini-batch training.                                         |

---

## Algorithm Explanation
The optimization problem solved by FFLR is the **ridge-regularized least squares**:

\[
\hat{\theta} = (X^T X + \alpha I)^{-1} X^T y
\]

- **LU decomposition** is used for fast solving of the regularized normal equations.  
- Supports **GPU acceleration** via CuPy for large datasets.  
- Supports **parallel processing** with Joblib by splitting data into chunks.  
- Supports **mini-batch gradient descent** with Numba-JIT for streaming/large data where exact matrix inversion is infeasible.  

This makes FFLR suitable for **both exact solutions** (small/medium datasets) and **approximate fast training** (massive datasets).

---

## Example Implementation

```python
import dask.array as da
import time
from sklearn.metrics import mean_squared_error
from chekml.supervise.FFLR import FFLR  # Or use FFLR_ from FFLR_.py

# Generate large Dask dataset
X = da.random.random((1000000, 10), chunks=(100000, 10))
y = da.random.random((1000000,), chunks=(100000,))
X_np = X[:100000].compute()  # Subset for training
y_np = y[:100000].compute()

# Train FFLR with parallelism
model = FFLR(use_parallel=True, n_jobs=4, alpha=0.1)
start = time.time()
model.fit(X_np, y_np)
print("Training Time:", time.time() - start)

# Predict and evaluate
y_pred = model.predict(X_np)
print("MSE:", mean_squared_error(y_np, y_pred))
```
