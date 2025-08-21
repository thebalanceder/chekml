## Semi-Supervised Decision Tree (SSDT)

### Algorithm Explanation
The **Semi-Supervised Decision Tree (SSDT)** extends a standard decision tree by leveraging **unlabeled data** through pseudo-labeling.

1. **Initialization** – takes a base decision tree (e.g., `DecisionTreeClassifier`) and a confidence threshold.  
2. **Initial Training** – fits the base tree on labeled data `(X_labeled, y_labeled)`.  
3. **Pseudo-Labeling** –  
   - Predicts probabilities for unlabeled data `(X_unlabeled)` using `predict_proba`.  
   - Selects samples where the maximum probability exceeds `confidence_threshold`.  
   - Assigns pseudo-labels to these samples.  
4. **Retraining** – combines labeled and confident pseudo-labeled data, retrains the tree.  
5. **Prediction** – uses the trained tree for new data predictions.  

**Advantages:** Uses unlabeled data to improve accuracy when labeled data is scarce.  
**Limitations:** Relies on high-confidence predictions; weak initial models may propagate errors.

---

### Parameters
| Parameter | Description |
|-----------|-------------|
| **base_tree** | A scikit-learn decision tree classifier (e.g., `DecisionTreeClassifier(max_depth=10, random_state=42)`). |
| **max_depth** | Maximum depth of the tree (from `base_tree`). |
| **min_samples_split** | Minimum samples required to split a node (default: `2`). |
| **min_samples_leaf** | Minimum samples required at a leaf node (default: `1`). |
| **random_state** | Seed for reproducibility (e.g., `42`). |
| **confidence_threshold** | Float in `[0, 1]` (default: `0.8`). Threshold for assigning pseudo-labels. |

---

### Example: Semi-Supervised Decision Tree on Synthetic Data
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from ssdt import SemiSupervisedDecisionTree

# Generate synthetic dataset
X, y = make_classification(
    n_samples=10000, n_features=50, n_informative=30, 
    n_redundant=10, n_classes=2, weights=[0.8, 0.2], 
    class_sep=0.5, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mask labels for semi-supervised learning (only 5% labeled)
y_train_semi = y_train.copy()
num_labeled = int(0.05 * len(y_train))
y_train_semi[num_labeled:] = -1  # unlabeled marked as -1

# Train SSDT
base_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
semi_tree = SemiSupervisedDecisionTree(base_tree, confidence_threshold=0.8)
semi_tree.fit(X_train[:num_labeled], y_train[:num_labeled], X_train[num_labeled:])

# Evaluate
y_pred = semi_tree.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print("Semi-Supervised Decision Tree Results:")
print(f"Accuracy: {acc:.4f}")
print(f"F1-Score: {f1:.4f}")
```

## Adaptive Latent Variable Clustering (ALVC)

### Algorithm Explanation
The **Adaptive Latent Variable Clustering (ALVC)** algorithm is similar to k-means, but designed for **adaptive latent variable modeling** with improved robustness.

1. **Initialization** – randomly select `n_clusters` data points as initial cluster centers.  
2. **Assignment Step** – assign each data point to the nearest cluster center using Euclidean distance.  
3. **Update Step** – update cluster centers as the mean of assigned points (if a cluster is empty, retain its previous center).  
4. **Convergence Check** – stop if the change in cluster centers is below `tol` or after `max_iter` iterations.  
5. **Prediction** – assign new data points to the nearest cluster center.  

**Advantages:** Simple, effective for well-separated clusters, and robust to empty clusters.  
**Limitations:** Sensitive to initial center selection; assumes spherical clusters.

---

### Parameters
| Parameter | Description |
|-----------|-------------|
| **n_clusters** | Integer. Number of clusters to form (e.g., `3`). |
| **max_iter** | Integer (default: `100`). Maximum number of iterations for convergence. |
| **tol** | Float (default: `1e-4`). Convergence tolerance for cluster center updates. |

---

### Example: ALVC on Synthetic Data
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from alvc import ALVC

# Generate synthetic dataset
n_samples = 300
n_features = 2
n_clusters = 3
X, y_true = make_blobs(
    n_samples=n_samples, n_features=n_features, 
    centers=n_clusters, cluster_std=1.0, random_state=42
)

# Initialize and fit ALVC
alvc = ALVC(n_clusters=n_clusters)
alvc.fit(X)

# Predict clusters
y_pred = alvc.predict(X)

# Plot results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50)
plt.scatter(alvc.cluster_centers_[:, 0], alvc.cluster_centers_[:, 1], 
            c='red', marker='x', s=200, linewidths=3)
plt.title("ALVC Clustering Demo")
plt.show()
```
