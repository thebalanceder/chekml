## 1.ContinuousNearestNeighbors

### Algorithm Explanation
The Continuous Nearest Neighbors (CNN) model extends traditional KNN by using a kernel function to weigh neighbors continuously based on their distances, rather than a fixed k. It stores the training data during fit (lazy learning). During prediction, it computes distances to all training points, applies a kernel (Gaussian or Epanechnikov) to get weights, and selects the class with the highest weighted sum. This provides smoother decision boundaries and is suitable for datasets where neighbor influence decreases gradually with distance. Training time is negligible; prediction can be computationally intensive for large datasets.

### Parameters
| Parameter    | Description |
|--------------|-------------|
| **kernel**   | Type of kernel for weighting distances. Options: `'gaussian'` (bell-shaped, infinite support) or `'epanechnikov'` (parabolic, finite support). Defaults to `'gaussian'`. |
| **bandwidth** | Controls kernel spread (similar to sigma in Gaussian). Larger values smooth more; smaller values emphasize closer neighbors. Must be positive. Defaults to `1.0`. |

### Demonstration
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from chekml.neighbor.continuous_knn import ContinuousNearestNeighbors

np.random.seed(42)

X, y = make_classification(n_samples=1000, n_features=5, n_classes=3, n_informative=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=== ContinuousNearestNeighbors Demo ===")
cnn = ContinuousNearestNeighbors(kernel='gaussian', bandwidth=0.5)
cnn.fit(X_train, y_train)
y_pred = cnn.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Training Time: {cnn.training_time:.6f} seconds")
print(f"Prediction Time: {cnn.prediction_time:.6f} seconds")
print(f"Accuracy: {accuracy:.4f}")
```

## 2.HierarchicalNeighborNetwork

### Algorithm Explanation
The Hierarchical Neighbor Network (HNN) is a graph neural network (GNN) using GraphSAGE convolutions to aggregate neighbor information hierarchically: local (first layer), regional (second), and global (third). It processes graph-structured data where nodes have features and edges define neighborhoods. Training uses negative log-likelihood loss on node classifications. It's designed for tasks like node classification in graphs with multi-scale neighbor dependencies. Synthetic data is generated with random features, edges, and labels. Evaluation computes accuracy on the same graph (for demo purposes).

### Parameters
| Parameter           | Description |
|---------------------|-------------|
| **in_channels**     | Number of input features per node. Must match the data's feature dimension. |
| **hidden_channels** | Number of hidden features in intermediate layers. Larger values increase capacity but risk overfitting and computation cost. |
| **out_channels**    | Number of output classes for classification. |
| **epochs**          | Number of training epochs. Defaults to `100`. |
| **lr**              | Learning rate for Adam optimizer. Defaults to `0.01`. |

### Demonstration
```python
import torch
from chekml.neighbor.hnn import HierarchicalNeighborNetwork, generate_hierarchical_graph, train_hnn, evaluate_hnn

torch.manual_seed(42)

print("=== HierarchicalNeighborNetwork Demo ===")
hnn_data = generate_hierarchical_graph(num_nodes=100, num_edges=300, num_features=5, num_classes=3)
hnn_model = HierarchicalNeighborNetwork(in_channels=5, hidden_channels=8, out_channels=3)
train_time = train_hnn(hnn_model, hnn_data, epochs=100, lr=0.01)
_, accuracy, pred_time = evaluate_hnn(hnn_model, hnn_data)
print(f"Training Time: {train_time:.6f} seconds")
print(f"Prediction Time: {pred_time:.6f} seconds")
print(f"Accuracy: {accuracy:.4f}")
```

## 3.AdaptiveKNN

### Algorithm Explanation
Adaptive KNN dynamically adjusts the number of neighbors (**k**) for each test point based on local density, estimated from average distances to all training points. It clips k between **k_min** and **k_max**, then applies a kernel to weight the selected neighbors and votes by weighted majority.  
This allows the model to adapt to varying data densities (e.g., sparse regions use more neighbors).  
Like other KNN variants, it's lazy during fit and computationally intensive during prediction.  
It is particularly useful for imbalanced or clustered datasets.

### Parameters
| Parameter    | Description |
|--------------|-------------|
| **k_min**    | Minimum number of neighbors. Defaults to `3`. Ensures robustness with a lower bound. |
| **k_max**    | Maximum number of neighbors. Defaults to `15`. Prevents over-smoothing in dense regions. |
| **kernel**   | Weighting kernel: `'gaussian'` or `'epanechnikov'`. Defaults to `'gaussian'`. |
| **bandwidth**| Controls kernel spread. Larger values smooth more; smaller values emphasize close neighbors. Must be positive. Defaults to `1.0`. |

### Demonstration
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from chekml.neighbor.adaptive_knn import AdaptiveKNN

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic classification data
X, y = make_classification(n_samples=1000, n_features=5, n_classes=3, 
                           n_informative=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("=== AdaptiveKNN Demo ===")
aknn = AdaptiveKNN(k_min=3, k_max=15, kernel='gaussian', bandwidth=0.5)
aknn.fit(X_train, y_train)
y_pred = aknn.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print(f"Training Time: {aknn.training_time:.6f} seconds")
print(f"Prediction Time: {aknn.prediction_time:.6f} seconds")
print(f"Accuracy: {accuracy:.4f}")
```

## 4.GraphTransformerNeighbors (GTN)

### Algorithm Explanation
Graph Transformer Neighbors (GTN) combines **Graph Attention Networks (GAT)** for multi-head attention on neighbors with **Transformer convolutions** for global context.  
- The forward pass applies **GAT** for local attention.  
- Then applies **TransformerConv** for sequence-like/global modeling.  
- Training minimizes **negative log-likelihood loss (NLL)**.  

It is useful for **node classification** in graphs where **important neighbors should be attended differently**.

### Parameters
| Parameter          | Description |
|--------------------|-------------|
| **in_channels**    | Input node feature dimension. |
| **hidden_channels**| Hidden feature size. |
| **out_channels**   | Number of output classes. |
| **heads**          | Number of attention heads in GATConv (default `2`). More heads capture diverse neighbor aspects. |
| **epochs**         | Training epochs (default `100`). |
| **lr**             | Learning rate for Adam optimizer (default `0.01`). |

### Demonstration
```python
import torch
from chekml.neighbor.gtn import GraphTransformerNeighbors, generate_graph_data, train_gtn, evaluate_gtn

# Set random seed for reproducibility
torch.manual_seed(42)

print("=== GraphTransformerNeighbors Demo ===")
gtn_data = generate_graph_data(num_nodes=100, num_edges=300, num_features=5, num_classes=3)
gtn_model = GraphTransformerNeighbors(in_channels=5, hidden_channels=8, out_channels=3, heads=2)
train_time = train_gtn(gtn_model, gtn_data, epochs=100, lr=0.01)
_, accuracy, pred_time = evaluate_gtn(gtn_model, gtn_data)
print(f"Training Time: {train_time:.6f} seconds")
print(f"Prediction Time: {pred_time:.6f} seconds")
print(f"Accuracy: {accuracy:.4f}")
```

## 5.LearnedIndex (LI_NNS)

### Algorithm Explanation
The **Learned Index for Nearest Neighbor Search (LI-NNS)** uses a **neural network** to learn embeddings that approximate a sorted index for faster nearest neighbor searches.  
- A **Multi-Layer Perceptron (MLP)** maps inputs to embeddings.  
- After training with **cross-entropy loss**, embeddings are indexed using **FAISS** for efficient KNN search.  
- Predictions use **majority vote** from k nearest embeddings.  

This method is ideal for **high-dimensional data** where traditional indexes are less efficient.

### Parameters
| Parameter       | Description |
|-----------------|-------------|
| **input_dim**   | Input feature dimension. |
| **hidden_dim**  | Hidden layer size (default `32`). |
| **output_dim**  | Embedding dimension (default `16`). Lower dimensions = faster search, but may lose information. |
| **epochs**      | Training epochs (default `50`). |
| **batch_size**  | Batch size (default `64`). |
| **lr**          | Learning rate (default `0.01`). |
| **k**           | Number of neighbors for prediction (default `5`). |

### Demonstration
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from chekml.neighbor.li_nns import LearnedIndex, train_li_nns, build_faiss_index, predict_li_nns

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic classification data (larger for demo)
X, y = make_classification(n_samples=5000, n_features=10, n_classes=3, n_informative=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=== LearnedIndex (LI_NNS) Demo ===")
li_model = LearnedIndex(input_dim=10)
train_time = train_li_nns(li_model, X_train, y_train, epochs=50, batch_size=64, lr=0.01)
faiss_index = build_faiss_index(li_model, X_train)
y_pred, pred_time = predict_li_nns(li_model, faiss_index, X_test, y_train, k=5)
accuracy = np.mean(y_pred == y_test)
print(f"Training Time: {train_time:.6f} seconds")
print(f"Prediction Time: {pred_time:.6f} seconds")
print(f"Accuracy: {accuracy:.4f}")
```

## 6.HGNN (Hypergraph Neural Network)

### Algorithm Explanation
The **Hypergraph Neural Network (HGNN)** extends GNNs to **hypergraphs**, where **edges (hyperedges)** can connect **multiple nodes** simultaneously.  
- Uses **HypergraphConv layers** to aggregate over hyperedges.  
- Applies **two convolution layers** with ReLU activation, then outputs class probabilities.  
- Trained with **Negative Log-Likelihood (NLL) loss**.  

This model is suited for data with **group interactions** (e.g., co-authorship networks, group memberships).

### Parameters
| Parameter          | Description |
|--------------------|-------------|
| **in_channels**    | Input feature dimension per node. |
| **hidden_channels**| Hidden layer size. Controls model capacity. |
| **out_channels**   | Number of output classes for classification. |
| **epochs**         | Number of training epochs (default `100`). |
| **lr**             | Learning rate for Adam optimizer (default `0.01`). |

### Demonstration
```python
import torch
from chekml.neighbor.hgnn import HGNN, generate_hypergraph_data, train_hgnn, evaluate_hgnn

# Set random seed for reproducibility
torch.manual_seed(42)

print("=== HGNN Demo ===")
hgnn_data = generate_hypergraph_data(num_nodes=10, num_edges=5, num_features=5, num_classes=3)
hgnn_model = HGNN(in_channels=5, hidden_channels=10, out_channels=3)
train_time = train_hgnn(hgnn_model, hgnn_data, epochs=100, lr=0.01)
_, accuracy, pred_time = evaluate_hgnn(hgnn_model, hgnn_data)
print(f"Training Time: {train_time:.6f} seconds")
print(f"Prediction Time: {pred_time:.6f} seconds")
print(f"Accuracy: {accuracy:.4f}")
```

## 7.MetricLearningKNN

### Algorithm Explanation
The **Metric Learning KNN (ML-KNN)** improves standard KNN by **learning a distance metric** (e.g., Mahalanobis) that adapts to the data distribution.  
- Optionally applies **PCA** for dimensionality reduction.  
- Learns a **covariance-based metric** during training.  
- Predictions use neighbors selected by the learned metric, with **majority voting**.  

This makes KNN more effective on datasets with **correlated or scaled features**.

---

### Parameters
| Parameter          | Description |
|--------------------|-------------|
| **k**              | Number of neighbors (default `5`). |
| **metric**         | Distance metric (default `'mahalanobis'`). |
| **use_pca**        | Whether to apply PCA (default `False`). |
| **pca_components** | Number of PCA components if used (default `2`). |

---

### Demonstration
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from chekml.neighbor.ml_knn import MetricLearningKNN

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic classification data
X, y = make_classification(n_samples=1000, n_features=5, n_classes=3, n_informative=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=== MetricLearningKNN (ML_KNN) Demo ===")
mlknn = MetricLearningKNN(k=5, metric='mahalanobis', use_pca=True, pca_components=3)
mlknn.fit(X_train, y_train)
y_pred = mlknn.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Training Time: {mlknn.training_time:.6f} seconds")
print(f"Prediction Time: {mlknn.prediction_time:.6f} seconds")
print(f"Accuracy: {accuracy:.4f}")
```

## 8.BayesianKNN

### Algorithm Explanation
The **Bayesian KNN** extends traditional KNN by incorporating **prior class probabilities** into neighbor voting.  
- During prediction, it computes distances, selects **k neighbors**, and updates posterior probabilities:  

\[
Posterior \propto Prior \times Likelihood
\]

- The final class is the **argmax of normalized posterior probabilities**.  
- This makes the algorithm naturally handle **class imbalance** by adjusting predictions using priors.  
- Training is lazy; predictions can also provide **probabilistic outputs**.

### Parameters
| Parameter | Description |
|-----------|-------------|
| **k**     | Number of neighbors (default `5`). |
| **prior** | Optional dictionary of class priors. If `None`, priors are computed empirically from the training data. |

### Demonstration
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from chekml.neighbor.bayesian_knn import BayesianKNN

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic classification data
X, y = make_classification(n_samples=1000, n_features=5, n_classes=3, n_informative=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=== BayesianKNN Demo ===")
bknn = BayesianKNN(k=5)
bknn.fit(X_train, y_train)
y_pred = bknn.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Training Time: {bknn.training_time:.6f} seconds")
print(f"Prediction Time: {bknn.prediction_time:.6f} seconds")
print(f"Accuracy: {accuracy:.4f}")
```

## 9.NeuralEmbedding (NNE)

### Algorithm Explanation
The **Neural Embedding (NNE)** model uses a **multi-layer perceptron (MLP)** to learn **low-dimensional embeddings** of input data, which are then used for nearest neighbor search.  
- The MLP has **two linear layers** with ReLU activation.  
- Trained with **cross-entropy loss** to predict class labels.  
- After training, embeddings are indexed using **FAISS** for efficient KNN search.  
- Predictions are made by finding **k nearest neighbors** in the embedding space and applying **majority voting**.  

This approach is effective for **high-dimensional data**, where direct KNN is expensive, since embeddings reduce dimensionality while preserving discriminative features.

### Parameters
| Parameter       | Description |
|-----------------|-------------|
| **input_dim**   | Number of input features (dimension of data). |
| **hidden_dim**  | Number of units in the hidden layer (default `32`). Higher values increase capacity but risk overfitting. |
| **output_dim**  | Dimension of learned embeddings (default `16`). Smaller values reduce cost but may lose information. |
| **epochs**      | Training epochs (default `50`). More improves convergence but increases training time. |
| **batch_size**  | Training batch size (default `64`). Smaller batches stabilize training but slow it down. |
| **lr**          | Learning rate for Adam optimizer (default `0.01`). Controls training speed and stability. |
| **k**           | Number of neighbors used in prediction (default `5`). |

### Demonstration
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from chekml.neighbor.nne import NeuralEmbedding, train_nne, build_faiss_index, predict_nne

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic classification data (larger for demo)
X, y = make_classification(n_samples=5000, n_features=10, n_classes=3, n_informative=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=== NeuralEmbedding (NNE) Demo ===")
ne_model = NeuralEmbedding(input_dim=10, hidden_dim=32, output_dim=16)
train_time = train_nne(ne_model, X_train, y_train, epochs=50, batch_size=64, lr=0.01)
faiss_index = build_faiss_index(ne_model, X_train)
y_pred, pred_time = predict_nne(ne_model, faiss_index, X_test, y_train, k=5)
accuracy = np.mean(y_pred == y_test)
print(f"Training Time: {train_time:.6f} seconds")
print(f"Prediction Time: {pred_time:.6f} seconds")
print(f"Accuracy: {accuracy:.4f}")
```

## 10.SelfSupervisedNeighborNetwork (S2_N2)

### Algorithm Explanation
The **SelfSupervisedNeighborNetwork (S2_N2)** applies **contrastive learning** to train an encoder without labels, using the **NT-Xent loss**.  
- Similar samples (augmented pairs) are pulled closer in the embedding space.  
- The encoder can be:
  - **MLP** for tabular data  
  - **CNN** for images  
- After training, embeddings can be:
  - Classified using **KNN**  
  - Clustered with **KMeans** or **DBSCAN**  

This method is especially effective when **labeled data is limited**, since it leverages unlabeled data for representation learning.

### Parameters
| Parameter        | Description |
|------------------|-------------|
| **model_type**   | Encoder type (`'mlp'` for tabular, `'cnn'` for images). Default: `'mlp'`. |
| **input_dim**    | Input feature dimension (for MLP). Default: `128`. Ignored for CNN. |
| **embedding_dim**| Dimension of output embeddings. Default: `128`. |
| **temperature**  | Temperature parameter for NT-Xent loss. Default: `0.5`. Controls softmax sharpness. |
| **epochs**       | Training epochs. Default: `10`. |
| **batch_size**   | Contrastive dataloader batch size. Default: `32`. |
| **n_clusters**   | Number of clusters for KMeans. Default: `10`. |
| **method**       | Clustering method (`'kmeans'` or `'dbscan'`). Default: `'kmeans'`. |

### Demonstration
```python
import numpy as np
import torch
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, TensorDataset
from chekml.neighbor.s2_n2 import SelfSupervisedNeighborNetwork, create_contrastive_dataloader, cluster_embeddings

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic classification data
X, y = make_classification(n_samples=1000, n_features=5, n_classes=3, n_informative=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=== SelfSupervisedNeighborNetwork (S2_N2) with KMeans Demo ===")
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# Create contrastive dataloader
dataloader = create_contrastive_dataloader(X_train, batch_size=32)

# Initialize model
s2_model = SelfSupervisedNeighborNetwork(model_type="mlp", input_dim=5, embedding_dim=128, temperature=0.5)

# Train with contrastive learning
for epoch, loss in enumerate(s2_model.train(dataloader, epochs=10), 1):
    print(f"Epoch {epoch} | Loss: {loss:.4f}")
train_time = s2_model.training_time

# Extract embeddings
train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train)), batch_size=32)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test)), batch_size=32)
train_embeddings = s2_model.get_embeddings(train_loader)
test_embeddings = s2_model.get_embeddings(test_loader)

# KNN classifier on embeddings
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_embeddings, y_train)
start_time = time.time()
y_pred = knn.predict(test_embeddings)
pred_time = time.time() - start_time

# Evaluate
accuracy = np.mean(y_pred == y_test)
cluster_labels = cluster_embeddings(train_embeddings, method="kmeans", n_clusters=10)
print(f"Training Time: {train_time:.6f} seconds")
print(f"Prediction Time: {pred_time:.6f} seconds")
print(f"Accuracy: {accuracy:.4f}")
print(f"Cluster Counts: {np.unique(cluster_labels, return_counts=True)}")
```

## 11.SelfLearningKNN (SL_KNN)

### Algorithm Explanation
The **SelfLearningKNN (SL_KNN)** is a **semi-supervised KNN** that iteratively labels unlabeled data to expand the training set.  
- It starts with labeled data.  
- Predicts pseudo-labels for unlabeled samples using KNN.  
- Adds only **high-confidence predictions** (above a threshold) to the training set.  
- Repeats for a fixed number of iterations or until no unlabeled data remains.  

Final predictions are made using standard KNN on the augmented dataset.  
This is useful when **labeled data is scarce but unlabeled data is abundant**.

### Parameters
| Parameter              | Description |
|------------------------|-------------|
| **k**                  | Number of neighbors for voting. Default: `5`. |
| **confidence_threshold** | Minimum confidence for accepting pseudo-labels. Default: `0.8`. Higher values increase reliability but reduce augmentation. |
| **max_iterations**     | Maximum number of self-learning iterations. Default: `10`. Prevents infinite loops. |

### Demonstration
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from chekml.neighbor.sl_knn import SelfLearningKNN

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic classification data
X, y = make_classification(n_samples=1000, n_features=5, n_classes=3, n_informative=3, random_state=42)

# Split into labeled and unlabeled data
X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X, y, test_size=0.7, random_state=42)
X_sl_train, X_sl_test, y_sl_train, y_sl_test = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=42)

print("=== SelfLearningKNN (SL_KNN) Demo ===")
slknn = SelfLearningKNN(k=5, confidence_threshold=0.8, max_iterations=10)

# Train with self-learning
slknn.fit(X_sl_train, y_sl_train, X_unlabeled)

# Predict on test set
y_pred = slknn.predict(X_sl_test)

# Evaluate
accuracy = np.mean(y_pred == y_sl_test)
print(f"Training Time: {slknn.training_time:.6f} seconds")
print(f"Prediction Time: {slknn.prediction_time:.6f} seconds")
print(f"Accuracy: {accuracy:.4f}")
```
