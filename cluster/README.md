 # Algorithm overview and usage
 ## 1. Self Evolving Clustering
 - Description: Self-Evolving Clustering is an online clustering algorithm that incrementally updates cluster centroids based on incoming data points.
 - Parameters:
   - `distance_threshold` (float,default=1.0) :Maximum distance to assign a point to an existing cluster. Larger values allow points to join clusters farther away.
   - `learning_rate` (float, default=0.1) :Rate at which cluster centroids are updated toward new ponits. Higher values make clusters adapt faster.
   -  `merge_threshold` (float,default=0.5) :Distance below which two clusters are merged. Smaller values result in fewer merges.
 - Algorithm Explanation:
   - For each data point, find the nearest cluster centroid.
   - If the distance to the nearest centroid is less than `distance_threshold` ,update the centroid using the learning rate.
   - Otherwise, create a new cluster with the point as its centroid.
   - Merge clusters whose centroids are closer than `merge_threshold`.
  
  - Usage Example:

**Python**
```python
from chekml.cluster.self_evolving_clustering import SelfEvolvingClustering
import numpy as np
sec = SelfEvolvingClustering(distance_threshold=0.2,learning_rate=0.05,merge_threshold=0.1)
sec.fit(data)
labels = sec.predict(data)
print(f"Number of clusters: {len(sec.cluster)})"
```

 ## 2. Transformer Based Clustering
 - Description: This algorithm uses a transformer model (e.g. BERT) to generate embeddings for text data, followed by KMeans clustering. It is ideal for clustering text documents based on semantic similarity.
 - Parameters:
   - `model_name` (str,default="bert-base-uncased",num_clusters=2)
   - `num_clusters` (int,default=2) :Number of clusters for KMeans. Must be set based on the expected number of groups.
 - Algorithm Explanation:
   - Tokenize and encode input text using the specified transformer model to obtain embeddings.
   - Apply KMeans clustering to the embeddings to assign cluster labels.
  
   - Usage Example:

  **Python**
```python
from transformer_clustering import TransformerBasedClustering
documents = ["Machine learning is amazing!","Deep learning drives AI progress"]
tbc=TransformerBasedClustering(model_name="bert-base-uncased",num_clusters=2)
labels,embeddings=tbc.fit_predict(documents)
reduced_embeddings=tbc.visualize(embeddings,labels)
print("CLuster labels:",labels)
```

 ## 3. DDBC(Differentiable Density-based Clustering)
 - Description: DDBC is a density-based clustering algorithm that uses a differentiable approach with Gaussian kernel density estimation and a soft reachability matrix. It produces soft cluster assignments, suitable for complex data distributions.
 - Parameters:
   - `bandwidth` (float,default=0.2) :Bandwidth for Gaussian kernel density estimation. Smaller values make the density estimate more sensitive to local variations.
   - `reachability_threshold` (float,default=0.5) :Threshold for the soft reachability function. Higher values make clusters less connected
 - Algorithm Explanations:
   - Compute kernel density estimates for each data point using a Gaussian kernel.
   - Construct a soft reachability matrix based on pairwise distances and density differences.
   - Use iteration to compute soft cluster, allow points to belong to multiple clusters with many probability states.
  
   - Usage Example:

**Python**
```python
from chekml.cluster.ddbc_clustering import DDBC
from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=300,noise=0.1,random_state=42)
ddbc = DDBC(bandwidth=0.2,reachability_threshold=0.5)
soft_labels = ddbc.fit_predict(X)
print("Soft labels(first 10):",soft_labels[:10]) 
```

 ## 4. Adaptive Graph Clustering
 - Description: This algorithm constructs a graph based on k-nearest neighbors, computes a normalized graph Laplacian, performs spectral clustering with an adaptive number of clusters determined by eigenvalue gaps. 
 - Parameters:
   - `k` (int,default=5) :Number of nearest neighbors to construct the affinity matrix. Higher values increase connectivtiy in the graph.
 - Algorithm Explanations:
   - build an affinity matrix using k-nearest neighbors and a Gaussian kernel based on distances
   - Compute the normalized graph Laplacian.
   - Calculate eigenvalues and eigenvectors, then estimate the number of clusters using the eigenvalue gap.
   - Apply KMeans to the eigenvectors to obtain cluster labels
   - Evaluate clustering quality using Adjusted Rand Index(ARI) and Normalized Mutual Information(NMI) if ground truth labels are available.
    
   - Usage Example:

**Python**
```python
from chekml.cluster.adaptive_graph_clustering import AdaptiveGraphClustering
from sklearn.datasets import load_iris

iris = load_iris()
data = iris.data
true_labels = iris.target
agc = AdaptiveGraphClustering(k=5)
labels = agc.fit_predict(data)
ari,nmi = agc.evaluate(true_labels, labels)
print("Cluster labels:",labels[:10])
print(f"ARI:{ari:.4f},NMI:{nmi:.4f}")
```
