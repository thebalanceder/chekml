 # Algorithm overview and usage
 ## 1. SelfEvolvingClustering
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
  
  - Usage Example

```python
from chekml.cluster.self_evolving_clustering import SelfEvolvingClustering
import numpy as np
sec = SelfEvolvingClustering(distance_threshold=0.2,learning_rate=0.05,merge_threshold=0.1)
sec.fit(data)
labels = sec.predict(data)
print(f"Number of clusters: {len(sec.cluster)})"
```

 ## 2. TransformerBasedClustering
 - Description: This algorithm uses a transformer model (e.g. BERT) to generate embeddings for text data, followed by KMeans clustering. It is ideal for clustering text documents based on semantic similarity.
 - Parameters:
   - 'model_name' (str,default="bert-base-uncased",num_clusters=2)
   - 'num_clusters' (int,default=2) :Number of clusters for KMeans. Must be set based on the expected number of groups.
 - Algorithm Explanation:
   -
