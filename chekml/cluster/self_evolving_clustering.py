import numpy as np

class SelfEvolvingClustering:
    def __init__(self, distance_threshold=1.0, learning_rate=0.1, merge_threshold=0.5):
        """
        Initialize Self-Evolving Clustering.
        :param distance_threshold: Max distance to assign point to existing cluster
        :param learning_rate: Rate for updating cluster centroids
        :param merge_threshold: Distance below which clusters are merged
        """
        self.distance_threshold = distance_threshold
        self.learning_rate = learning_rate
        self.merge_threshold = merge_threshold
        self.clusters = []

    def _find_nearest_cluster(self, point):
        """
        Find nearest cluster centroid.
        :param point: Input data point
        :return: Tuple of (nearest cluster index, distance)
        """
        if not self.clusters:
            return None, float('inf')
        clusters_array = np.array(self.clusters)
        distances = np.linalg.norm(clusters_array - point, axis=1)
        nearest_idx = np.argmin(distances)
        return nearest_idx, distances[nearest_idx]

    def _merge_clusters(self):
        """
        Merge clusters closer than merge_threshold.
        """
        if len(self.clusters) < 2:
            return
        new_clusters = []
        merged = set()
        clusters_array = np.array(self.clusters)
        for i in range(len(clusters_array)):
            if i in merged:
                continue
            for j in range(i + 1, len(clusters_array)):
                if np.linalg.norm(clusters_array[i] - clusters_array[j]) < self.merge_threshold:
                    clusters_array[i] = (clusters_array[i] + clusters_array[j]) / 2
                    merged.add(j)
            new_clusters.append(clusters_array[i])
        self.clusters = new_clusters

    def fit(self, data):
        """
        Fit clustering model on data.
        :param data: Numpy array of shape (n_samples, n_features)
        :return: Self
        """
        for point in data:
            nearest_idx, min_distance = self._find_nearest_cluster(point)
            if min_distance < self.distance_threshold:
                self.clusters[nearest_idx] += self.learning_rate * (point - self.clusters[nearest_idx])
            else:
                self.clusters.append(point)
            self._merge_clusters()
        return self

    def predict(self, data):
        """
        Predict cluster labels for data.
        :param data: Numpy array of shape (n_samples, n_features)
        :return: Cluster labels
        """
        labels = []
        for point in data:
            nearest_idx, _ = self._find_nearest_cluster(point)
            labels.append(nearest_idx)
        return np.array(labels)
