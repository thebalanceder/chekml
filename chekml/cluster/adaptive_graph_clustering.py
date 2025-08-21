import numpy as np
from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

class AdaptiveGraphClustering:
    def __init__(self, k=5):
        """
        Initialize Adaptive Graph Clustering.
        :param k: Number of nearest neighbors
        """
        self.k = k

    def _construct_affinity_matrix(self, data):
        """
        Construct affinity matrix using k-nearest neighbors and Gaussian kernel.
        :param data: Numpy array of shape (n_samples, n_features)
        :return: Affinity matrix
        """
        n_samples = data.shape[0]
        affinity_matrix = np.zeros((n_samples, n_samples))
        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree').fit(data)
        distances, indices = nbrs.kneighbors(data)
        for i in range(n_samples):
            for j, dist in zip(indices[i], distances[i]):
                if i != j:
                    sigma = distances[i].mean()
                    affinity_matrix[i, j] = np.exp(-dist**2 / (2 * sigma**2))
                    affinity_matrix[j, i] = affinity_matrix[i, j]
        return affinity_matrix

    def _compute_laplacian(self, affinity_matrix):
        """
        Compute normalized graph Laplacian.
        :param affinity_matrix: Numpy array of shape (n_samples, n_samples)
        :return: Laplacian matrix
        """
        degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
        sqrt_degree_matrix = np.sqrt(degree_matrix)
        sqrt_degree_matrix_inv = np.linalg.inv(sqrt_degree_matrix)
        laplacian = np.eye(affinity_matrix.shape[0]) - np.dot(np.dot(sqrt_degree_matrix_inv, affinity_matrix), sqrt_degree_matrix_inv)
        return laplacian

    def _adaptive_thresholding(self, eigenvalues):
        """
        Determine number of clusters using eigenvalue gap.
        :param eigenvalues: Numpy array of eigenvalues
        :return: Number of clusters
        """
        diff = np.diff(eigenvalues)
        return np.argmax(diff) + 1

    def fit_predict(self, data):
        """
        Perform adaptive graph clustering.
        :param data: Numpy array of shape (n_samples, n_features)
        :return: Cluster labels
        """
        affinity_matrix = self._construct_affinity_matrix(data)
        laplacian = self._compute_laplacian(affinity_matrix)
        eigenvalues, eigenvectors = eigh(laplacian, subset_by_index=[0, min(10, data.shape[0] - 1)])
        n_clusters = self._adaptive_thresholding(eigenvalues)
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(eigenvectors[:, :n_clusters])
        return labels

    def evaluate(self, true_labels, predicted_labels):
        """
        Evaluate clustering using ARI and NMI.
        :param true_labels: Ground truth labels
        :param predicted_labels: Predicted cluster labels
        :return: Tuple of (ARI, NMI)
        """
        ari = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        return ari, nmi
