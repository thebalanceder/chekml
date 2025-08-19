import time
import numpy as np
from scipy.spatial.distance import cdist

class AdaptiveKNN:
    def __init__(self, k_min=3, k_max=15, kernel='gaussian', bandwidth=1.0):
        """
        Initialize Adaptive KNN model.
        :param k_min: Minimum number of neighbors
        :param k_max: Maximum number of neighbors
        :param kernel: Kernel type ('gaussian' or 'epanechnikov')
        :param bandwidth: Kernel bandwidth
        """
        self.k_min = k_min
        self.k_max = k_max
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.X_train = None
        self.y_train = None
        self.training_time = None
        self.prediction_time = None

    def _kernel_function(self, distances):
        """
        Apply kernel to smooth neighbor weights.
        :param distances: Array of distances
        :return: Kernel weights
        """
        if self.kernel == 'gaussian':
            return np.exp(-0.5 * (distances / self.bandwidth) ** 2) / (self.bandwidth * np.sqrt(2 * np.pi))
        elif self.kernel == 'epanechnikov':
            return np.maximum(0, 0.75 * (1 - (distances / self.bandwidth) ** 2))
        else:
            raise ValueError("Unsupported kernel type")

    def _estimate_density(self, distances):
        """
        Estimate local density based on distances to neighbors.
        :param distances: Array of distances
        :return: Density estimates
        """
        return np.mean(distances, axis=1)

    def fit(self, X, y):
        """
        Store training data and record training time.
        :param X: Training data (n_samples, n_features)
        :param y: Training labels
        :return: Self
        """
        start_time = time.time()
        self.X_train = X
        self.y_train = y
        self.training_time = time.time() - start_time
        return self

    def predict(self, X_test):
        """
        Predict class labels using adaptive neighbor selection.
        :param X_test: Test data (n_samples, n_features)
        :return: Predicted labels
        """
        start_time = time.time()
        distances = cdist(X_test, self.X_train)
        density = self._estimate_density(distances)
        k_values = np.clip((self.k_max - density * (self.k_max - self.k_min)), self.k_min, self.k_max).astype(int)
        predictions = []
        for i in range(len(X_test)):
            sorted_indices = np.argsort(distances[i])[:k_values[i]]
            nearest_distances = distances[i, sorted_indices]
            nearest_labels = self.y_train[sorted_indices]
            weights = self._kernel_function(nearest_distances)
            weighted_counts = {}
            for j, label in enumerate(nearest_labels):
                weighted_counts[label] = weighted_counts.get(label, 0) + weights[j]
            predictions.append(max(weighted_counts, key=weighted_counts.get))
        self.prediction_time = time.time() - start_time
        return np.array(predictions)
