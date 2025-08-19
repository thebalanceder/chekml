import time
import numpy as np
from scipy.spatial.distance import cdist

class ContinuousNearestNeighbors:
    def __init__(self, kernel='gaussian', bandwidth=1.0):
        """
        Initialize Continuous Nearest Neighbors model.
        :param kernel: Kernel type ('gaussian' or 'epanechnikov')
        :param bandwidth: Kernel bandwidth
        """
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
        Predict class labels using weighted neighbor influence.
        :param X_test: Test data (n_samples, n_features)
        :return: Predicted labels
        """
        start_time = time.time()
        distances = cdist(X_test, self.X_train)
        weights = self._kernel_function(distances)
        predictions = []
        for i in range(len(X_test)):
            weighted_counts = {}
            for j, label in enumerate(self.y_train):
                weight = weights[i, j]
                weighted_counts[label] = weighted_counts.get(label, 0) + weight
            predictions.append(max(weighted_counts, key=weighted_counts.get))
        self.prediction_time = time.time() - start_time
        return np.array(predictions)
