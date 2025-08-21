import time
import numpy as np
from scipy.spatial.distance import cdist

class BayesianKNN:
    def __init__(self, k=5, prior=None):
        """
        Initialize Bayesian KNN model.
        :param k: Number of neighbors
        :param prior: Dictionary of prior class probabilities (optional)
        """
        self.k = k
        self.prior = prior
        self.X_train = None
        self.y_train = None
        self.classes = None
        self.training_time = None
        self.prediction_time = None

    def fit(self, X, y):
        """
        Store training data and compute prior probabilities.
        :param X: Training data (n_samples, n_features)
        :param y: Training labels
        :return: Self
        """
        start_time = time.time()
        self.X_train = X
        self.y_train = y
        self.classes, class_counts = np.unique(y, return_counts=True)
        if self.prior is None:
            self.prior = {c: class_counts[i] / len(y) for i, c in enumerate(self.classes)}
        self.training_time = time.time() - start_time
        return self

    def predict_proba(self, X_test):
        """
        Predict class probabilities using Bayesian inference.
        :param X_test: Test data (n_samples, n_features)
        :return: List of dictionaries with class probabilities
        """
        start_time = time.time()
        distances = cdist(X_test, self.X_train)
        sorted_indices = np.argsort(distances, axis=1)[:, :self.k]
        nearest_labels = self.y_train[sorted_indices]
        probabilities = []
        for labels in nearest_labels:
            class_probs = {c: self.prior.get(c, 1e-6) for c in self.classes}
            unique, counts = np.unique(labels, return_counts=True)
            for i, c in enumerate(unique):
                class_probs[c] *= counts[i] / self.k
            total_prob = sum(class_probs.values())
            class_probs = {c: class_probs[c] / total_prob for c in class_probs}
            probabilities.append(class_probs)
        self.prediction_time = time.time() - start_time
        return probabilities

    def predict(self, X_test):
        """
        Predict class labels using Bayesian inference.
        :param X_test: Test data (n_samples, n_features)
        :return: Predicted labels
        """
        probabilities = self.predict_proba(X_test)
        return np.array([max(prob.keys(), key=lambda c: prob[c]) for prob in probabilities])
