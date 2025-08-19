import time
import numpy as np
from scipy.spatial.distance import cdist

class SelfLearningKNN:
    def __init__(self, k=5, confidence_threshold=0.8, max_iterations=10):
        """
        Initialize Self-Learning KNN model.
        :param k: Number of neighbors
        :param confidence_threshold: Confidence threshold for pseudo-labeling
        :param max_iterations: Maximum iterations for self-learning
        """
        self.k = k
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.X_train = None
        self.y_train = None
        self.training_time = None
        self.prediction_time = None

    def fit(self, X_labeled, y_labeled, X_unlabeled):
        """
        Fit with self-learning on unlabeled data.
        :param X_labeled: Labeled training data
        :param y_labeled: Labeled training labels
        :param X_unlabeled: Unlabeled data
        :return: Self
        """
        start_time = time.time()
        self.X_train = X_labeled
        self.y_train = y_labeled
        X_unlabeled_copy = X_unlabeled.copy()
        for iteration in range(self.max_iterations):
            if len(X_unlabeled_copy) == 0:
                break
            distances = cdist(X_unlabeled_copy, self.X_train)
            sorted_indices = np.argsort(distances, axis=1)[:, :self.k]
            nearest_labels = self.y_train[sorted_indices]
            predictions = []
            confidences = []
            for labels in nearest_labels:
                unique, counts = np.unique(labels, return_counts=True)
                majority_class = unique[np.argmax(counts)]
                confidence = np.max(counts) / self.k
                predictions.append(majority_class)
                confidences.append(confidence)
            predictions = np.array(predictions)
            confidences = np.array(confidences)
            confident_mask = confidences >= self.confidence_threshold
            new_X = X_unlabeled_copy[confident_mask]
            new_y = predictions[confident_mask]
            X_unlabeled_copy = X_unlabeled_copy[~confident_mask]
            self.X_train = np.vstack((self.X_train, new_X))
            self.y_train = np.hstack((self.y_train, new_y))
        self.training_time = time.time() - start_time
        return self

    def predict(self, X_test):
        """
        Predict class labels.
        :param X_test: Test data
        :return: Predicted labels
        """
        start_time = time.time()
        distances = cdist(X_test, self.X_train)
        sorted_indices = np.argsort(distances, axis=1)[:, :self.k]
        nearest_labels = self.y_train[sorted_indices]
        predictions = []
        for labels in nearest_labels:
            unique, counts = np.unique(labels, return_counts=True)
            predictions.append(unique[np.argmax(counts)])
        self.prediction_time = time.time() - start_time
        return np.array(predictions)
