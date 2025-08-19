import time
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance

class MetricLearningKNN:
    def __init__(self, k=5, metric='mahalanobis', use_pca=False, pca_components=2):
        """
        Initialize Metric Learning KNN model.
        :param k: Number of neighbors
        :param metric: Distance metric ('mahalanobis' or others from cdist)
        :param use_pca: Whether to use PCA for dimensionality reduction
        :param pca_components: Number of PCA components
        """
        self.k = k
        self.metric = metric
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.metric_matrix = None
        self.pca = None
        self.X_train = None
        self.y_train = None
        self.training_time = None
        self.prediction_time = None

    def fit(self, X, y):
        """
        Fit the model with metric learning.
        :param X: Training data
        :param y: Training labels
        :return: Self
        """
        start_time = time.time()
        self.X_train = X
        self.y_train = y
        if self.use_pca:
            self.pca = PCA(n_components=self.pca_components)
            self.X_train = self.pca.fit_transform(X)
        if self.metric == 'mahalanobis':
            cov_estimator = EmpiricalCovariance()
            cov_estimator.fit(self.X_train)
            self.metric_matrix = np.linalg.inv(cov_estimator.covariance_)
        self.training_time = time.time() - start_time
        return self

    def predict(self, X_test):
        """
        Predict class labels.
        :param X_test: Test data
        :return: Predicted labels
        """
        start_time = time.time()
        if self.use_pca:
            X_test = self.pca.transform(X_test)
        if self.metric == 'mahalanobis':
            distances = cdist(X_test, self.X_train, metric='mahalanobis', VI=self.metric_matrix)
        else:
            distances = cdist(X_test, self.X_train, metric=self.metric)
        sorted_indices = np.argsort(distances, axis=1)[:, :self.k]
        nearest_labels = self.y_train[sorted_indices]
        predictions = []
        for labels in nearest_labels:
            unique, counts = np.unique(labels, return_counts=True)
            predictions.append(unique[np.argmax(counts)])
        self.prediction_time = time.time() - start_time
        return np.array(predictions)
