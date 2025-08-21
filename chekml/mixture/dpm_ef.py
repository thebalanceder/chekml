import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

class DPMEF:
    def __init__(self, num_clusters=10, alpha=1.0):
        """
        Initialize Dirichlet Process Mixture of Exponential Families.
        :param num_clusters: Number of clusters for stick-breaking process
        :param alpha: Concentration parameter for Dirichlet Process
        """
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.weights = None
        self.means = None
        self.variances = None
        self.scaler = StandardScaler()

    def _stick_breaking(self):
        """
        Generate stick-breaking process weights.
        :return: Normalized weights
        """
        betas = np.random.beta(1, self.alpha, self.num_clusters)
        remaining_stick = np.cumprod(1 - betas[:-1])
        weights = np.concatenate(([betas[0]], betas[1:] * remaining_stick))
        return weights / weights.sum()

    def fit(self, X):
        """
        Fit the DPM-EF on input data.
        :param X: Numpy array of shape (n_samples, n_features)
        :return: Self
        """
        X_scaled = self.scaler.fit_transform(X)
        self.weights = self._stick_breaking()
        self.means = np.random.normal(loc=0, scale=1, size=(self.num_clusters, X_scaled.shape[1]))
        self.variances = np.random.gamma(shape=2, scale=2, size=(self.num_clusters, X_scaled.shape[1]))

        responsibilities = np.zeros((X_scaled.shape[0], self.num_clusters))
        for k in range(self.num_clusters):
            likelihood = stats.norm.pdf(X_scaled, loc=self.means[k], scale=np.sqrt(self.variances[k]))
            responsibilities[:, k] = self.weights[k] * np.prod(likelihood, axis=1)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        self.labels = np.argmax(responsibilities, axis=1)
        return self

    def predict(self, X):
        """
        Predict cluster labels for input data.
        :param X: Numpy array of shape (n_samples, n_features)
        :return: Cluster labels
        """
        X_scaled = self.scaler.transform(X)
        responsibilities = np.zeros((X_scaled.shape[0], self.num_clusters))
        for k in range(self.num_clusters):
            likelihood = stats.norm.pdf(X_scaled, loc=self.means[k], scale=np.sqrt(self.variances[k]))
            responsibilities[:, k] = self.weights[k] * np.prod(likelihood, axis=1)
        return np.argmax(responsibilities, axis=1)

    def compute_log_likelihood(self, X):
        """
        Compute log-likelihood of data given the model.
        :param X: Numpy array of shape (n_samples, n_features)
        :return: Log-likelihood value
        """
        X_scaled = self.scaler.transform(X)
        log_likelihood = 0
        for k in range(self.num_clusters):
            log_likelihood += self.weights[k] * np.prod(stats.norm.pdf(X_scaled, loc=self.means[k], scale=np.sqrt(self.variances[k])), axis=1)
        return np.sum(np.log(log_likelihood))
