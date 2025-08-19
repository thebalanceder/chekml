import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

class CustomGMM:
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        """
        Initialize Custom Gaussian Mixture Model.
        :param n_components: Number of mixture components
        :param max_iter: Maximum number of EM iterations
        :param tol: Convergence tolerance for log-likelihood
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.trained = False
        self.weights = None
        self.means = None
        self.covs = None
        self.scaler = StandardScaler()

    def fit(self, X):
        """
        Fit the GMM on input data using EM algorithm.
        :param X: Numpy array of shape (n_samples, n_features)
        :return: Self
        """
        X_scaled = self.scaler.fit_transform(X)
        n, d = X_scaled.shape
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X_scaled[np.random.choice(n, self.n_components, replace=False)]
        self.covs = np.array([np.eye(d) for _ in range(self.n_components)])

        for _ in range(self.max_iter):
            likelihoods = np.array([
                stats.multivariate_normal.pdf(X_scaled, mean=self.means[k], cov=self.covs[k])
                for k in range(self.n_components)
            ]).T * self.weights
            responsibilities = likelihoods / likelihoods.sum(axis=1, keepdims=True)

            Nk = responsibilities.sum(axis=0)
            self.weights = Nk / n
            self.means = (responsibilities.T @ X_scaled) / Nk[:, None]
            self.covs = np.array([
                np.cov(X_scaled.T, aweights=responsibilities[:, k]) for k in range(self.n_components)
            ])

            log_likelihood = np.sum(np.log(likelihoods.sum(axis=1)))
            if self.trained and np.abs(log_likelihood - self.last_log_likelihood) < self.tol:
                break
            self.last_log_likelihood = log_likelihood
            self.trained = True

        return self

    def predict(self, X):
        """
        Predict cluster labels for input data.
        :param X: Numpy array of shape (n_samples, n_features)
        :return: Cluster labels
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before predicting.")
        X_scaled = self.scaler.transform(X)
        likelihoods = np.array([
            stats.multivariate_normal.pdf(X_scaled, mean=self.means[k], cov=self.covs[k])
            for k in range(self.n_components)
        ]).T * self.weights
        return np.argmax(likelihoods, axis=1)

    def compute_log_likelihood(self, X):
        """
        Compute log-likelihood of data given the model.
        :param X: Numpy array of shape (n_samples, n_features)
        :return: Log-likelihood value
        """
        if not self.trained:
            raise RuntimeError("Cannot compute likelihood on an untrained model.")
        X_scaled = self.scaler.transform(X)
        likelihood = np.zeros(X_scaled.shape[0])
        for k in range(self.n_components):
            likelihood += self.weights[k] * stats.multivariate_normal.pdf(X_scaled, mean=self.means[k], cov=self.covs[k])
        return np.sum(np.log(likelihood))
