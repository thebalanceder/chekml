import numpy as np
from scipy.special import factorial

class PoissonMixtureModel:
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        """
        Initialize Poisson Mixture Model.
        :param n_components: Number of mixture components
        :param max_iter: Maximum number of EM iterations
        :param tol: Convergence tolerance for log-likelihood
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.lambdas = None

    def _poisson_pmf(self, X, lambdas):
        """
        Compute Poisson probability mass function.
        :param X: Input data
        :param lambdas: Poisson parameters
        :return: PMF values
        """
        return np.exp(-lambdas) * (lambdas ** X) / factorial(X)

    def fit(self, X):
        """
        Fit the PMM on input data using EM algorithm.
        :param X: Numpy array of shape (n_samples, n_features)
        :return: Self
        """
        n, d = X.shape
        self.weights = np.ones(self.n_components) / self.n_components
        self.lambdas = np.random.uniform(low=1, high=np.max(X), size=(self.n_components, d))

        log_likelihoods = []
        for _ in range(self.max_iter):
            likelihoods = np.array([
                self.weights[k] * np.prod(self._poisson_pmf(X, self.lambdas[k]), axis=1)
                for k in range(self.n_components)
            ]).T
            responsibilities = likelihoods / likelihoods.sum(axis=1, keepdims=True)

            Nk = responsibilities.sum(axis=0)
            self.weights = Nk / n
            self.lambdas = (responsibilities.T @ X) / Nk[:, None]

            log_likelihood = np.sum(np.log(likelihoods.sum(axis=1)))
            log_likelihoods.append(log_likelihood)
            if len(log_likelihoods) > 1 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < self.tol:
                break

        return self

    def predict(self, X):
        """
        Predict cluster labels for input data.
        :param X: Numpy array of shape (n_samples, n_features)
        :return: Cluster labels
        """
        likelihoods = np.array([
            self.weights[k] * np.prod(self._poisson_pmf(X, self.lambdas[k]), axis=1)
            for k in range(self.n_components)
        ]).T
        return np.argmax(likelihoods, axis=1)

    def compute_log_likelihood(self, X):
        """
        Compute log-likelihood of data given the model.
        :param X: Numpy array of shape (n_samples, n_features)
        :return: Log-likelihood value
        """
        likelihood = np.zeros(X.shape[0])
        for k in range(self.n_components):
            likelihood += self.weights[k] * np.prod(self._poisson_pmf(X, self.lambdas[k]), axis=1)
        return np.sum(np.log(likelihood))
