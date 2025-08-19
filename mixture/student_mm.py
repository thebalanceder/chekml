import numpy as np
from scipy.special import gammaln
from sklearn.preprocessing import StandardScaler

class StudentMixtureModel:
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        """
        Initialize Student-t Mixture Model.
        :param n_components: Number of mixture components
        :param max_iter: Maximum number of EM iterations
        :param tol: Convergence tolerance for log-likelihood
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.means = None
        self.covs = None
        self.df = None
        self.scaler = StandardScaler()

    def _student_t_pdf(self, X, mean, cov, df):
        """
        Compute Student-t probability density function.
        :param X: Input data
        :param mean: Mean vector
        :param cov: Covariance matrix
        :param df: Degrees of freedom
        :return: PDF values
        """
        d = X.shape[1]
        X_centered = X - mean
        inv_cov = np.linalg.inv(cov)
        mahalanobis = np.sum(X_centered @ inv_cov * X_centered, axis=1)
        norm_const = (
            np.exp(gammaln((df + d) / 2) - gammaln(df / 2))
            / ((df * np.pi) ** (d / 2) * np.linalg.det(cov) ** 0.5)
        )
        return norm_const * (1 + mahalanobis / df) ** (-(df + d) / 2)

    def fit(self, X):
        """
        Fit the SMM on input data using EM algorithm.
        :param X: Numpy array of shape (n_samples, n_features)
        :return: Self
        """
        X_scaled = self.scaler.fit_transform(X)
        n, d = X_scaled.shape
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X_scaled[np.random.choice(n, self.n_components, replace=False)]
        self.covs = np.array([np.cov(X_scaled.T) for _ in range(self.n_components)])
        self.df = np.ones(self.n_components) * 10

        log_likelihoods = []
        for _ in range(self.max_iter):
            likelihoods = np.array([
                self.weights[k] * self._student_t_pdf(X_scaled, self.means[k], self.covs[k], self.df[k])
                for k in range(self.n_components)
            ]).T
            responsibilities = likelihoods / likelihoods.sum(axis=1, keepdims=True)

            Nk = responsibilities.sum(axis=0)
            self.weights = Nk / n
            self.means = (responsibilities.T @ X_scaled) / Nk[:, None]
            for k in range(self.n_components):
                X_centered = X_scaled - self.means[k]
                weighted_cov = (responsibilities[:, k][:, None] * X_centered).T @ X_centered
                self.covs[k] = weighted_cov / Nk[k]
                df_k = self.df[k]
                for _ in range(5):
                    psi_term = np.sum(responsibilities[:, k] * (np.log(1 + (np.sum(X_centered @ np.linalg.inv(self.covs[k]) * X_centered, axis=1) / df_k))))
                    psi_derivative = -np.sum(responsibilities[:, k]) / (2 * df_k) + 0.5 * psi_term
                    df_k = max(df_k - psi_derivative, 1.1)
                self.df[k] = df_k

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
        X_scaled = self.scaler.transform(X)
        likelihoods = np.array([
            self.weights[k] * self._student_t_pdf(X_scaled, self.means[k], self.covs[k], self.df[k])
            for k in range(self.n_components)
        ]).T
        return np.argmax(likelihoods, axis=1)

    def compute_log_likelihood(self, X):
        """
        Compute log-likelihood of data given the model.
        :param X: Numpy array of shape (n_samples, n_features)
        :return: Log-likelihood value
        """
        X_scaled = self.scaler.transform(X)
        likelihood = np.zeros(X_scaled.shape[0])
        for k in range(self.n_components):
            likelihood += self.weights[k] * self._student_t_pdf(X_scaled, self.means[k], self.covs[k], self.df[k])
        return np.sum(np.log(likelihood))
