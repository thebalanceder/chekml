import torch
import torch.nn.functional as F
import numpy as np

class DDBC:
    def __init__(self, bandwidth=0.2, reachability_threshold=0.5):
        """
        Initialize Differentiable Density-Based Clustering.
        :param bandwidth: Bandwidth for Gaussian Kernel Density Estimation
        :param reachability_threshold: Soft threshold for reachability function
        """
        self.bandwidth = bandwidth
        self.reachability_threshold = reachability_threshold

    def _kernel_density_estimate(self, X):
        """
        Compute kernel density estimate.
        :param X: (N, D) tensor of data points
        :return: (N,) tensor of density values
        """
        N = X.shape[0]
        pairwise_dists = torch.cdist(X, X, p=2)
        density = torch.exp(- (pairwise_dists ** 2) / (2 * self.bandwidth ** 2)).sum(dim=1)
        return density / (N * (self.bandwidth * np.sqrt(2 * np.pi)))

    def _reachability_matrix(self, X, density):
        """
        Compute differentiable reachability matrix.
        :param X: (N, D) tensor of data points
        :param density: (N,) tensor of density estimates
        :return: (N, N) soft reachability matrix
        """
        pairwise_dists = torch.cdist(X, X, p=2)
        density_diffs = density[:, None] - density[None, :]
        reachability = torch.sigmoid(-pairwise_dists + self.reachability_threshold) * torch.sigmoid(density_diffs)
        return reachability

    def _soft_clustering(self, reachability):
        """
        Compute soft cluster assignment using power iteration.
        :param reachability: (N, N) soft reachability matrix
        :return: (N,) soft cluster assignments
        """
        N = reachability.shape[0]
        soft_labels = torch.ones(N, device=reachability.device) / N
        for _ in range(10):
            soft_labels = reachability @ soft_labels
            soft_labels /= soft_labels.max()
        return soft_labels

    def fit_predict(self, X):
        """
        Fit and predict cluster labels.
        :param X: Numpy array of shape (n_samples, n_features)
        :return: Soft cluster labels as numpy array
        """
        X = torch.tensor(X, dtype=torch.float32)
        density = self._kernel_density_estimate(X)
        reachability = self._reachability_matrix(X, density)
        soft_labels = self._soft_clustering(reachability)
        return soft_labels.detach().numpy()
