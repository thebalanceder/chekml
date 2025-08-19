import numpy as np
from scipy.stats import norm
from sklearn.metrics import pairwise_distances_argmin_min

class ALVC:
    """Adaptive Latent Variable Clustering (ALVC) model."""
    
    def __init__(self, n_clusters, max_iter=100, tol=1e-4):
        """Initialize ALVC model.
        
        Args:
            n_clusters (int): Number of clusters
            max_iter (int): Maximum number of iterations
            tol (float): Convergence tolerance
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X):
        """Fit the ALVC model to the data.
        
        Args:
            X (array-like): Input data of shape (n_samples, n_features)
            
        Returns:
            self: Fitted model
        """
        n_samples, n_features = X.shape

        # Initialize cluster centers randomly
        self.cluster_centers_ = X[np.random.choice(n_samples, self.n_clusters, replace=False)]

        for iteration in range(self.max_iter):
            # Assign labels based on closest cluster center
            labels, _ = pairwise_distances_argmin_min(X, self.cluster_centers_)

            # Update cluster centers
            new_centers = np.zeros_like(self.cluster_centers_)
            for k in range(self.n_clusters):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    new_centers[k] = np.mean(cluster_points, axis=0)
                else:
                    new_centers[k] = self.cluster_centers_[k]

            # Check for convergence
            if np.linalg.norm(new_centers - self.cluster_centers_) < self.tol:
                break

            self.cluster_centers_ = new_centers

        self.labels_ = labels
        return self

    def predict(self, X):
        """Predict cluster labels for input data.
        
        Args:
            X (array-like): Input data of shape (n_samples, n_features)
            
        Returns:
            array: Predicted cluster labels
        """
        labels, _ = pairwise_distances_argmin_min(X, self.cluster_centers_)
        return labels
