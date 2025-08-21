import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

tfd = tfp.distributions
tfb = tfp.bijectors

class NFMM:
    def __init__(self, n_components, event_dims, num_flows=5, n_init=10, random_state=42):
        """
        Initialize Normalizing Flow Mixture Model.
        :param n_components: Number of mixture components
        :param event_dims: Dimensionality of the data
        :param num_flows: Number of flow transformations
        :param n_init: Number of initializations for GMM
        :param random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.event_dims = event_dims
        self.num_flows = num_flows
        self.n_init = n_init
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full', n_init=n_init, random_state=random_state)
        self.flow_model = self._build_realnvp_flow()

    def _build_realnvp_flow(self):
        """
        Build RealNVP flow model.
        :return: Transformed distribution
        """
        bijectors = []
        for i in range(self.num_flows):
            bijectors.append(tfb.RealNVP(num_masked=self.event_dims // 2,
                                        shift_and_log_scale_fn=tfb.real_nvp_default_template(hidden_layers=[64, 64])))
            bijectors.append(tfb.Permute(permutation=list(reversed(range(self.event_dims)))))
        flow_bijector = tfb.Chain(bijectors[::-1])
        base_distribution = tfd.MultivariateNormalDiag(loc=tf.zeros(self.event_dims))
        return tfd.TransformedDistribution(distribution=base_distribution, bijector=flow_bijector)

    def fit(self, X):
        """
        Fit the NFMM on input data.
        :param X: Numpy array of shape (n_samples, n_features)
        :return: Self
        """
        X_scaled = self.scaler.fit_transform(X)
        X_transformed = self.flow_model.sample(X_scaled.shape[0]).numpy()
        self.gmm.fit(X_transformed)
        return self

    def predict(self, X):
        """
        Predict cluster labels for input data.
        :param X: Numpy array of shape (n_samples, n_features)
        :return: Cluster labels
        """
        X_scaled = self.scaler.transform(X)
        X_transformed = self.flow_model.sample(X_scaled.shape[0]).numpy()
        return self.gmm.predict(X_transformed)

    def compute_bic(self, X):
        """
        Compute BIC for the model.
        :param X: Numpy array of shape (n_samples, n_features)
        :return: BIC value
        """
        X_scaled = self.scaler.transform(X)
        X_transformed = self.flow_model.sample(X_scaled.shape[0]).numpy()
        return self.gmm.bic(X_transformed)
