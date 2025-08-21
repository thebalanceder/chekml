import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

class DGMM:
    def __init__(self, n_components, input_dim, encoding_dim=10, n_init=10, random_state=42):
        """
        Initialize Deep Gaussian Mixture Model.
        :param n_components: Number of mixture components
        :param input_dim: Dimensionality of input data
        :param encoding_dim: Dimensionality of encoded features
        :param n_init: Number of initializations for GMM
        :param random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.n_init = n_init
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.autoencoder, self.encoder = self._build_autoencoder()
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full', n_init=n_init, random_state=random_state)

    def _build_autoencoder(self):
        """
        Build autoencoder for feature extraction.
        :return: Tuple of (autoencoder, encoder)
        """
        input_layer = layers.Input(shape=(self.input_dim,))
        encoded = layers.Dense(self.encoding_dim, activation='relu')(input_layer)
        decoded = layers.Dense(self.input_dim, activation='sigmoid')(encoded)
        autoencoder = keras.Model(input_layer, decoded)
        encoder = keras.Model(input_layer, encoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder, encoder

    def fit(self, X, epochs=50, batch_size=256):
        """
        Fit the DGMM on input data.
        :param X: Numpy array of shape (n_samples, n_features)
        :param epochs: Number of epochs for autoencoder training
        :param batch_size: Batch size for autoencoder training
        :return: Self
        """
        X_scaled = self.scaler.fit_transform(X)
        self.autoencoder.fit(X_scaled, X_scaled, epochs=epochs, batch_size=batch_size, verbose=0)
        X_encoded = self.encoder.predict(X_scaled, verbose=0)
        self.gmm.fit(X_encoded)
        return self

    def predict(self, X):
        """
        Predict cluster labels for input data.
        :param X: Numpy array of shape (n_samples, n_features)
        :return: Cluster labels
        """
        X_scaled = self.scaler.transform(X)
        X_encoded = self.encoder.predict(X_scaled, verbose=0)
        return self.gmm.predict(X_encoded)

    def compute_bic(self, X):
        """
        Compute BIC for the model.
        :param X: Numpy array of shape (n_samples, n_features)
        :return: BIC value
        """
        X_scaled = self.scaler.transform(X)
        X_encoded = self.encoder.predict(X_scaled, verbose=0)
        return self.gmm.bic(X_encoded)
