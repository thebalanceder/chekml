import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from pyod.models.cof import COF
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
import time

class RobustOneClassSVM:
    def __init__(self, nu=0.1, gamma=0.1, batch_size=32, delta=1.0, rff_dim=100):
        self.nu = nu
        self.gamma = gamma
        self.batch_size = batch_size
        self.delta = delta
        self.rff_dim = rff_dim
        self.w = None
        self.rho = None
        self.omega = None
        self.b = None

    def _random_fourier_features(self, X):
        if self.omega is None or self.b is None:
            self.omega = np.random.randn(X.shape[1], self.rff_dim) * np.sqrt(2 * self.gamma)
            self.b = np.random.uniform(0, 2 * np.pi, self.rff_dim)
        Z = np.sqrt(2.0 / self.rff_dim) * np.cos(np.dot(X, self.omega) + self.b)
        return Z

    def _huber_loss(self, xi):
        return np.where(xi <= self.delta, 0.5 * xi ** 2, self.delta * (xi - 0.5 * self.delta))

    def fit(self, X, epochs=100, lr=0.01):
        Z = self._random_fourier_features(X)
        n, d = Z.shape
        self.w = np.zeros(d)
        self.rho = 0.0

        for epoch in range(epochs):
            indices = np.random.permutation(n)
            for i in range(0, n, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                Z_batch = Z[batch_indices]

                margin = Z_batch @ self.w - self.rho
                xi = np.maximum(0, 1 - margin)
                loss = self._huber_loss(xi).mean()

                grad_w = -(Z_batch * (xi <= self.delta)[:, None]).mean(axis=0)
                grad_rho = (xi <= self.delta).mean() * -1

                self.w -= lr * grad_w
                self.rho -= lr * grad_rho
        return self

    def decision_function(self, X):
        Z = self._random_fourier_features(X)
        return Z @ self.w - self.rho

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)

class OutlierEvaluator:
    @staticmethod
    def evaluate_models(X_train, X_test, y_test):
        results = {}
        
        # RobustOneClassSVM
        rsvm = RobustOneClassSVM(nu=0.1, gamma=0.1, batch_size=32, delta=1.0, rff_dim=100)
        start = time.time()
        rsvm.fit(X_train)
        rsvm_time = time.time() - start
        rsvm_preds = rsvm.predict(X_test)
        rsvm_preds = np.where(rsvm_preds == 1, -1, 1)
        results['RobustOneClassSVM'] = accuracy_score(y_test, rsvm_preds)
        results['time_rsvm'] = rsvm_time

        # OneClassSVM
        ocs = OneClassSVM(nu=0.1, kernel='rbf', gamma=0.1)
        start = time.time()
        ocs.fit(X_train)
        results['time_ocs'] = time.time() - start
        ocs_preds = ocs.predict(X_test)
        results['OneClassSVM'] = accuracy_score(y_test, ocs_preds)

        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
        start = time.time()
        iso_forest.fit(X_train)
        results['time_iso'] = time.time() - start
        iso_preds = iso_forest.predict(X_test)
        results['IsolationForest'] = accuracy_score(y_test, iso_preds)

        # Local Outlier Factor
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, n_jobs=-1)
        start = time.time()
        lof_preds = lof.fit_predict(X_test)
        results['time_lof'] = time.time() - start
        results['LocalOutlierFactor'] = accuracy_score(y_test, lof_preds)

        # Elliptic Envelope
        ee = EllipticEnvelope(contamination=0.1)
        start = time.time()
        ee.fit(X_train)
        results['time_ee'] = time.time() - start
        ee_preds = ee.predict(X_test)
        results['EllipticEnvelope'] = accuracy_score(y_test, ee_preds)

        # HBOS
        hbos = HBOS(contamination=0.1)
        start = time.time()
        hbos.fit(X_train)
        results['time_hbos'] = time.time() - start
        hbos_preds = hbos.predict(X_test)
        results['HBOS'] = accuracy_score(y_test, hbos_preds)

        # KNN
        knn = KNN(contamination=0.1, n_jobs=-1)
        start = time.time()
        knn.fit(X_train)
        results['time_knn'] = time.time() - start
        knn_preds = knn.predict(X_test)
        results['KNN'] = accuracy_score(y_test, knn_preds)

        # COF
        cof = COF(contamination=0.1)
        start = time.time()
        cof.fit(X_train)
        results['time_cof'] = time.time() - start
        cof_preds = cof.predict(X_test)
        results['COF'] = accuracy_score(y_test, cof_preds)

        return results
