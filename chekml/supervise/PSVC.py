import numpy as np
import scipy.sparse as sp
from concurrent.futures import ThreadPoolExecutor

# Check if CuPy is available for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class ProximalSVM:
    def __init__(self, C=1.0, use_gpu=False, kernel=None):
        self.C = C
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.kernel = kernel  # Custom kernel function
        self.w = None
        self.b = None

    def _to_device(self, array):
        """Move array to GPU if CuPy is available and use_gpu is True."""
        if self.use_gpu:
            return cp.array(array)
        return np.array(array)

    def _from_device(self, array):
        """Move array back to CPU if using GPU."""
        if self.use_gpu:
            return cp.asnumpy(array)
        return array

    def _compute_kernel(self, X, Y=None):
        """Compute the kernel matrix if a custom kernel is provided."""
        if self.kernel:
            return self.kernel(X, Y) if Y is not None else self.kernel(X, X)
        return X @ Y.T if Y is not None else X @ X.T  # Use dot product for linear case

    def fit(self, X, y):
        """Train the Proximal SVM model."""
        X = self._to_device(X)
        y = self._to_device(y)
        
        if sp.issparse(X):
            X = X.toarray()  # Convert sparse to dense for computation

        m, n = X.shape
        y = y.reshape(-1, 1)

        if self.kernel:
            I = self._to_device(np.eye(m))
            K = self._to_device(self._compute_kernel(X))
            K_I = K + I / self.C
            self.alpha = self._from_device(np.linalg.solve(K_I, y))
            self.X_train = X  # Store support vectors
        else:  # Solve for w directly in the linear case
            I = self._to_device(np.eye(n))
            XtX = X.T @ X
            XtX_I = XtX + I / self.C
            Xty = X.T @ y
            self.w = self._from_device(np.linalg.solve(XtX_I, Xty))  # Compute weight vector

        self.b = self._from_device(np.mean(y - X @ self.w)) if not self.kernel else self._from_device(np.mean(y - K @ self.alpha))

    def predict(self, X):
        """Predict using the trained model."""
        X = self._to_device(X)
        if sp.issparse(X):
            X = X.toarray()

        if self.kernel:
            K_test = self._to_device(self._compute_kernel(X, self.X_train))
            predictions = K_test @ self._to_device(self.alpha) + self._to_device(self.b)
        else:  # Use directly computed w and b for linear case
            predictions = X @ self._to_device(self.w).reshape(-1, 1) + self._to_device(self.b)

        return self._from_device(np.sign(predictions).flatten())

# Example: High-Accuracy Evaluation on a Real Dataset
if __name__ == "__main__":
    import time
    from sklearn.svm import LinearSVC, SVC
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
    
    # Generate a synthetic dataset (high-accuracy scenario)
    X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
    y = 2 * y - 1  # Convert labels to {-1, 1}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Proximal SVM with RBF kernel
    model = ProximalSVM(C=1.0, use_gpu=True, kernel=None)  
    start=time.time()
    model.fit(X_train, y_train)
    end=time.time()-start
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Proximal SVM Accuracy with RBF Kernel: {accuracy:.4f}")
    print(f"used time:{end:0.4f}")
    
    # Train and evaluate sklearn LinearSVC
    linear_svc = LinearSVC(C=1.0, dual=False, max_iter=10000)
    start=time.time()
    linear_svc.fit(X_train, y_train)
    end=time.time()-start
    y_pred_sklearn_linear = linear_svc.predict(X_test)
    accuracy_linear = accuracy_score(y_test, y_pred_sklearn_linear)
    print(f"LinearSVC Accuracy: {accuracy_linear:.4f}")
    print(f"used time:{end:0.4f}")
    
    # Train and evaluate sklearn SVC with RBF kernel
    svc = SVC(C=1.0, kernel='rbf', gamma='scale')
    start=time.time()
    svc.fit(X_train, y_train)
    end=time.time()-start
    y_pred_sklearn_svc = svc.predict(X_test)
    accuracy_svc = accuracy_score(y_test, y_pred_sklearn_svc)
    print(f"SVC (RBF) Accuracy: {accuracy_svc:.4f}")
    print(f"used time:{end:0.4f}")

