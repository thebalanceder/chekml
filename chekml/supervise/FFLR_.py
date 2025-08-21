import numpy as np
import os
from numba import njit
from scipy.linalg import lu_factor, lu_solve
import dask.array as da
from joblib import Parallel, delayed

# ✅ Try Importing CuPy (Disable GPU if Not Available)
GPU_AVAILABLE = True
try:
    import cupy as cp
    from cupyx.scipy.linalg import lu_factor as cu_lu_factor, lu_solve as cu_lu_solve
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ CuPy not available. GPU functions disabled.")

# ✅ Force NumPy to use all CPU cores
os.environ["OPENBLAS_NUM_THREADS"] = "8"  

# ✅ Preprocess LU with Regularization (Ridge)
def preprocess_lu(X, y, alpha):
    """Precomputes LU decomposition with Ridge regularization."""
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))
    XT_X = X.T @ X + alpha * np.eye(X.shape[1])  # Ridge regularization
    XT_y = X.T @ y
    lu, piv = lu_factor(XT_X)
    return lu, piv, XT_y

# ✅ Parallel LU Factorization (Fixed)
def parallel_lu_factor(X, y, alpha, n_jobs):
    """Parallel LU decomposition using joblib."""
    X_chunks = np.array_split(X, n_jobs)  
    y_chunks = np.array_split(y, n_jobs)  

    results = Parallel(n_jobs=n_jobs)(
        delayed(preprocess_lu)(X_chunk, y_chunk, alpha)
        for X_chunk, y_chunk in zip(X_chunks, y_chunks)
    )
    
    return results[0]  

# ✅ Mini-Batch Gradient Descent for Streaming Data
@njit
def mini_batch_gradient_descent(X, y, lr, epochs, batch_size):
    """Performs mini-batch gradient descent for Linear Regression."""
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    
    for epoch in range(epochs):
        for i in range(0, n_samples, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            gradients = -2 * X_batch.T @ (y_batch - X_batch @ theta) / batch_size
            theta -= lr * gradients  
    return theta

@njit
def fast_predict(X, theta):
    """JIT-compiled prediction function."""
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))
    return X @ theta  

# ✅ Custom Linear Regression with Hyperparameters
class FFLR_:
    def __init__(self, 
                 use_gpu=False, 
                 use_parallel=False, 
                 use_mini_batch=False,
                 alpha=0.1,  # Ridge regularization
                 lr=0.01,    # Learning rate for mini-batch SGD
                 epochs=10,  # Number of training iterations
                 batch_size=10000,  # Mini-batch size
                 n_jobs=4):  # Number of parallel jobs
        self.theta = None  
        self.use_gpu = use_gpu and GPU_AVAILABLE  
        self.use_parallel = use_parallel
        self.use_mini_batch = use_mini_batch
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit model using the optimized LU, parallel LU, or mini-batch SGD."""
        if self.use_mini_batch:
            self.theta = mini_batch_gradient_descent(X, y, self.lr, self.epochs, self.batch_size)
        elif self.use_gpu:
            print("🚀 Using GPU for LU Decomposition")
            try:
                lu, piv, XT_y = preprocess_lu_gpu(X, y)  
                self.theta = cu_lu_solve((lu, piv), XT_y)  
            except RuntimeError:
                print("⚠️ GPU function disabled. Falling back to CPU.")
                self.use_gpu = False
                self.fit(X, y)  
        elif self.use_parallel:
            print("🚀 Using Parallel CPU Training")
            lu, piv, XT_y = parallel_lu_factor(X, y, self.alpha, self.n_jobs)  
            self.theta = lu_solve((lu, piv), XT_y)  
        else:
            print("🚀 Using Standard LU Decomposition")
            lu, piv, XT_y = preprocess_lu(X, y, self.alpha)  
            self.theta = lu_solve((lu, piv), XT_y)  

    def predict(self, X):
        """Predict using optimized JIT function. Convert Dask array to NumPy first."""
        if isinstance(X, da.Array):  
            X = X.compute()  
        return fast_predict(X, self.theta)

