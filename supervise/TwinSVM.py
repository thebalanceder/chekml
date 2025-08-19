import numpy as np
try:
    import cupy as cp
    xp = cp  # Use GPU if available
    gpu_available = True
except ImportError:
    xp = np  # Fallback to CPU
    gpu_available = False

from joblib import Parallel, delayed
from scipy.linalg import inv
from collections import defaultdict
from itertools import combinations
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from collections import Counter
import multiprocessing

class TwinSVM:
    def __init__(self, C1=1, C2=1, kernel="linear", gamma=1, degree=3, custom_kernel=None):
        self.C1 = C1
        self.C2 = C2
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.custom_kernel = custom_kernel
        self.xp = xp  # Use GPU if available

    def kernel_function(self, X1, X2):
        """Apply the selected kernel function."""
        if self.custom_kernel:
            return self.custom_kernel(X1, X2)
        elif self.kernel == "linear":
            return X1 @ X2.T
        elif self.kernel == "rbf":
            sq_dists = self.xp.linalg.norm(X1[:, None] - X2, axis=2) ** 2
            return self.xp.exp(-self.gamma * sq_dists)
        elif self.kernel == "poly":
            return (X1 @ X2.T + 1) ** self.degree
        else:
            raise ValueError("Unsupported kernel type")

    def fit(self, X, y):
        """Train Twin SVM using kernelized approach."""
        X1 = X[y == 1]
        X2 = X[y == -1]

        n1, d = X1.shape
        n2, _ = X2.shape

        # Move data to GPU if available
        X1 = self.xp.array(X1)
        X2 = self.xp.array(X2)

        # Compute kernel matrices
        K1 = self.kernel_function(X1, X1) + self.C1 * self.xp.eye(n1)
        K2 = self.kernel_function(X2, X2) + self.C2 * self.xp.eye(n2)

        # Solve Twin SVM optimization
        G1 = self.xp.linalg.inv(K1) @ self.xp.ones(n1)
        G2 = self.xp.linalg.inv(K2) @ self.xp.ones(n2)

        self.X1, self.X2 = X1, X2  # Store support vectors
        self.G1, self.G2 = G1, G2  # Store solutions

    def predict(self, X):
        """Predict class labels using kernel distances."""
        X = self.xp.array(X)  # Move input to GPU if available

        # Compute kernel distances
        dist1 = self.xp.abs(self.kernel_function(X, self.X1) @ self.G1)
        dist2 = self.xp.abs(self.kernel_function(X, self.X2) @ self.G2)

        return self.xp.where(dist1 < dist2, 1, -1)

class MultiClassTwinSVM:
    def __init__(self, C1=1, C2=1, kernel="linear", gamma=1, degree=3, custom_kernel=None):
        self.C1 = C1
        self.C2 = C2
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.custom_kernel = custom_kernel
        self.classifiers = {}
        self.classes = None

    def fit(self, X, y):
        """Train a One-Versus-One Twin SVM for multi-class classification."""
        self.classes = xp.unique(y)  # Get unique class labels
        for class1, class2 in combinations(self.classes, 2):
            # Filter dataset for the two classes
            mask = (y == class1) | (y == class2)
            X_binary, y_binary = X[mask], y[mask]
            
            # Convert labels to -1, 1 for binary classification
            y_binary = xp.where(y_binary == class1, 1, -1)

            # Train Twin SVM
            clf = TwinSVM(self.C1, self.C2, self.kernel, self.gamma, self.degree, self.custom_kernel)
            clf.fit(X_binary, y_binary)
            self.classifiers[(class1, class2)] = clf

    def predict(self, X):
        """Predict multi-class labels using majority voting."""
        votes = xp.zeros((X.shape[0], len(self.classes)), dtype=int)

        for (class1, class2), clf in self.classifiers.items():
            preds = clf.predict(X)  # Binary predictions
            for i, pred in enumerate(preds):
                if pred == 1:
                    votes[i, xp.where(self.classes == class1)] += 1
                else:
                    votes[i, xp.where(self.classes == class2)] += 1

        # Assign the class with the highest votes
        return self.classes[xp.argmax(votes, axis=1)]

def train_tsvm(C1, C2, kernel, gamma, custom_kernel, X_train, y_train, X_test, y_test):
    """Train and evaluate Twin SVM with given C1, C2, and kernel."""
    model = TwinSVM(C1, C2, kernel, gamma, custom_kernel=custom_kernel)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = xp.mean(y_pred == y_test)
    return (C1, C2, kernel, gamma, accuracy, model)

if __name__ == "__main__":
    # Generate an imbalanced dataset
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, 
                               n_redundant=0, n_clusters_per_class=1, weights=[0.9, 0.1], 
                               random_state=42)

    y = np.where(y == 0, -1, 1)  # Convert labels from (0,1) to (-1,1)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("GPU Available:", gpu_available)
    print("Training class distribution:", Counter(y_train))
    print("Testing class distribution:", Counter(y_test))

    # Move data to GPU if available
    X_train = xp.array(X_train)
    X_test = xp.array(X_test)
    y_train = xp.array(y_train)
    y_test = xp.array(y_test)

    # Define a custom kernel function
    def custom_sigmoid_kernel(X1, X2):
        return xp.tanh(0.5 * (X1 @ X2.T) + 1)

    # Parallel Hyperparameter Tuning
    C1_values = [0.1, 1, 10]
    C2_values = [0.1, 1, 10]
    kernel_types = ["linear", "rbf", "poly", "custom"]
    gamma_values = [0.1, 1, 10]
    num_cores = multiprocessing.cpu_count()

    results = Parallel(n_jobs=num_cores)(
        delayed(train_tsvm)(
            C1, C2, kernel, gamma, custom_sigmoid_kernel if kernel == "custom" else None,
            X_train, y_train, X_test, y_test
        )
        for C1 in C1_values for C2 in C2_values for kernel in kernel_types for gamma in gamma_values
    )

    # Select Best Model
    best_C1, best_C2, best_kernel, best_gamma, best_acc, best_model = max(results, key=lambda x: x[4])

    print(f"Best Model: C1={best_C1}, C2={best_C2}, Kernel={best_kernel}, Gamma={best_gamma}, Accuracy={best_acc:.4f}")

    # Final Evaluation
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test.get() if gpu_available else y_test, y_pred.get() if gpu_available else y_pred))
    print("AUC-ROC:", roc_auc_score(y_test.get() if gpu_available else y_test, y_pred.get() if gpu_available else y_pred))

    # Generate a 3-class dataset
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, 
                               n_redundant=0, n_classes=3, n_clusters_per_class=1, 
                               random_state=42)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Move data to GPU if available
    X_train = xp.array(X_train)
    X_test = xp.array(X_test)
    y_train = xp.array(y_train)
    y_test = xp.array(y_test)

    # Train Multi-Class Twin SVM
    model = MultiClassTwinSVM(C1=1, C2=1, kernel="linear")
    model.fit(X_train, y_train)

    # Predict and Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test.get() if gpu_available else y_test, 
                                y_pred.get() if gpu_available else y_pred))
