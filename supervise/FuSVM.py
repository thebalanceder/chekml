import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

class FuzzySVM:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', fuzzy_method='distance'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.fuzzy_method = fuzzy_method
        self.model = None

    def compute_fuzzy_membership(self, X, y):
        """ Computes fuzzy membership values for each sample."""
        n_samples = X.shape[0]
        fuzzy_weights = np.ones(n_samples)
        
        if self.fuzzy_method == 'distance':
            # Compute mean per class
            class_means = {}
            for cls in np.unique(y):
                class_means[cls] = np.mean(X[y == cls], axis=0)
            
            # Assign membership based on distance to class mean
            for i in range(n_samples):
                cls = y[i]
                distance = np.linalg.norm(X[i] - class_means[cls])
                fuzzy_weights[i] = 1 / (1 + distance)
        
        return fuzzy_weights

    def fit(self, X, y):
        """ Trains the Fuzzy SVM with multi-threading."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Normalize data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Compute fuzzy memberships in parallel
        num_cores = multiprocessing.cpu_count()
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            fuzzy_weights = np.array(list(executor.map(lambda i: self.compute_fuzzy_membership(X_train, y_train)[i], range(len(X_train)))))
        
        # Fit SVM with weighted samples
        self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, probability=True)
        self.model.fit(X_train, y_train, sample_weight=fuzzy_weights)
        
        print("Training Accuracy:", self.model.score(X_train, y_train))
        print("Test Accuracy:", self.model.score(X_test, y_test))
    
    def predict(self, X):
        """ Predicts using the trained Fuzzy SVM model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        return self.model.predict(X)
    
    def optimize_hyperparameters(self, X, y):
        """ Performs hyperparameter tuning using GridSearchCV."""
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1]
        }
        
        num_cores = multiprocessing.cpu_count()
        grid_search = GridSearchCV(SVC(kernel=self.kernel), param_grid, cv=5, n_jobs=num_cores)
        grid_search.fit(X, y)
        
        print("Best Parameters:", grid_search.best_params_)
        self.C = grid_search.best_params_['C']
        self.gamma = grid_search.best_params_['gamma']
        self.model = grid_search.best_estimator_

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    fuzzy_svm = FuzzySVM(kernel='rbf')
    fuzzy_svm.fit(X, y)
    predictions = fuzzy_svm.predict(X[:5])
    print("Predictions:", predictions)

