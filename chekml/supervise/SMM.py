import numpy as np
import concurrent.futures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import Parallel, delayed

class SupportMatrixMachine:
    def __init__(self, learning_rate=0.01, epochs=1000, reg_param=0.1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_param = reg_param
        self.weights = None
        self.bias = 0

    def _hinge_loss(self, X, y, w, b):
        margins = 1 - y * (np.dot(X, w) + b)
        loss = np.maximum(0, margins)
        return np.mean(loss) + self.reg_param * np.linalg.norm(w)
    
    def _compute_gradient(self, X, y, w, b):
        margins = 1 - y * (np.dot(X, w) + b)
        misclassified = margins > 0
        dw = -np.dot(X[misclassified].T, y[misclassified]) / len(y) + self.reg_param * w
        db = -np.sum(y[misclassified]) / len(y)
        return dw, db
    
    def _train_batch(self, X_batch, y_batch):
        w = np.zeros(X_batch.shape[1])
        b = 0
        for _ in range(self.epochs):
            dw, db = self._compute_gradient(X_batch, y_batch, w, b)
            w -= self.learning_rate * dw
            b -= self.learning_rate * db
        return w, b

    def fit(self, X, y):
        num_cores = min(len(y), 4)  # Limit to 4 cores to avoid overload
        X_batches = np.array_split(X, num_cores)
        y_batches = np.array_split(y, num_cores)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
            results = list(executor.map(self._train_batch, X_batches, y_batches))
        
        self.weights = np.mean([r[0] for r in results], axis=0)
        self.bias = np.mean([r[1] for r in results])

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)

# Simulated dataset
np.random.seed(42)
X = np.random.randn(1000, 20)
y = np.random.choice([-1, 1], size=1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
smm = SupportMatrixMachine()
smm.fit(X_train, y_train)

# Evaluate
preds = smm.predict(X_test)
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average='binary', pos_label=1)
recall = recall_score(y_test, preds, average='binary', pos_label=1)
f1 = f1_score(y_test, preds, average='binary', pos_label=1)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

