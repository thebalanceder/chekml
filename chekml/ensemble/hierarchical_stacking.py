import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

class HierarchicalStacking:
    """
    Stacking ensemble with base models feeding a meta-learner.
    Special function: Hierarchical meta-learning from base predictions.
    """
    def __init__(self, base_models, task="regression"):
        self.base_models = base_models  # List of (name, model) tuples
        self.task = task
        self.meta_learner = LinearRegression() if task == "regression" else LogisticRegression()

    def fit(self, X, y):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        
        meta_features = np.zeros((X_valid.shape[0], len(self.base_models)))
        for i, (name, model) in enumerate(self.base_models):
            model.fit(X_train, y_train)
            meta_features[:, i] = model.predict(X_valid)
        
        self.meta_learner.fit(meta_features, y_valid)

    def predict(self, X_test):
        meta_features = np.zeros((X_test.shape[0], len(self.base_models)))
        for i, (name, model) in enumerate(self.base_models):
            meta_features[:, i] = model.predict(X_test)
        return self.meta_learner.predict(meta_features)
