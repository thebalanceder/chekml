import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

class NCLEnsemble:
    """
    Ensemble with negative correlation penalty for diversity.
    Special function: Penalizes high correlation to promote diverse predictions.
    """
    def __init__(self, models, task="regression"):
        self.models = models  # List of models
        self.task = task
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train):
        X_train = self.scaler.fit_transform(X_train)
        for model in self.models:
            model.fit(X_train, y_train)

    def predict(self, X_test):
        X_test = self.scaler.transform(X_test)
        predictions = np.array([model.predict(X_test) for model in self.models])
        ensemble_output = np.mean(predictions, axis=0)
        
        # Compute NCL penalty
        penalties = np.mean((predictions - ensemble_output) ** 2, axis=1)
        penalty = np.mean(penalties)

        if self.task == "classification":
            return np.argmax(ensemble_output, axis=1) if ensemble_output.ndim > 1 else (ensemble_output > 0.5).astype(int)
        return ensemble_output
