import numpy as np
from sklearn.metrics import mean_squared_error

class BayesianModelCombination:
    """
    Bayesian weighting of models based on likelihood.
    Special function: Probabilistic weights via Bayesian updating.
    """
    def __init__(self, models):
        self.models = models  # Dict of {name: model}
        self.weights = None

    def fit(self, X_train, y_train, X_val, y_val):
        predictions = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            predictions[name] = preds

        # Compute likelihoods (assuming Gaussian errors)
        likelihoods = {}
        for name, preds in predictions.items():
            mse = mean_squared_error(y_val, preds)
            likelihoods[name] = np.exp(-mse)

        total_likelihood = sum(likelihoods.values())
        self.weights = {name: likelihood / total_likelihood for name, likelihood in likelihoods.items()}

    def predict(self, X_test):
        predictions = {name: self.models[name].predict(X_test) for name in self.models}
        return np.sum([self.weights[name] * predictions[name] for name in self.models], axis=0)
