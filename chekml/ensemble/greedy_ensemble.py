import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.optimize import minimize

class GreedyEnsembleSelection:
    """
    Greedy ensemble that selects and weights models iteratively to minimize error.
    Special function: Greedy addition with weight optimization.
    """
    def __init__(self, models, task="regression"):
        self.models = models  # Dict of {name: model}
        self.task = task
        self.selected_models = []
        self.weights = []

    def _objective(self, weights, predictions, y_val):
        ensemble_pred = np.sum(weights[:, None] * predictions, axis=0)
        if self.task == "regression":
            return mean_squared_error(y_val, ensemble_pred)
        else:
            if ensemble_pred.ndim == 1:
                return -accuracy_score(y_val, (ensemble_pred > 0.5).astype(int))
            else:
                return -accuracy_score(y_val, np.argmax(ensemble_pred, axis=1))

    def fit(self, X_train, y_train, X_val, y_val):
        model_preds = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            model_preds[name] = np.array(y_pred)

        self.selected_models = []
        self.weights = []
        remaining_models = list(self.models.keys())

        while remaining_models:
            best_model = None
            best_score = float("inf")
            best_weight = None

            for model_name in remaining_models:
                temp_models = self.selected_models + [model_name]
                predictions = np.array([model_preds[m] for m in temp_models])
                init_weights = np.ones(len(temp_models)) / len(temp_models)
                
                res = minimize(self._objective, init_weights, args=(predictions, y_val),
                               bounds=[(0, 1)] * len(temp_models),
                               constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
                
                if res.fun < best_score:
                    best_score = res.fun
                    best_model = model_name
                    best_weight = res.x

            if best_model:
                self.selected_models.append(best_model)
                self.weights = best_weight
                remaining_models.remove(best_model)

    def predict(self, X_test):
        preds = np.array([self.models[m].predict(X_test) for m in self.selected_models])
        ensemble_pred = np.sum(self.weights[:, None] * preds, axis=0)
        if self.task == "regression":
            return ensemble_pred
        else:
            if ensemble_pred.ndim == 1:
                return (ensemble_pred > 0.5).astype(int)
            else:
                return np.argmax(ensemble_pred, axis=1)
