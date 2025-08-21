import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

# Import the modules (adjust paths if needed)
from greedy_ensemble import GreedyEnsembleSelection
from bayesian_model_combination import BayesianModelCombination
from negative_correlation_ensemble import NCLEnsemble
from rl_ensemble import RLEnsemble
from hierarchical_stacking import HierarchicalStacking

# Generate dummy regression data
np.random.seed(42)
X_reg = np.random.rand(100, 2) * 10
y_reg = 2 * X_reg[:, 0] + 3 * X_reg[:, 1] + np.random.randn(100)  # Linear with noise
X_train_reg, X_test_reg = X_reg[:80], X_reg[80:]
y_train_reg, y_test_reg = y_reg[:80], y_reg[80:]

# Generate dummy classification data
X_clf = np.random.rand(100, 2) * 10
y_clf = (X_clf[:, 0] + X_clf[:, 1] > 10).astype(int)  # Binary class
X_train_clf, X_test_clf = X_clf[:80], X_clf[80:]
y_train_clf, y_test_clf = y_clf[:80], y_clf[80:]

# Define sample base models
reg_models_dict = {
    "Linear": LinearRegression(),
    "SVR": SVR(),
    "Tree": DecisionTreeRegressor()
}
clf_models_list = [SVC(probability=True), DecisionTreeClassifier()]

if __name__ == "__main__":
    # 1. Greedy Ensemble Selection (Regression example)
    gens = GreedyEnsembleSelection(reg_models_dict, task="regression")
    gens.fit(X_train_reg, y_train_reg, X_test_reg, y_test_reg)  # Uses test as val for simplicity
    preds_gens = gens.predict(X_test_reg)
    print("Greedy Ensemble MSE:", mean_squared_error(y_test_reg, preds_gens))

    # 2. Bayesian Model Combination (Regression example)
    bmc = BayesianModelCombination(reg_models_dict)
    bmc.fit(X_train_reg, y_train_reg, X_test_reg, y_test_reg)
    preds_bmc = bmc.predict(X_test_reg)
    print("Bayesian MC MSE:", mean_squared_error(y_test_reg, preds_bmc))

    # 3. Negative Correlation Ensemble (Classification example)
    ncl = NCLEnsemble(clf_models_list, task="classification")
    ncl.fit(X_train_clf, y_train_clf)
    preds_ncl = ncl.predict(X_test_clf)
    print("NCL Ensemble Accuracy:", accuracy_score(y_test_clf, preds_ncl))

    # 4. RL Ensemble (Regression example)
    rl = RLEnsemble(reg_models_dict, num_episodes=50)  # Reduced episodes for speed
    rl.fit(X_train_reg, y_train_reg, X_test_reg, y_test_reg)
    preds_rl = rl.predict(X_test_reg)
    print("RL Ensemble MSE:", mean_squared_error(y_test_reg, preds_rl))

    # 5. Hierarchical Stacking (Classification example)
    base_models_clf = [("SVC", SVC()), ("Tree", DecisionTreeClassifier())]
    hs = HierarchicalStacking(base_models_clf, task="classification")
    hs.fit(X_clf, y_clf)  # Uses internal split
    preds_hs = hs.predict(X_test_clf)
    print("Hierarchical Stacking Accuracy:", accuracy_score(y_test_clf, preds_hs.round()))
