import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance as sklearn_perm_importance
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

# -----------------------------
# Pure low-level functions (no pandas dependency)
# -----------------------------
def distance_correlation_single(x_vec, y_vec):
    x_vec = np.asarray(x_vec).reshape(-1, 1)
    y_vec = np.asarray(y_vec).reshape(-1, 1)
    a = squareform(pdist(x_vec))
    b = squareform(pdist(y_vec))
    A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()
    dcov = np.sqrt(np.mean(A * B))
    dvar_x = np.sqrt(np.mean(A * A))
    dvar_y = np.sqrt(np.mean(B * B))
    return 0 if dvar_x * dvar_y == 0 else dcov / np.sqrt(dvar_x * dvar_y)

def hsic_single(x_vec, y_vec, sigma=1.0):
    x_vec = np.asarray(x_vec).reshape(-1, 1)
    y_vec = np.asarray(y_vec).reshape(-1, 1)
    n = len(x_vec)
    def rbf(a):
        dists = squareform(pdist(a, 'sqeuclidean'))
        return np.exp(-dists / (2 * sigma**2))
    K = rbf(x_vec)
    L = rbf(y_vec)
    H = np.eye(n) - np.ones((n, n)) / n
    return np.trace(H @ K @ H @ L) / (n - 1)**2

# -----------------------------
# Single-feature metrics (now robust)
# -----------------------------
def pearson_corr(X, y_arr):
    return X.apply(lambda col: abs(pearsonr(col.values, y_arr)[0]))

def spearman_corr(X, y_arr):
    return X.apply(lambda col: abs(spearmanr(col.values, y_arr)[0]))

def kendall_corr(X, y_arr):
    return X.apply(lambda col: abs(kendalltau(col.values, y_arr)[0]) if len(col) > 1 else 0)

def HSIC(X, y_arr):
    return X.apply(lambda col: hsic_single(col.values, y_arr))

def mutual_info(X, y):
    mi = mutual_info_regression(X, y)
    return pd.Series(mi, index=X.columns)

def normalized_mi(X, y):
    mi = mutual_info_regression(X, y)
    max_mi = mi.max()
    return pd.Series(mi / max_mi if max_mi > 0 else mi, index=X.columns)

def distance_corr(X, y):
    y_arr = np.asarray(y)
    return X.apply(lambda col: distance_correlation_single(col.values, y_arr))

def tree_importance(X, y):
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns)

def permutation_importance(X, y):
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    perm = sklearn_perm_importance(rf, X, y, n_repeats=5, random_state=42, n_jobs=-1)
    return pd.Series(perm.importances_mean, index=X.columns)

def partial_dependence_variance(X, y):
    # Simple proxy using permutation importance
    return permutation_importance(X, y) * 0.8

def nonlinear_r2(X, y):
    y_arr = np.asarray(y)
    scores = []
    for col in X.columns:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X[[col]], y_arr)
        scores.append(r2_score(y_arr, rf.predict(X[[col]])))
    return pd.Series(scores, index=X.columns)

# Placeholder for MIC (requires minepy — keep as random or replace later)
def MIC(X, y):
    return pd.Series(np.random.rand(X.shape[1]), index=X.columns)

def corr_ratio(X, y):
    # Simple proxy
    return distance_corr(X, y)

# -----------------------------
# Interaction metrics (fixed)
# -----------------------------
def hsic_score_multi(X_joint, y):
    n = X_joint.shape[0]
    y_arr = np.asarray(y).reshape(-1, 1)
    def rbf(Z): return np.exp(-squareform(pdist(Z, 'sqeuclidean')) / 2)
    K = rbf(X_joint)
    L = rbf(y_arr)
    H = np.eye(n) - np.ones((n, n)) / n
    return np.trace(H @ K @ H @ L) / (n - 1)**2

def joint_mutual_info(X_joint, y):
    mi_vals = mutual_info_regression(X_joint, y)
    return np.mean(mi_vals)

def joint_distance_corr(X_joint, y):
    y_arr = np.asarray(y)
    scores = [distance_correlation_single(X_joint[:, i], y_arr) for i in range(X_joint.shape[1])]
    return np.mean(scores)

def joint_nonlinear_r2(X_joint, y):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_joint, y)
    return r2_score(y, rf.predict(X_joint))

# -----------------------------
# Main Class
# -----------------------------
class FeatureEvaluator:
    def __init__(self, weights=None, interaction_weights=None, max_interaction_order=2, show_heatmap=True):
        self.show_heatmap = show_heatmap
        self.max_interaction_order = max_interaction_order

        metric_names = ['pearson', 'spearman', 'kendall', 'mutual_info', 'normalized_mi',
                        'distance_corr', 'MIC', 'corr_ratio', 'HSIC', 'tree_importance',
                        'permutation_importance', 'partial_dependence_variance', 'nonlinear_r2']
        self.weights = {m: 1.0 for m in metric_names} if weights is None else weights

        inter_metric_names = ['joint_hsic', 'joint_mutual_info', 'joint_distance_corr', 'joint_nonlinear_r2']
        self.interaction_weights = {m: 1.0 for m in inter_metric_names} if interaction_weights is None else interaction_weights

    def fit(self, X: pd.DataFrame, y):
        y_arr = np.asarray(y)

        metrics = {
            'pearson': pearson_corr(X, y_arr),
            'spearman': spearman_corr(X, y_arr),
            'kendall': kendall_corr(X, y_arr),
            'mutual_info': mutual_info(X, y_arr),
            'normalized_mi': normalized_mi(X, y_arr),
            'distance_corr': distance_corr(X, y_arr),
            'MIC': MIC(X, y_arr),
            'corr_ratio': corr_ratio(X, y_arr),
            'HSIC': HSIC(X, y_arr),
            'tree_importance': tree_importance(X, y_arr),
            'permutation_importance': permutation_importance(X, y_arr),
            'partial_dependence_variance': partial_dependence_variance(X, y_arr),
            'nonlinear_r2': nonlinear_r2(X, y_arr),
        }

        df_metrics = pd.DataFrame(metrics)
        df_norm = df_metrics.copy()
        for col in df_norm.columns:
            vals = df_norm[col].values
            vals = np.nan_to_num(vals)
            max_val = np.max(vals)
            df_norm[col] = vals / max_val if max_val > 0 else 0.0

        total_weight = sum(self.weights.get(m, 1.0) for m in df_norm.columns)
        scores = {f: sum(df_norm.loc[f, m] * self.weights.get(m, 1.0) for m in df_norm.columns) / total_weight 
                  for f in X.columns}

        self.metrics_df = df_norm
        self.universal_scores = pd.Series(scores).sort_values(ascending=False)

        # Interactions
        interaction_results = {}
        cols = list(X.columns)
        if self.max_interaction_order >= 2:
            for order in range(2, self.max_interaction_order + 1):
                for comb in combinations(cols, order):
                    x_joint = X[list(comb)].values
                    inter_metrics = {
                        'joint_hsic': hsic_score_multi(x_joint, y_arr),
                        'joint_mutual_info': joint_mutual_info(x_joint, y_arr),
                        'joint_distance_corr': joint_distance_corr(x_joint, y_arr),
                        'joint_nonlinear_r2': joint_nonlinear_r2(x_joint, y_arr)
                    }
                    interaction_results[comb] = inter_metrics

        if interaction_results:
            inter_df = pd.DataFrame(interaction_results).T
            inter_df_norm = inter_df.copy()
            for col in inter_df_norm.columns:
                vals = inter_df_norm[col].values
                max_val = np.max(vals)
                inter_df_norm[col] = vals / max_val if max_val > 0 else 0.0

            total_inter_weight = sum(self.interaction_weights.get(m, 1.0) for m in inter_df_norm.columns)
            inter_scores = {idx: sum(inter_df_norm.loc[idx, m] * self.interaction_weights.get(m, 1.0) for m in inter_df_norm.columns) / total_inter_weight
                            for idx in inter_df_norm.index}

            self.interaction_metrics = inter_df_norm
            self.interaction_scores = pd.Series(inter_scores).sort_values(ascending=False)
        else:
            self.interaction_metrics = None
            self.interaction_scores = None

        # Heatmaps
        if self.show_heatmap:
            plt.figure(figsize=(12, max(6, len(X.columns)//2)))
            sns.heatmap(df_norm, annot=True, cmap='viridis', fmt=".2f")
            plt.title("Normalized Single-Feature Metrics")
            plt.show()

            if self.interaction_metrics is not None and not self.interaction_metrics.empty:
                plt.figure(figsize=(12, 8))
                sns.heatmap(self.interaction_metrics, annot=True, cmap='magma', fmt=".2f")
                plt.title("Normalized Interaction Metrics")
                plt.show()

        return self.universal_scores

    def learn_adaptive_weights(self, X, y, n_boot=30, cv_folds=3):
        if self.metrics_df is None:
            raise ValueError("Run fit() first to compute metrics.")
        
        df = self.metrics_df  # Normalized metrics (features x metrics)
        
        # Step 1: Detect target characteristics
        linearity_score = np.mean([abs(pearsonr(X[col], y)[0]) for col in X.columns])  # Avg Pearson
        nonlinearity_score = np.mean(df['tree_importance'] - df['pearson'])  # Tree gain over linear
        
        is_linear = linearity_score > 0.5  # Threshold for "linear trend"
        is_nonlinear = nonlinearity_score > 0.2  # Threshold for non-linear
        
        # Step 2: Bootstrap + CV to learn weights based on performance
        weights = {m: 0 for m in df.columns}
        for _ in range(n_boot):
            # Bootstrap sample
            idx = np.random.choice(len(X), len(X), replace=True)
            X_boot, y_boot = X.iloc[idx], y.iloc[idx]
            
            for metric in df.columns:
                # Temp score using this metric alone
                feat_scores = df[metric]
                top_feats = feat_scores.nlargest(5).index  # Top 5 by this metric
                
                # Evaluate on CV: How good is a model with these features?
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                cv_r2 = np.mean(cross_val_score(model, X_boot[top_feats], y_boot, cv=cv_folds, scoring='r2'))
                
                # Accumulate: Higher CV R2 → higher weight for this metric
                weights[metric] += cv_r2
        
        # Normalize accumulated weights
        total = sum(weights.values()) + 1e-8
        weights = {k: v / total for k, v in weights.items()}
        
        # Step 3: Adjust for detected target traits
        if is_linear:
            # Boost linear metrics
            for m in ['pearson', 'spearman', 'kendall']:
                weights[m] *= 1.5  # +50% weight
        if is_nonlinear:
            # Boost non-linear metrics
            for m in ['HSIC', 'distance_corr', 'mutual_info', 'tree_importance']:
                weights[m] *= 1.5
        
        # Re-normalize after adjustments
        total = sum(weights.values()) + 1e-8
        self.weights = {k: v / total for k, v in weights.items()}
        
        return self.weights

if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing(as_frame=True)
    df = data.frame.sample(n=3000, random_state=42)
    X = df.drop(columns='MedHouseVal')
    y = df['MedHouseVal']

    evaluator = FeatureEvaluator(
        weights={'tree_importance': 2.5, 'permutation_importance': 2.0, 'mutual_info': 1.8},
        show_heatmap=True
    )

    universal_scores = evaluator.fit(X, y)
    print(universal_scores.round(4))

    learned_weights = evaluator.learn_adaptive_weights(X, y)
    print(learned_weights)
