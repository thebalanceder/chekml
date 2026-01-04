import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression, f_classif, chi2
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr, spearmanr, kendalltau, rankdata
from sklearn.feature_selection import SelectKBest
from itertools import combinations
import warnings
import cloudpickle
import os
warnings.filterwarnings("ignore")


class InformationRepurposedFeaturizerWrapper:
    """Wrapper exposing a fit/transform API around the functional IRF implementation.

    Notes:
    - `fit(df)` will run the existing InformationRepurposedFeaturizer(...) function
      with `save_models=True` to persist trained models to `saved_models/` and
      will record which prediction-columns were selected.
    - `transform(df)` will load the saved models (or use returned trained models)
      and produce predictions for the same feature combinations on `df`.

    Caveats: the original implementation fits a per-combo `SimpleImputer` and
    `StandardScaler` which are *not* persisted by the functional implementation.
    For transform-time preprocessing we fit a fresh imputer/scaler on the
    provided `df[combo_columns]` as a pragmatic fallback.
    """

    def __init__(self, models=None, loss_functions=None, metrics=None,
                 prediction_mode="top_n", top_n=5, score_key="mutual_information",
                 specific_metrics=None, level=1, save_results_file=None):
        self.models = models
        self.loss_functions = loss_functions
        self.metrics = metrics
        self.prediction_mode = prediction_mode
        self.top_n = top_n
        self.score_key = score_key
        self.specific_metrics = specific_metrics
        self.level = level
        self.save_results_file = save_results_file
        self.selected_pred_names = None
        self.trained_models = None

    def fit(self, df):
        # Run the functional IRF to obtain trained models and selected prediction columns
        result_df, metric_scores_df, feature_mi, trained_models = InformationRepurposedFeaturizer(
            df=df,
            models=self.models,
            loss_functions=self.loss_functions,
            metrics=self.metrics,
            prediction_mode=self.prediction_mode,
            top_n=self.top_n,
            score_key=self.score_key,
            specific_metrics=self.specific_metrics,
            level=self.level,
            save_models=True,
            save_results_file=self.save_results_file
        )

        base_cols = [c for c in df.columns if c != 'target']
        # Selected predictions are the extra columns added by the function
        self.selected_pred_names = [c for c in result_df.columns if c not in base_cols + ['target']]
        self.trained_models = trained_models or {}
        return {'result_df': result_df, 'metric_scores_df': metric_scores_df, 'feature_mi': feature_mi}

    def transform(self, df):
        if self.selected_pred_names is None:
            raise RuntimeError('Featurizer has not been fitted. Call fit() first.')

        new_df = df.copy()
        os.makedirs('saved_models', exist_ok=True)

        for pred_name in self.selected_pred_names:
            parts = pred_name.split('_')
            # reverse of f"{combo_name}_{model_name}_{metric_name}_{loss_function_name}"
            if len(parts) < 4:
                # unexpected format; skip
                new_df[pred_name] = np.nan
                continue
            combo_parts = parts[:-3]
            combo_cols = combo_parts

            # Ensure combo columns exist
            missing = [c for c in combo_cols if c not in df.columns]
            if missing:
                new_df[pred_name] = np.nan
                continue

            X_combo = df[combo_cols]
            imputer = SimpleImputer(strategy='mean')
            scaler = StandardScaler()
            try:
                X_imputed = imputer.fit_transform(X_combo)
                X_scaled = scaler.fit_transform(X_imputed)
            except Exception:
                new_df[pred_name] = np.nan
                continue

            # Load model: prefer in-memory returned model, else load from disk
            model = None
            if self.trained_models and pred_name in self.trained_models:
                model = self.trained_models[pred_name]
            else:
                model_path = f"saved_models/{pred_name}.pkl"
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        model = cloudpickle.load(f)

            if model is None:
                new_df[pred_name] = np.nan
                continue

            try:
                preds = model.predict(X_scaled)
                new_df[pred_name] = preds
            except Exception:
                new_df[pred_name] = np.nan

        return new_df

def custom_digitize(data, bins):
    """Custom binning function to handle edge cases."""
    if len(np.unique(data)) < 2:
        return np.zeros_like(data, dtype=int)
    
    try:
        bin_edges = np.linspace(np.min(data), np.max(data) + 1e-10, bins + 1)
        return np.digitize(data, bin_edges, right=False)
    except:
        return np.zeros_like(data, dtype=int)

def compute_mic(x, y, max_bins=10):
    """Compute Maximal Information Coefficient (MIC)."""
    n = len(x)
    max_bins = min(max_bins, int(np.ceil(n ** 0.6)))
    mic = 0.0
    
    for i in range(2, max_bins + 1):
        for j in range(2, max_bins + 1):
            if i * j > max_bins ** 2:
                continue
            try:
                x_bins = pd.qcut(x, q=i, labels=False, duplicates='drop')
                y_bins = pd.qcut(y, q=j, labels=False, duplicates='drop')
                joint_hist, _, _ = np.histogram2d(x_bins, y_bins, bins=(i, j), density=True)
                x_hist = np.sum(joint_hist, axis=1)
                y_hist = np.sum(joint_hist, axis=0)
                mi = 0.0
                for k in range(i):
                    for l in range(j):
                        if joint_hist[k, l] > 0:
                            mi += joint_hist[k, l] * np.log2(
                                joint_hist[k, l] / (x_hist[k] * y_hist[l] + 1e-10)
                            )
                norm_mi = mi / np.log2(min(i, j))
                mic = max(mic, norm_mi)
            except:
                continue
    
    return mic

def compute_copula_measure(x, y):
    """Compute a simplified copula-based dependence measure."""
    x_rank = rankdata(x, method='average') / (len(x) + 1)
    y_rank = rankdata(y, method='average') / (len(y) + 1)
    copula_corr = spearmanr(x_rank, y_rank)[0]
    return abs(copula_corr)

def compute_distance_correlation(x, y):
    """Compute distance correlation."""
    n = len(x)
    if n < 2:
        return 0.0
    
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    A = np.abs(x - x.T)
    B = np.abs(y - y.T)
    
    A_mean = np.mean(A)
    A_row_mean = np.mean(A, axis=1)
    A_col_mean = np.mean(A, axis=0)
    A_centered = A - A_row_mean[:, None] - A_col_mean[None, :] + A_mean
    
    B_mean = np.mean(B)
    B_row_mean = np.mean(B, axis=1)
    B_col_mean = np.mean(B, axis=0)
    B_centered = B - B_row_mean[:, None] - B_col_mean[None, :] + B_mean
    
    dCov = np.sqrt(np.mean(A_centered * B_centered))
    dVar_x = np.sqrt(np.mean(A_centered * A_centered))
    dVar_y = np.sqrt(np.mean(B_centered * B_centered))
    
    if dVar_x > 0 and dVar_y > 0:
        dCor = dCov / np.sqrt(dVar_x * dVar_y)
    else:
        dCor = 0.0
    
    return dCor

def compute_hsic(x, y):
    """Compute Hilbert-Schmidt Independence Criterion (HSIC) with Gaussian kernels."""
    n = len(x)
    if n < 2:
        return 0.0
    
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    dist_x = np.abs(x - x.T)
    dist_y = np.abs(y - y.T)
    
    sigma_x = np.median(dist_x[dist_x > 0]) if np.any(dist_x > 0) else 1.0
    sigma_y = np.median(dist_y[dist_y > 0]) if np.any(dist_y > 0) else 1.0
    
    K = np.exp(-dist_x ** 2 / (2 * sigma_x ** 2 + 1e-10))
    L = np.exp(-dist_y ** 2 / (2 * sigma_y ** 2 + 1e-10))
    
    H = np.eye(n) - np.ones((n, n)) / n
    HK = np.dot(H, K)
    HL = np.dot(H, L)
    hsic = np.sum(HK * HL.T) / (n ** 2)
    
    return np.sqrt(max(hsic, 0))

def compute_energy_distance_correlation(x, y):
    """Compute energy distance correlation."""
    n = len(x)
    if n < 2:
        return 0.0
    
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    
    dist_xy = np.abs(x - y.T).mean()
    dist_xx = np.abs(x - x.T).mean()
    dist_yy = np.abs(y - y.T).mean()
    
    energy_dist = 2 * dist_xy - dist_xx - dist_yy
    
    var_xx = np.sqrt(np.mean((np.abs(x - x.T) - dist_xx) ** 2))
    var_yy = np.sqrt(np.mean((np.abs(y - y.T) - dist_yy) ** 2))
    
    if var_xx > 0 and var_yy > 0:
        energy_corr = energy_dist / np.sqrt(var_xx * var_yy)
    else:
        energy_corr = 0.0
    
    return abs(energy_corr)

def default_metrics():
    """Define default evaluation metrics."""
    return {
        "pearson": (lambda x, y: pearsonr(x, y)[0], "maximize"),
        "spearman": (lambda x, y: spearmanr(x, y)[0], "maximize"),
        "kendall": (lambda x, y: kendalltau(x, y)[0], "maximize"),
        "mutual_information": (lambda x, y: mutual_info_regression(x.reshape(-1, 1), y)[0], "maximize"),
        "normalized_mutual_information": (
            lambda x, y: mutual_info_score(
                custom_digitize(x, min(10, len(np.unique(x))) if len(np.unique(x)) >= 2 else 2),
                custom_digitize(y, min(10, len(np.unique(y))) if len(np.unique(y)) >= 2 else 2)
            ) / max(
                mutual_info_score(
                    custom_digitize(x, min(10, len(np.unique(x))) if len(np.unique(x)) >= 2 else 2),
                    custom_digitize(x, min(10, len(np.unique(x))) if len(np.unique(x)) >= 2 else 2)
                ),
                1e-10
            ) if len(np.unique(x)) >= 2 and len(np.unique(y)) >= 2 else 0.0,
            "maximize"
        ),
        "conditional_mutual_information": (
            lambda x, y: mutual_info_regression(x.reshape(-1, 1), y)[0],
            "maximize"
        ),
        "maximal_information_coefficient": (compute_mic, "maximize"),
        "symmetric_uncertainty": (
            lambda x, y: 2 * mutual_info_regression(x.reshape(-1, 1), y)[0] / (np.var(x) + np.var(y) + 1e-10),
            "maximize"
        ),
        "total_correlation": (
            lambda x, y: mutual_info_regression(x.reshape(-1, 1), y)[0],
            "maximize"
        ),
        "distance_correlation": (compute_distance_correlation, "maximize"),
        "hsic": (compute_hsic, "maximize"),
        "copula": (compute_copula_measure, "maximize"),
        "gini_correlation": (
            lambda x, y: np.corrcoef(np.argsort(x), np.argsort(y))[0, 1],
            "maximize"
        ),
        "fisher_score": (
            lambda x, y: f_classif(
                x.reshape(-1, 1),
                custom_digitize(y, min(10, len(np.unique(y))) if len(np.unique(y)) >= 2 else 2)
            )[0][0] if len(np.unique(y)) >= 2 else 0.0,
            "maximize"
        ),
        "relief_score": (
            lambda x, y: SelectKBest(score_func=mutual_info_regression, k=1).fit(x.reshape(-1, 1), y).scores_[0],
            "maximize"
        ),
        "anova_f_statistic": (
            lambda x, y: f_classif(
                x.reshape(-1, 1),
                custom_digitize(y, min(10, len(np.unique(y))) if len(np.unique(y)) >= 2 else 2)
            )[0][0] if len(np.unique(y)) >= 2 else 0.0,
            "maximize"
        ),
        "chi_square_score": (
            lambda x, y: chi2(
                (x - x.min() + 1e-10).reshape(-1, 1),
                custom_digitize(y, min(10, len(np.unique(y))) if len(np.unique(y)) >= 2 else 2)
            )[0][0] if len(np.unique(y)) >= 2 else 0.0,
            "maximize"
        ),
        "partial_correlation": (lambda x, y: pearsonr(x, y)[0], "maximize"),
        "information_gain": (
            lambda x, y: mutual_info_score(
                custom_digitize(x, min(10, len(np.unique(x))) if len(np.unique(x)) >= 2 else 2),
                custom_digitize(y, min(10, len(np.unique(y))) if len(np.unique(y)) >= 2 else 2)
            ) if len(np.unique(x)) >= 2 and len(np.unique(y)) >= 2 else 0.0,
            "maximize"
        ),
        "gain_ratio": (
            lambda x, y: mutual_info_score(
                custom_digitize(x, min(10, len(np.unique(x))) if len(np.unique(x)) >= 2 else 2),
                custom_digitize(y, min(10, len(np.unique(y))) if len(np.unique(y)) >= 2 else 2)
            ) / (np.var(y) + 1e-10) if len(np.unique(x)) >= 2 and len(np.unique(y)) >= 2 else 0.0,
            "maximize"
        ),
        "energy_distance_correlation": (compute_energy_distance_correlation, "maximize"),
        "hellinger_distance": (
            lambda x, y: np.sqrt(np.sum((
                np.sqrt(np.histogram(x, bins=min(10, len(np.unique(x))) if len(np.unique(x)) >= 2 else 2,
                                    range=(min(x.min(), y.min()), max(x.max(), y.max()) + 1e-10),
                                    density=True)[0]) -
                np.sqrt(np.histogram(y, bins=min(10, len(np.unique(y))) if len(np.unique(y)) >= 2 else 2,
                                    range=(min(x.min(), y.min()), max(x.max(), y.max()) + 1e-10),
                                    density=True)[0])
            ) ** 2)) / np.sqrt(2) if len(np.unique(x)) >= 2 and len(np.unique(y)) >= 2 else float('inf'),
            "minimize"
        ),
        "bhattacharyya_distance": (
            lambda x, y: -np.log(np.sum(np.sqrt(
                np.histogram(x, bins=min(10, len(np.unique(x))) if len(np.unique(x)) >= 2 else 2,
                             range=(min(x.min(), y.min()), max(x.max(), y.max()) + 1e-10),
                             density=True)[0] *
                np.histogram(y, bins=min(10, len(np.unique(y))) if len(np.unique(y)) >= 2 else 2,
                             range=(min(x.min(), y.min()), max(x.max(), y.max()) + 1e-10),
                             density=True)[0]
            )) + 1e-10) if len(np.unique(x)) >= 2 and len(np.unique(y)) >= 2 else float('inf'),
            "minimize"
        ),
        "variation_of_information": (
            lambda x, y: (
                np.var(x) + np.var(y) - 2 * mutual_info_score(
                    custom_digitize(x, min(10, len(np.unique(x))) if len(np.unique(x)) >= 2 else 2),
                    custom_digitize(y, min(10, len(np.unique(y))) if len(np.unique(y)) >= 2 else 2)
                )
            ) if len(np.unique(x)) >= 2 and len(np.unique(y)) >= 2 else float('inf'),
            "minimize"
        )
    }

def default_loss_functions():
    """Define default loss functions for training."""
    return {
        "pearson": (
            lambda x: x,
            "maximize"
        ),
        "spearman": (
            lambda x: rankdata(x, method='average'),
            "maximize"
        ),
        "kendall": (
            lambda x: rankdata(x, method='average'),
            "maximize"
        ),
        "mutual_information": (
            lambda x: custom_digitize(x, min(10, len(np.unique(x))) if len(np.unique(x)) >= 2 else 2),
            "maximize"
        ),
        "normalized_mutual_information": (
            lambda x: custom_digitize(x, min(10, len(np.unique(x))) if len(np.unique(x)) >= 2 else 2),
            "maximize"
        ),
        "conditional_mutual_information": (
            lambda x: custom_digitize(x, min(10, len(np.unique(x))) if len(np.unique(x)) >= 2 else 2),
            "maximize"
        ),
        "maximal_information_coefficient": (
            lambda x: custom_digitize(x, min(10, int(np.ceil(len(x) ** 0.6))) if len(np.unique(x)) >= 2 else 2),
            "maximize"
        ),
        "symmetric_uncertainty": (
            lambda x: custom_digitize(x, min(10, len(np.unique(x))) if len(np.unique(x)) >= 2 else 2),
            "maximize"
        ),
        "total_correlation": (
            lambda x: custom_digitize(x, min(10, len(np.unique(x))) if len(np.unique(x)) >= 2 else 2),
            "maximize"
        ),
        "distance_correlation": (
            lambda x: (x - np.mean(x)) / (np.std(x) + 1e-10),
            "maximize"
        ),
        "hsic": (
            lambda x: (x - np.mean(x)) / (np.std(x) + 1e-10),
            "maximize"
        ),
        "copula": (
            lambda x: rankdata(x, method='average') / (len(x) + 1),
            "maximize"
        ),
        "gini_correlation": (
            lambda x: rankdata(x, method='average'),
            "maximize"
        ),
        "fisher_score": (
            lambda x: custom_digitize(x, min(10, len(np.unique(x))) if len(np.unique(x)) >= 2 else 2),
            "maximize"
        ),
        "relief_score": (
            lambda x: custom_digitize(x, min(10, len(np.unique(x))) if len(np.unique(x)) >= 2 else 2),
            "maximize"
        ),
        "anova_f_statistic": (
            lambda x: custom_digitize(x, min(10, len(np.unique(x))) if len(np.unique(x)) >= 2 else 2),
            "maximize"
        ),
        "chi_square_score": (
            lambda x: custom_digitize(x, min(10, len(np.unique(x))) if len(np.unique(x)) >= 2 else 2),
            "maximize"
        ),
        "partial_correlation": (
            lambda x: x,
            "maximize"
        ),
        "information_gain": (
            lambda x: custom_digitize(x, min(10, len(np.unique(x))) if len(np.unique(x)) >= 2 else 2),
            "maximize"
        ),
        "gain_ratio": (
            lambda x: custom_digitize(x, min(10, len(np.unique(x))) if len(np.unique(x)) >= 2 else 2),
            "maximize"
        ),
        "energy_distance_correlation": (
            lambda x: (x - np.mean(x)) / (np.std(x) + 1e-10),
            "maximize"
        ),
        "hellinger_distance": (
            lambda x: custom_digitize(x, min(10, len(np.unique(x))) if len(np.unique(x)) >= 2 else 2),
            "minimize"
        ),
        "bhattacharyya_distance": (
            lambda x: custom_digitize(x, min(10, len(np.unique(x))) if len(np.unique(x)) >= 2 else 2),
            "minimize"
        ),
        "variation_of_information": (
            lambda x: custom_digitize(x, min(10, len(np.unique(x))) if len(np.unique(x)) >= 2 else 2),
            "minimize"
        )
    }

def train_model(model_instance, X, y, loss_function):
    """Train a model with a specified loss function."""
    y_transformed = loss_function(y) if loss_function else y
    model_instance.fit(X, y_transformed)
    return model_instance

def get_feature_combinations(features, level):
    """Generate all possible feature combinations up to specified level."""
    all_combinations = []
    for r in range(1, min(level + 1, len(features) + 1)):
        all_combinations.extend(combinations(features, r))
    return [list(combo) for combo in all_combinations]

def InformationRepurposedFeaturizer(
    df,
    models=None,
    loss_functions=None,
    metrics=None,
    prediction_mode="all",
    top_n=5,
    score_key="mutual_information",
    specific_metrics=None,
    level=1,
    save_models=False,
    save_results_file=None
):
    """
    Train and evaluate models with custom parameters and feature combinations.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame with features and 'target' column.
    - models (list): List of (model_name, model_instance) tuples. Default: LinearRegression, RandomForest, XGBoost, GradientBoosting.
    - loss_functions (dict): Dict of metric_name: (loss_function, direction). Default: pearson, spearman, kendall, hellinger, bhattacharyya.
    - metrics (dict): Dict of metric_name: (metric_function, direction). Default: comprehensive set of correlation/distance metrics.
    - prediction_mode (str): 'top_n', 'specific_metrics', or 'all'. Default: 'all'.
    - top_n (int): Number of top predictions to include if prediction_mode='top_n'. Default: 5.
    - score_key (str): Metric to rank predictions (e.g., 'mutual_information'). Default: 'mutual_information'.
    - specific_metrics (list): Metrics to include if prediction_mode='specific_metrics'. Default: None.
    - level (int): Maximum number of features to combine (1 for single features, 2 for pairs, etc.). Default: 1.
    - save_models (bool): If True, save trained models to pkl files and return them. Default: False.
    - save_results_file (str): If provided, save printed results to this file. Default: None.
    
    Returns:
    - result_df (pd.DataFrame): Original data + selected predictions.
    - metric_scores_df (pd.DataFrame): Model-metric performance scores.
    - feature_mi (dict): Mutual information of features with target.
    - trained_models (dict): Trained models if save_models=True, else None.
    """
    # Validate inputs
    if "target" not in df.columns:
        raise ValueError("DataFrame must contain a 'target' column.")
    
    # Set default models
    if models is None:
        models = [
            ("linear_regressor", LinearRegression()),
            ("random_forest", RandomForestRegressor(random_state=42, n_estimators=50)),
            ("xg_boost", XGBRegressor(random_state=42, n_estimators=50)),
            ("gradient_boost", GradientBoostingRegressor(random_state=42, n_estimators=50))
        ]
    
    # Set default loss functions and metrics
    loss_functions = default_loss_functions() if loss_functions is None else loss_functions
    metrics = default_metrics() if metrics is None else metrics
    
    # Create directory for saving models
    if save_models:
        os.makedirs("saved_models", exist_ok=True)
    
    # Get feature combinations
    features = [col for col in df.columns if col != "target"]
    feature_combos = get_feature_combinations(features, level)
    
    # Preprocess data
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    
    y = df["target"].values
    
    # Compute mutual information for original features
    feature_mi = {}
    X_full = df.drop(columns=["target"])
    X_imputed_full = imputer.fit_transform(X_full)
    for col in X_full.columns:
        mi = mutual_info_regression(X_imputed_full[:, X_full.columns.get_loc(col)].reshape(-1, 1), y)[0]
        feature_mi[col] = mi
    
    # Store predictions, scores, and models
    predictions = {}
    mi_scores = {}
    metric_scores = []
    trained_models = {} if save_models else None
    
    # Train and evaluate each model for each metric and feature combination
    for combo in feature_combos:
        combo_name = "_".join(combo)
        X = df[combo]
        X_imputed = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imputed)
        
        for model_name, model_instance in models:
            for metric_name, (metric_function, metric_direction) in metrics.items():
                # Get loss function
                loss_function, _ = loss_functions.get(metric_name, (lambda x: x, "maximize"))
                loss_function_name = metric_name if metric_name in loss_functions else "identity"
                
                # Train model
                model = train_model(model_instance, X_scaled, y, loss_function)
                y_pred = model.predict(X_scaled)
                
                # Store predictions
                pred_name = f"{combo_name}_{model_name}_{metric_name}_{loss_function_name}"
                predictions[pred_name] = y_pred
                
                # Compute mutual information
                mi_score = mutual_info_regression(y.reshape(-1, 1), y_pred)[0]
                mi_scores[pred_name] = mi_score
                
                # Compute metric score
                try:
                    score = metric_function(y, y_pred)
                except:
                    score = 0.0 if metric_direction == "maximize" else float('inf')
                
                metric_scores.append({
                    "feature_combo": combo_name,
                    "model": model_name,
                    "metric": metric_name,
                    "loss_function": loss_function_name,
                    "score": score,
                    "mutual_information": mi_score,
                    "direction": metric_direction
                })
    
    # Select predictions based on prediction_mode
    result_df = df.copy()
    selected_pred_names = []
    
    if prediction_mode == "all":
        selected_pred_names = list(predictions.keys())
        for pred_name in predictions:
            result_df[pred_name] = predictions[pred_name]
    elif prediction_mode == "top_n":
        # Rank by score_key
        top_preds = sorted(
            mi_scores.items(),
            key=lambda x: (
                metric_scores[[f"{s['feature_combo']}_{s['model']}_{s['metric']}_{s['loss_function']}" 
                               for s in metric_scores].index(x[0])]["score"]
                if score_key in metrics and metrics[score_key][1] == "maximize"
                else -metric_scores[[f"{s['feature_combo']}_{s['model']}_{s['metric']}_{s['loss_function']}" 
                                    for s in metric_scores].index(x[0])]["score"]
                if score_key in metrics
                else x[1]
            ),
            reverse=True
        )[:top_n]
        selected_pred_names = [pred_name for pred_name, _ in top_preds]
        for pred_name, _ in top_preds:
            result_df[pred_name] = predictions[pred_name]
    elif prediction_mode == "specific_metrics":
        if specific_metrics is None:
            raise ValueError("specific_metrics must be provided when prediction_mode='specific_metrics'.")
        for pred_name in predictions:
            metric = pred_name.split("_")[-2]
            if metric in specific_metrics:
                selected_pred_names.append(pred_name)
                result_df[pred_name] = predictions[pred_name]
    
    # Save models if requested
    if save_models:
        for combo in feature_combos:
            combo_name = "_".join(combo)
            X = df[combo]
            X_imputed = imputer.fit_transform(X)
            X_scaled = scaler.fit_transform(X_imputed)
            
            for model_name, model_instance in models:
                for metric_name, (metric_function, metric_direction) in metrics.items():
                    loss_function, _ = loss_functions.get(metric_name, (lambda x: x, "maximize"))
                    loss_function_name = metric_name if metric_name in loss_functions else "identity"
                    pred_name = f"{combo_name}_{model_name}_{metric_name}_{loss_function_name}"
                    
                    if pred_name in selected_pred_names:
                        # Train model again to ensure we save the exact model used
                        model = train_model(model_instance, X_scaled, y, loss_function)
                        trained_models[pred_name] = model
                        
                        # Save model to pkl file
                        model_filename = f"saved_models/{pred_name}.pkl"
                        with open(model_filename, 'wb') as f:
                            cloudpickle.dump(model, f)
    
    # Create metric scores DataFrame
    metric_scores_df = pd.DataFrame(metric_scores)
    
    # Save printed results if requested
    output_str = ""
    if save_results_file:
        output_str += "Feature Mutual Information with Target:\n"
        for feature, mi in feature_mi.items():
            output_str += f"{feature}: {mi:.4f}\n"
        output_str += "\nPredictions DataFrame:\n"
        output_str += result_df.head().to_string() + "\n"
        output_str += "\nMetric Scores DataFrame:\n"
        output_str += metric_scores_df.to_string() + "\n"
        output_str += f"\nNumber of trained models: {len(trained_models) if trained_models else 0}\n"
        
        with open(save_results_file, 'w') as f:
            f.write(output_str)
    
    return result_df, metric_scores_df, feature_mi, trained_models

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_df = pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "feature3": np.random.randn(100),
        "target": np.random.randn(100) + np.random.randn(100) * 0.5
    })
    
    # Define custom model
    from sklearn.tree import DecisionTreeRegressor
    custom_models = [
        ("decision_tree", DecisionTreeRegressor(random_state=42)),
        ("xg_boost", XGBRegressor(random_state=42, n_estimators=50))
    ]
    
    # Define custom loss function
    custom_loss = {
        "pearson": (lambda x: x, "maximize"),
        "spearman": (lambda x: rankdata(x, method='average'), "maximize"),
        "custom_metric": (lambda x: np.log1p(np.abs(x)), "maximize")
    }
    
    # Define custom metric
    custom_metrics = {
        "pearson": (lambda x, y: pearsonr(x, y)[0], "maximize"),
        "spearman": (lambda x, y: spearmanr(x, y)[0], "maximize"),
        "custom_metric": (lambda x, y: np.mean((x - y) ** 2), "minimize")
    }
    
    # Run with custom settings
    result_df, metric_scores_df, feature_mi, trained_models = InformationRepurposedFeaturizer(
        df=sample_df,
        models=custom_models,
        loss_functions=custom_loss,
        metrics=custom_metrics,
        prediction_mode="top_n",
        top_n=3,
        score_key="spearman",
        level=1,
        save_models=True,
        save_results_file="model_results.txt"
    )
    
    print("Feature Mutual Information with Target:")
    for feature, mi in feature_mi.items():
        print(f"{feature}: {mi:.4f}")
    print("\nPredictions DataFrame:")
    print(result_df.head())
    print("\nMetric Scores DataFrame:")
    print(metric_scores_df)
    print("\nNumber of trained models:", len(trained_models) if trained_models else 0)
