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
import copy
from joblib import Parallel, delayed
from . import metrics  # Pybind11 module with OpenCL
warnings.filterwarnings("ignore")


class InformationRepurposedFeaturizerWrapper:
    """Wrapper exposing a fit/transform API for the fast IRF implementation.

    Works similarly to the slow wrapper: `fit(df)` runs the functional
    `InformationRepurposedFeaturizer` with `save_models=True` and records the
    selected prediction names; `transform(df)` loads saved models and applies
    them to new data.
    """

    def __init__(self, models=None, loss_functions=None, metrics=None,
                 prediction_mode="top_n", top_n=5, score_key="mutual_information",
                 specific_metrics=None, level=1, save_results_file=None, n_jobs=-1):
        self.models = models
        self.loss_functions = loss_functions
        self.metrics = metrics
        self.prediction_mode = prediction_mode
        self.top_n = top_n
        self.score_key = score_key
        self.specific_metrics = specific_metrics
        self.level = level
        self.save_results_file = save_results_file
        self.n_jobs = n_jobs
        self.selected_pred_names = None
        self.trained_models = None

    def fit(self, df):
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
            save_results_file=self.save_results_file,
            n_jobs=self.n_jobs
        )

        base_cols = [c for c in df.columns if c != 'target']
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
            if len(parts) < 4:
                new_df[pred_name] = np.nan
                continue
            combo_parts = parts[:-3]
            combo_cols = combo_parts

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
    return metrics.compute_mic(x, y, max_bins)

def compute_copula_measure(x, y):
    """Compute a simplified copula-based dependence measure."""
    return metrics.compute_copula_measure(x, y)

def compute_distance_correlation(x, y):
    """Compute distance correlation."""
    return metrics.compute_distance_correlation(x, y)

def compute_hsic(x, y):
    """Compute Hilbert-Schmidt Independence Criterion (HSIC)."""
    return metrics.compute_hsic(x, y)

def compute_energy_distance_correlation(x, y):
    """Compute energy distance correlation."""
    return metrics.compute_energy_distance_correlation(x, y)

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

def process_combination(combo, models, metrics, loss_functions, df, y, imputer, scaler):
    """Process a single feature combination, model, and metric in parallel."""
    combo_name = "_".join(combo)
    X = df[combo].values
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    
    predictions = {}
    mi_scores = {}
    metric_scores = []
    
    for model_name, model_instance in models:
        model_instance = copy.deepcopy(model_instance)
        for metric_name, (metric_function, metric_direction) in metrics.items():
            loss_function, _ = loss_functions.get(metric_name, (lambda x: x, "maximize"))
            loss_function_name = metric_name if metric_name in loss_functions else "identity"
            
            model = train_model(model_instance, X_scaled, y, loss_function)
            y_pred = model.predict(X_scaled)
            
            if metric_name in ["hellinger_distance", "bhattacharyya_distance"]:
                y_pred = np.interp(y_pred, (y_pred.min(), y_pred.max()), (y.min(), y.max()))
            
            pred_name = f"{combo_name}_{model_name}_{metric_name}_{loss_function_name}"
            predictions[pred_name] = y_pred
            
            mi_score = mutual_info_regression(y.reshape(-1, 1), y_pred)[0]
            mi_scores[pred_name] = mi_score
            
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
    
    return predictions, mi_scores, metric_scores

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
    save_results_file=None,
    n_jobs=-1
):
    """
    Train and evaluate models with custom parameters and feature combinations.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame with features and 'target' column.
    - models (list): List of (model_name, model_instance) tuples.
    - loss_functions (dict): Dict of metric_name: (loss_function, direction).
    - metrics (dict): Dict of metric_name: (metric_function, direction).
    - prediction_mode (str): 'top_n', 'specific_metrics', or 'all'.
    - top_n (int): Number of top predictions to include if prediction_mode='top_n'.
    - score_key (str): Metric to rank predictions.
    - specific_metrics (list): Metrics to include if prediction_mode='specific_metrics'.
    - level (int): Maximum number of features to combine.
    - save_models (bool): If True, save trained models to pkl files.
    - save_results_file (str): If provided, save printed results to this file.
    - n_jobs (int): Number of parallel jobs for feature combinations.
    
    Returns:
    - result_df (pd.DataFrame): Original data + selected predictions.
    - metric_scores_df (pd.DataFrame): Model-metric performance scores.
    - feature_mi (dict): Mutual information of features with target.
    - trained_models (dict): Trained models if save_models=True, else None.
    """
    if "target" not in df.columns:
        raise ValueError("DataFrame must contain a 'target' column.")
    
    if models is None:
        models = [
            ("linear_regressor", LinearRegression()),
            ("random_forest", RandomForestRegressor(random_state=42, n_estimators=50)),
            ("xg_boost", XGBRegressor(random_state=42, n_estimators=50)),
            ("gradient_boost", GradientBoostingRegressor(random_state=42, n_estimators=50))
        ]
    
    loss_functions = default_loss_functions() if loss_functions is None else loss_functions
    metrics = default_metrics() if metrics is None else metrics
    
    if save_models:
        os.makedirs("saved_models", exist_ok=True)
    
    features = [col for col in df.columns if col != "target"]
    feature_combos = get_feature_combinations(features, level)
    
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    y = df["target"].values
    
    feature_mi = {}
    X_full = df.drop(columns=["target"]).values
    X_imputed_full = imputer.fit_transform(X_full)
    for idx, col in enumerate(df.columns[:-1]):
        mi = mutual_info_regression(X_imputed_full[:, idx].reshape(-1, 1), y)[0]
        feature_mi[col] = mi
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_combination)(
            combo, models, metrics, loss_functions, df, y, copy.deepcopy(imputer), copy.deepcopy(scaler)
        )
        for combo in feature_combos
    )
    
    predictions = {}
    mi_scores = {}
    metric_scores = []
    for pred_dict, mi_dict, score_list in results:
        predictions.update(pred_dict)
        mi_scores.update(mi_dict)
        metric_scores.extend(score_list)
    
    result_df = df.copy()
    selected_pred_names = []
    
    if prediction_mode == "all":
        selected_pred_names = list(predictions.keys())
        for pred_name in predictions:
            result_df[pred_name] = predictions[pred_name]
    elif prediction_mode == "top_n":
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
    
    trained_models = {} if save_models else None
    if save_models:
        for combo in feature_combos:
            combo_name = "_".join(combo)
            X = df[combo].values
            X_imputed = imputer.fit_transform(X)
            X_scaled = scaler.fit_transform(X_imputed)
            
            for model_name, model_instance in models:
                model_instance = copy.deepcopy(model_instance)
                for metric_name, (metric_function, metric_direction) in metrics.items():
                    loss_function, _ = loss_functions.get(metric_name, (lambda x: x, "maximize"))
                    loss_function_name = metric_name if metric_name in loss_functions else "identity"
                    pred_name = f"{combo_name}_{model_name}_{metric_name}_{loss_function_name}"
                    
                    if pred_name in selected_pred_names:
                        model = train_model(model_instance, X_scaled, y, loss_function)
                        trained_models[pred_name] = model
                        
                        model_filename = f"saved_models/{pred_name}.pkl"
                        with open(model_filename, 'wb') as f:
                            cloudpickle.dump(model, f)
    
    metric_scores_df = pd.DataFrame(metric_scores)
    
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
    np.random.seed(42)
    sample_df = pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "feature3": np.random.randn(100),
        "target": np.random.randn(100) + np.random.randn(100) * 0.5
    })
    
    from sklearn.tree import DecisionTreeRegressor
    custom_models = [
        ("decision_tree", DecisionTreeRegressor(random_state=42)),
        ("xg_boost", XGBRegressor(random_state=42, n_estimators=50))
    ]
    
    custom_loss = {
        "pearson": (lambda x: x, "maximize"),
        "spearman": (lambda x: rankdata(x, method='average'), "maximize"),
        "custom_metric": (lambda x: np.log1p(np.abs(x)), "maximize")
    }
    
    custom_metrics = {
        "pearson": (lambda x, y: pearsonr(x, y)[0], "maximize"),
        "spearman": (lambda x, y: spearmanr(x, y)[0], "maximize"),
        "custom_metric": (lambda x, y: np.mean((x - y) ** 2), "minimize")
    }
    
    result_df, metric_scores_df, feature_mi, trained_models = InformationRepurposedFeaturizer(
        df=sample_df,
        models=custom_models,
        loss_functions=custom_loss,
        metrics=custom_metrics,
        prediction_mode="top_n",
        top_n=3,
        score_key="spearman",
        level=2,
        save_models=False,
        save_results_file="model_results.txt",
        n_jobs=4
    )
    
    print("Feature Mutual Information with Target:")
    for feature, mi in feature_mi.items():
        print(f"{feature}: {mi:.4f}")
    print("\nPredictions DataFrame:")
    print(result_df.head())
    print("\nMetric Scores DataFrame:")
    print(metric_scores_df)
    print("\nNumber of trained models:", len(trained_models) if trained_models else 0)
