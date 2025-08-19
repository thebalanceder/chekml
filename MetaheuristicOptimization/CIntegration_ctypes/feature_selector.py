import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, f_regression, RFE
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error
import xgboost as xgb
from wrapper import Wrapper  # From uploaded files
import logging
import warnings
import os
import ctypes
from joblib import Parallel, delayed

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# Load C shared library
try:
    lib = ctypes.CDLL("./libfeature_selection.so")
except OSError as e:
    raise RuntimeError(f"Failed to load libfeature_selection.so: {e}")

# Define C function signatures
lib.compute_variance.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
lib.compute_variance.restype = None

lib.select_top_features.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
lib.select_top_features.restype = None

lib.compute_correlation.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
lib.compute_correlation.restype = None

lib.compute_chi2.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
lib.compute_chi2.restype = None

def FeatureSelector(dataframes, top_n=5, problem_type="classification", save_to_csv=False):
    """
    Selects feature combinations from multiple DataFrames by merging them and ensuring a single 'target' column.
    Returns only DataFrames with unique feature combinations. Prints merged DataFrame and optionally saves
    each unique DataFrame to a CSV file. Optimized for speed with C extensions and parallelization.
    
    Parameters:
    - dataframes (list): List of pandas DataFrames, each with a 'target' column.
    - top_n (int): Number of top features to select for filter methods.
    - problem_type (str): 'classification' or 'regression'.
    - save_to_csv (bool): If True, save each unique DataFrame to a CSV file (default: False).
    
    Returns:
    - dict: Dictionary of DataFrames, each with unique selected features and 'target'.
    """
    # Validate inputs
    if not isinstance(dataframes, list) or not all(isinstance(df, pd.DataFrame) for df in dataframes):
        raise ValueError("dataframes must be a list of pandas DataFrames")
    if not dataframes:
        raise ValueError("At least one DataFrame must be provided")
    if problem_type not in ["classification", "regression"]:
        raise ValueError("problem_type must be 'classification' or 'regression'")
    
    # Check for target column in each DataFrame
    merged_df = None
    target_cols_all = []
    for i, df in enumerate(dataframes):
        target_cols = [col for col in df.columns if col.lower() == "target"]
        if not target_cols:
            raise ValueError(f"No 'target' column found in DataFrame {i}")
        target_cols_all.append(target_cols)
    
    # Merge DataFrames
    logging.info("Merging DataFrames...")
    try:
        # Normalize target columns
        dfs_normalized = []
        for df, target_cols in zip(dataframes, target_cols_all):
            df_copy = df.copy()
            df_copy = df_copy.rename(columns={target_cols[0]: "target"})
            if len(target_cols) > 1:
                df_copy = df_copy.drop(columns=target_cols[1:])
            dfs_normalized.append(df_copy)
        
        # Concatenate DataFrames vertically
        merged_df = pd.concat(dfs_normalized, axis=0, ignore_index=True)
        
        # Verify target consistency
        for i, (df, target_cols) in enumerate(zip(dataframes, target_cols_all)):
            for col in target_cols:
                if not df[col].equals(dataframes[0][target_cols_all[0][0]]):
                    raise ValueError(f"Target column {col} in DataFrame {i} differs from others")
    
    except Exception as e:
        raise ValueError(f"Error merging DataFrames: {e}")
    
    if merged_df is None or merged_df.empty:
        raise ValueError("Merged DataFrame is empty")
    
    # Handle NaNs
    logging.info("Checking for NaNs in merged DataFrame...")
    nan_count = merged_df.isna().sum().sum()
    if nan_count > 0:
        logging.info(f"Found {nan_count} NaN values. Imputing with mean strategy...")
        imputer = SimpleImputer(strategy="mean")
        merged_df = pd.DataFrame(
            imputer.fit_transform(merged_df),
            columns=merged_df.columns
        )
    
    # Print merged DataFrame info
    logging.info(f"Merged DataFrame shape: {merged_df.shape}")
    logging.info(f"Merged DataFrame NaN count: {merged_df.isna().sum().sum()}")
    print("\nMerged DataFrame:")
    print(merged_df.head())
    
    # Separate features and target
    X = merged_df.drop(columns=["target"]).values  # Convert to numpy
    y = merged_df["target"].values
    feature_names = merged_df.drop(columns=["target"]).columns.tolist()
    
    # Cache scaled data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    chi2_scaler = MinMaxScaler()
    X_chi2 = chi2_scaler.fit_transform(X)
    
    # Initialize result dictionary
    selected_dfs = {}
    
    # C-based Variance Thresholding
    def variance_threshold(X, top_n, feature_names):
        rows, cols = X.shape
        variances = np.zeros(cols, dtype=np.float64)
        variances_ptr = variances.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        X_contiguous = np.ascontiguousarray(X, dtype=np.float64)
        X_ptr = X_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        lib.compute_variance(X_ptr, rows, cols, variances_ptr)
        
        selected_indices = np.zeros(top_n, dtype=np.int32)
        selected_indices_ptr = selected_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        lib.select_top_features(variances_ptr, cols, min(top_n, cols), selected_indices_ptr)
        
        vt_features = [feature_names[i] for i in selected_indices if i < cols]
        return pd.DataFrame(X[:, selected_indices], columns=vt_features), vt_features
    
    # C-based Correlation Thresholding
    def correlation_threshold(X, top_n, feature_names, threshold=0.8):
        rows, cols = X.shape
        corr_matrix = np.zeros((cols, cols), dtype=np.float64)
        corr_matrix_ptr = corr_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        X_contiguous = np.ascontiguousarray(X, dtype=np.float64)
        X_ptr = X_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        lib.compute_correlation(X_ptr, rows, cols, corr_matrix_ptr)
        
        to_drop = []
        for j1 in range(cols):
            for j2 in range(j1 + 1, cols):
                if abs(corr_matrix[j1, j2]) > threshold:
                    to_drop.append(j2)
        corr_features = [feature_names[j] for j in range(cols) if j not in set(to_drop)]
        if len(corr_features) > top_n:
            corr_features = corr_features[:top_n]
        indices = [feature_names.index(f) for f in corr_features]
        return pd.DataFrame(X[:, indices], columns=corr_features), corr_features
    
    # C-based Chi-Square
    def chi2_select(X, y, top_n, feature_names):
        rows, cols = X.shape
        n_classes = len(np.unique(y))
        scores = np.zeros(cols, dtype=np.float64)
        scores_ptr = scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        X_contiguous = np.ascontiguousarray(X, dtype=np.float64)
        X_ptr = X_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        y_contiguous = np.ascontiguousarray(y, dtype=np.int32)
        y_ptr = y_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        
        lib.compute_chi2(X_ptr, rows, cols, y_ptr, n_classes, scores_ptr)
        
        selected_indices = np.zeros(top_n, dtype=np.int32)
        selected_indices_ptr = selected_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        lib.select_top_features(scores_ptr, cols, min(top_n, cols), selected_indices_ptr)
        
        chi2_features = [feature_names[i] for i in selected_indices if i < cols]
        return pd.DataFrame(X[:, selected_indices], columns=chi2_features), chi2_features
    
    # Execute C-based methods sequentially
    logging.info("Applying C-based filter methods...")
    df_vt, vt_features = variance_threshold(X_scaled, top_n, feature_names)
    selected_dfs["VarianceThreshold"] = pd.concat([df_vt, pd.Series(y, name="target")], axis=1)
    
    df_corr, corr_features = correlation_threshold(X, top_n, feature_names)
    selected_dfs["CorrelationThreshold"] = pd.concat([df_corr, pd.Series(y, name="target")], axis=1)
    
    if problem_type == "classification":
        df_chi2, chi2_features = chi2_select(X_chi2, y, top_n, feature_names)
        selected_dfs["ChiSquare"] = pd.concat([df_chi2, pd.Series(y, name="target")], axis=1)
    
    # Parallelized feature selection methods
    def select_features(method, X, X_scaled, y, top_n, feature_names, problem_type):
        if method == "MutualInfo" and problem_type == "classification":
            selector = SelectKBest(score_func=mutual_info_classif, k=min(top_n, X.shape[1]))
            selector.fit(X_scaled, y)
            mi_features = [feature_names[i] for i in selector.get_support(indices=True)]
            return method, pd.DataFrame(X[:, selector.get_support(indices=True)], columns=mi_features), mi_features
        elif method == "ANOVA" and problem_type == "classification":
            selector = SelectKBest(score_func=f_classif, k=min(top_n, X.shape[1]))
            selector.fit(X_scaled, y)
            anova_features = [feature_names[i] for i in selector.get_support(indices=True)]
            return method, pd.DataFrame(X[:, selector.get_support(indices=True)], columns=anova_features), anova_features
        elif method == "MutualInfo" and problem_type == "regression":
            selector = SelectKBest(score_func=mutual_info_regression, k=min(top_n, X.shape[1]))
            selector.fit(X_scaled, y)
            mi_features = [feature_names[i] for i in selector.get_support(indices=True)]
            return method, pd.DataFrame(X[:, selector.get_support(indices=True)], columns=mi_features), mi_features
        elif method == "ANOVA" and problem_type == "regression":
            selector = SelectKBest(score_func=f_regression, k=min(top_n, X.shape[1]))
            selector.fit(X_scaled, y)
            anova_features = [feature_names[i] for i in selector.get_support(indices=True)]
            return method, pd.DataFrame(X[:, selector.get_support(indices=True)], columns=anova_features), anova_features
        elif method == "RFE":
            estimator = LogisticRegression(max_iter=100) if problem_type == "classification" else LinearRegression()
            rfe = RFE(estimator=estimator, n_features_to_select=top_n)
            rfe.fit(X_scaled, y)
            rfe_features = [feature_names[i] for i in np.where(rfe.support_)[0]]
            return method, pd.DataFrame(X[:, rfe.support_], columns=rfe_features), rfe_features
        elif method == "Lasso":
            lasso = Lasso(alpha=0.1)
            lasso.fit(X_scaled, y)
            lasso_features = [feature_names[i] for i in np.where(lasso.coef_ != 0)[0]]
            if len(lasso_features) > top_n:
                lasso_features = lasso_features[:top_n]
            indices = [feature_names.index(f) for f in lasso_features]
            return method, pd.DataFrame(X[:, indices], columns=lasso_features), lasso_features
        elif method == "RandomForest":
            rf = RandomForestClassifier(n_estimators=50) if problem_type == "classification" else RandomForestRegressor(n_estimators=50)
            rf.fit(X, y)
            rf_importances = pd.Series(rf.feature_importances_, index=feature_names)
            rf_features = rf_importances.nlargest(top_n).index.tolist()
            indices = [feature_names.index(f) for f in rf_features]
            return method, pd.DataFrame(X[:, indices], columns=rf_features), rf_features
        elif method == "XGBoost":
            xgb_model = xgb.XGBClassifier(n_estimators=50) if problem_type == "classification" else xgb.XGBRegressor(n_estimators=50)
            xgb_model.fit(X, y)
            xgb_importances = pd.Series(xgb_model.feature_importances_, index=feature_names)
            xgb_features = xgb_importances.nlargest(top_n).index.tolist()
            indices = [feature_names.index(f) for f in xgb_features]
            return method, pd.DataFrame(X[:, indices], columns=xgb_features), xgb_features
        return None, None, None
    
    # Define methods to run in parallel
    methods = ["RFE", "Lasso", "RandomForest", "XGBoost"]
    if problem_type == "classification":
        methods.extend(["MutualInfo", "ANOVA"])
    else:
        methods.extend(["MutualInfo", "ANOVA"])
    
    # Parallel execution
    logging.info("Applying parallelized feature selection methods...")
    results = Parallel(n_jobs=-1)(
        delayed(select_features)(method, X, X_scaled, y, top_n, feature_names, problem_type)
        for method in methods
    )
    
    # Collect results
    for method, df, features in results:
        if method and df is not None:
            selected_dfs[method] = pd.concat([df, pd.Series(y, name="target")], axis=1)
    
    # Metaheuristic Method (optimized)
    logging.info("Applying metaheuristic methods...")
    
    def objective_function(solution, X, y, problem_type):
        if hasattr(solution, "contents"):
            solution = np.ctypeslib.as_array(solution, shape=(X.shape[1],))
        
        if isinstance(solution, np.ndarray):
            solution = solution.tolist()
        
        selected = [i for i, val in enumerate(solution) if val > 0.5]
        if not selected:
            return float('inf')
        
        X_subset = X[:, selected]
        model = LogisticRegression(max_iter=100) if problem_type == "classification" else LinearRegression()
        scoring = "accuracy" if problem_type == "classification" else "neg_mean_squared_error"
        
        try:
            scores = cross_val_score(model, X_subset, y, cv=2, scoring=scoring)  # Reduced CV folds
            return -scores.mean()
        except Exception as e:
            logging.error(f"Error in metaheuristic optimization: {e}")
            return float('inf')
    
    optimizer = Wrapper(
        dim=X.shape[1],
        bounds=[(0, 1)] * X.shape[1],
        population_size=10,  # Reduced population
        max_iter=20,  # Reduced iterations
        method="DISO"
    )
    
    try:
        optimizer.optimize(lambda sol: objective_function(sol, X, y, problem_type))
        best_solution, _ = optimizer.get_best_solution()
        selected_indices = [i for i, val in enumerate(best_solution) if val > 0.5]
        meta_features = [feature_names[i] for i in selected_indices]
        if len(meta_features) > top_n:
            meta_features = meta_features[:top_n]
        if meta_features:
            indices = [feature_names.index(f) for f in meta_features]
            selected_dfs["Metaheuristic"] = pd.concat(
                [pd.DataFrame(X[:, indices], columns=meta_features), pd.Series(y, name="target")], axis=1
            )
    except Exception as e:
        logging.error(f"Metaheuristic optimization failed: {e}")
    finally:
        optimizer.free()
    
    # Filter for unique feature combinations and optionally save to CSV
    unique_dfs = {}
    seen_features = set()
    for method, df in selected_dfs.items():
        features = tuple(sorted([col for col in df.columns if col != "target"]))
        if features not in seen_features:
            unique_dfs[method] = df
            seen_features.add(features)
            if save_to_csv:
                csv_path = f"{method}_features.csv"
                df.to_csv(csv_path, index=False)
                logging.info(f"Saved DataFrame for {method} to {os.path.abspath(csv_path)}")
        else:
            logging.info(f"Skipping {method}: Duplicate feature combination {features}")
    
    if not unique_dfs:
        logging.warning("No unique feature combinations found")
    
    return unique_dfs
