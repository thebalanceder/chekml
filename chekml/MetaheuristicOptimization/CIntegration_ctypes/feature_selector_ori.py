import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression, RFE
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet
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

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

def FeatureSelector(dataframes, top_n=5, problem_type="classification", save_to_csv=False):
    """
    Selects feature combinations from multiple DataFrames by merging them and ensuring a single 'target' column.
    Returns only DataFrames with unique feature combinations. Prints merged DataFrame and optionally saves
    each unique DataFrame to a CSV file.
    
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
    X = merged_df.drop(columns=["target"])
    y = merged_df["target"]
    
    # Initialize result dictionary
    selected_dfs = {}
    
    # Standardize features for methods requiring scaled data
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Scale features to non-negative for Chi-Square
    chi2_scaler = MinMaxScaler()
    X_chi2 = pd.DataFrame(chi2_scaler.fit_transform(X), columns=X.columns)
    
    # 1. Filter Methods
    logging.info("Applying filter methods...")
    
    # Variance Thresholding
    vt = VarianceThreshold(threshold=0.0)
    vt.fit(X_scaled)
    vt_features = X.columns[vt.get_support()].tolist()
    if len(vt_features) > top_n:
        vt_features = vt_features[:top_n]
    selected_dfs["VarianceThreshold"] = pd.concat([X[vt_features], y], axis=1)
    
    # Correlation Thresholding
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.8)]
    corr_features = [col for col in X.columns if col not in to_drop]
    if len(corr_features) > top_n:
        corr_features = corr_features[:top_n]
    selected_dfs["CorrelationThreshold"] = pd.concat([X[corr_features], y], axis=1)
    
    # Statistical Tests
    if problem_type == "classification":
        # Chi-Square Test (use X_chi2 for non-negative values)
        selector = SelectKBest(score_func=chi2, k=min(top_n, X.shape[1]))
        selector.fit(X_chi2, y)
        chi2_features = X.columns[selector.get_support()].tolist()
        selected_dfs["ChiSquare"] = pd.concat([X[chi2_features], y], axis=1)
        
        # Mutual Information
        selector = SelectKBest(score_func=mutual_info_classif, k=min(top_n, X.shape[1]))
        selector.fit(X_scaled, y)
        mi_features = X.columns[selector.get_support()].tolist()
        selected_dfs["MutualInfoClassif"] = pd.concat([X[mi_features], y], axis=1)
        
        # ANOVA F-test
        selector = SelectKBest(score_func=f_classif, k=min(top_n, X.shape[1]))
        selector.fit(X_scaled, y)
        anova_features = X.columns[selector.get_support()].tolist()
        selected_dfs["ANOVAClassif"] = pd.concat([X[anova_features], y], axis=1)
    
    else:
        # Mutual Information
        selector = SelectKBest(score_func=mutual_info_regression, k=min(top_n, X.shape[1]))
        selector.fit(X_scaled, y)
        mi_features = X.columns[selector.get_support()].tolist()
        selected_dfs["MutualInfoRegress"] = pd.concat([X[mi_features], y], axis=1)
        
        # ANOVA F-test
        selector = SelectKBest(score_func=f_regression, k=min(top_n, X.shape[1]))
        selector.fit(X_scaled, y)
        anova_features = X.columns[selector.get_support()].tolist()
        selected_dfs["ANOVARegress"] = pd.concat([X[anova_features], y], axis=1)
    
    # 2. Wrapper Methods
    logging.info("Applying wrapper methods...")
    estimator = LogisticRegression() if problem_type == "classification" else LinearRegression()
    rfe = RFE(estimator=estimator, n_features_to_select=top_n)
    rfe.fit(X_scaled, y)
    rfe_features = X.columns[rfe.support_].tolist()
    selected_dfs["RFE"] = pd.concat([X[rfe_features], y], axis=1)
    
    # 3. Embedded Methods
    logging.info("Applying embedded methods...")
    
    # Lasso
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_scaled, y)
    lasso_features = X.columns[lasso.coef_ != 0].tolist()
    if len(lasso_features) > top_n:
        lasso_features = lasso_features[:top_n]
    selected_dfs["Lasso"] = pd.concat([X[lasso_features], y], axis=1)
    
    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)
    ridge_coef = pd.Series(np.abs(ridge.coef_), index=X.columns)
    ridge_features = ridge_coef.nlargest(top_n).index.tolist()
    selected_dfs["Ridge"] = pd.concat([X[ridge_features], y], axis=1)
    
    # ElasticNet
    elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
    elastic.fit(X_scaled, y)
    elastic_features = X.columns[elastic.coef_ != 0].tolist()
    if len(elastic_features) > top_n:
        elastic_features = elastic_features[:top_n]
    selected_dfs["ElasticNet"] = pd.concat([X[elastic_features], y], axis=1)
    
    # RandomForest
    rf = RandomForestClassifier() if problem_type == "classification" else RandomForestRegressor()
    rf.fit(X, y)
    rf_importances = pd.Series(rf.feature_importances_, index=X.columns)
    rf_features = rf_importances.nlargest(top_n).index.tolist()
    selected_dfs["RandomForest"] = pd.concat([X[rf_features], y], axis=1)
    
    # XGBoost
    xgb_model = xgb.XGBClassifier() if problem_type == "classification" else xgb.XGBRegressor()
    xgb_model.fit(X, y)
    xgb_importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
    xgb_features = xgb_importances.nlargest(top_n).index.tolist()
    selected_dfs["XGBoost"] = pd.concat([X[xgb_features], y], axis=1)
    
    # 4. Metaheuristic Methods
    logging.info("Applying metaheuristic methods...")
    
    def objective_function(solution, X, y, problem_type):
        """Objective function for metaheuristic feature selection."""
        if hasattr(solution, "contents"):
            solution = np.ctypeslib.as_array(solution, shape=(X.shape[1],))
        
        if isinstance(solution, np.ndarray):
            solution = solution.tolist()
        
        selected = [i for i, val in enumerate(solution) if val > 0.5]
        if not selected:
            return float('inf')
        
        selected_features = X.columns[selected].tolist()
        X_subset = X[selected_features]
        
        model = LogisticRegression() if problem_type == "classification" else LinearRegression()
        scoring = "accuracy" if problem_type == "classification" else "neg_mean_squared_error"
        
        try:
            scores = cross_val_score(model, X_subset, y, cv=3, scoring=scoring)
            return -scores.mean()
        except Exception as e:
            logging.error(f"Error in metaheuristic optimization: {e}")
            return float('inf')
    
    optimizer = Wrapper(
        dim=X.shape[1],
        bounds=[(0, 1)] * X.shape[1],
        population_size=20,
        max_iter=50,
        method="DISO"
    )
    
    try:
        optimizer.optimize(lambda sol: objective_function(sol, X, y, problem_type))
        best_solution, _ = optimizer.get_best_solution()
        selected_indices = [i for i, val in enumerate(best_solution) if val > 0.5]
        meta_features = X.columns[selected_indices].tolist()
        if len(meta_features) > top_n:
            meta_features = meta_features[:top_n]
        if meta_features:
            selected_dfs["Metaheuristic"] = pd.concat([X[meta_features], y], axis=1)
    except Exception as e:
        logging.error(f"Metaheuristic optimization failed: {e}")
    finally:
        optimizer.free()
    
    # Filter for unique feature combinations and optionally save to CSV
    unique_dfs = {}
    seen_features = set()
    for method, df in selected_dfs.items():
        # Get features excluding 'target'
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
