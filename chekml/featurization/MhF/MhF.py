import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression, RFE
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, clone
import xgboost as xgb
import logging
import warnings
import torch
import torch.nn as nn
import torch.optim as optim

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


class MetaheuristicFeaturizerWrapper:
    """Wrapper providing a fit/transform API for the MetaheuristicFeaturizer.

    Usage:
      wrapper = MetaheuristicFeaturizerWrapper(...)
      wrapper.fit([train_df1, train_df2])
      selected = wrapper.transform(test_df)
    """
    def __init__(self, top_n=5, problem_type="classification", model=None, scorer=None,
                 wrapper_class=None, wrapper_method="DISO", wrapper_population_size=20,
                 wrapper_max_iter=50, **wrapper_kwargs):
        self.kwargs = dict(
            top_n=top_n,
            problem_type=problem_type,
            model=model,
            scorer=scorer,
            wrapper_class=wrapper_class,
            wrapper_method=wrapper_method,
            wrapper_population_size=wrapper_population_size,
            wrapper_max_iter=wrapper_max_iter,
            **wrapper_kwargs
        )
        self.selected_by_method = None

    def fit(self, dataframes):
        # Call functional MetaheuristicFeaturizer to compute selected DataFrames
        selected_dfs = MetaheuristicFeaturizer(dataframes, **self.kwargs)
        # Record selected columns for each method
        self.selected_by_method = {}
        for method, df in selected_dfs.items():
            cols = [c for c in df.columns if c != 'target']
            self.selected_by_method[method] = cols
        return self.selected_by_method

    def transform(self, df):
        if self.selected_by_method is None:
            raise RuntimeError('Featurizer has not been fitted. Call fit() first.')
        outputs = {}
        for method, cols in self.selected_by_method.items():
            missing = [c for c in cols if c not in df.columns]
            # Use available columns; missing ones will be filled with NaN
            present = [c for c in cols if c in df.columns]
            out_df = df[present].copy()
            for m in missing:
                out_df[m] = np.nan
            if 'target' in df.columns:
                out_df['target'] = df['target']
            outputs[method] = out_df[cols + (['target'] if 'target' in df.columns else [])]
        return outputs

class PyTorchWrapper(BaseEstimator):
    def __init__(self, model, criterion=None, optimizer=optim.Adam, lr=0.001, epochs=10, batch_size=32, device='cpu'):
        self.model = model
        self.criterion = criterion
        self.optimizer_class = optimizer
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.input_dim = None  # Will be set in fit

    def fit(self, X, y, **fit_params):
        # Convert DataFrame/Series to NumPy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        logging.info(f"PyTorchWrapper.fit: X shape={X.shape}, y shape={y.shape}, criterion={self.criterion}")
        
        # Dynamically adjust input layer if necessary
        input_dim = X.shape[1]
        if self.input_dim is None or self.input_dim != input_dim:
            self.input_dim = input_dim
            self.model.layer1 = nn.Linear(input_dim, self.model.layer1.out_features).to(self.device)
            logging.info(f"Adjusted SimpleNN input layer to {input_dim} features")

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device).view(-1, 1)  # Always float for regression
        
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.lr)
        self.model.train()
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Early stopping parameters
        best_loss = float('inf')
        patience = 3
        no_improve = 0
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
            epoch_loss /= len(dataset)
            # Early stopping
            if epoch_loss < best_loss - 1e-3:
                best_loss = epoch_loss
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
        
        return self

    def predict(self, X):
        # Convert DataFrame to NumPy array
        if isinstance(X, pd.DataFrame):
            X = X.values
        logging.info(f"PyTorchWrapper.predict: X shape={X.shape}")
        
        # Adjust input layer if necessary
        input_dim = X.shape[1]
        if self.input_dim != input_dim:
            self.input_dim = input_dim
            self.model.layer1 = nn.Linear(input_dim, self.model.layer1.out_features).to(self.device)
            logging.info(f"Adjusted SimpleNN input layer to {input_dim} features")

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
        return outputs.cpu().numpy().flatten()

    def get_params(self, deep=True):
        return {
            'model': self.model,
            'criterion': self.criterion,
            'optimizer': self.optimizer_class,
            'lr': self.lr,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'device': self.device
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

class ImportanceWrapper(BaseEstimator):
    def __init__(self, estimator, scorer, n_repeats=5, random_state=42):
        self.estimator = estimator
        self.scorer = scorer
        self.n_repeats = n_repeats
        self.random_state = random_state

    def fit(self, X, y, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        result = permutation_importance(
            self.estimator, X, y, scoring=self.scorer, n_repeats=self.n_repeats, random_state=self.random_state
        )
        self.feature_importances_ = result.importances_mean
        if hasattr(self.estimator, 'classes_'):
            self.classes_ = self.estimator.classes_
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def score(self, X, y):
        return self.estimator.score(X, y)

    def get_params(self, deep=True):
        params = {
            'scorer': self.scorer,
            'n_repeats': self.n_repeats,
            'random_state': self.random_state
        }
        if deep and hasattr(self.estimator, 'get_params'):
            params.update({'estimator__' + k: v for k, v in self.estimator.get_params(deep=deep).items()})
        return params

    def set_params(self, **params):
        for param, value in params.items():
            if param.startswith('estimator__'):
                self.estimator.set_params(**{param[11:]: value})
            else:
                setattr(self, param, value)
        return self

def MetaheuristicFeaturizer(
    dataframes, 
    top_n=5, 
    problem_type="classification",
    model=None,
    scorer=None,
    wrapper_class=None,
    wrapper_method="DISO",
    wrapper_population_size=20,
    wrapper_max_iter=50,
    **wrapper_kwargs
):
    """
    Selects feature combinations from multiple DataFrames by merging them horizontally.
    
    Parameters:
    - dataframes (list): List of pandas DataFrames, each with a 'target' column.
    - top_n (int): Number of top features to select for filter methods.
    - problem_type (str): 'classification' or 'regression'.
    - model: Custom model instance (scikit-learn or PyTorch nn.Module).
    - scorer: Custom scorer for cross_val_score.
    - wrapper_class: The Wrapper class for metaheuristic optimization.
    - wrapper_method (str): Method for the Wrapper optimizer (e.g., "DISO").
    - wrapper_population_size (int): Population size for the Wrapper optimizer.
    - wrapper_max_iter (int): Maximum iterations for the Wrapper optimizer.
    - **wrapper_kwargs: Additional keyword arguments for Wrapper initialization.
    
    Returns:
    - dict: Dictionary of DataFrames, each with selected features and 'target'.
    """
    # Validate inputs
    if not isinstance(dataframes, list) or not all(isinstance(df, pd.DataFrame) for df in dataframes):
        raise ValueError("dataframes must be a list of pandas DataFrames")
    if not dataframes:
        raise ValueError("At least one DataFrame must be provided")
    if problem_type not in ["classification", "regression"]:
        raise ValueError("problem_type must be 'classification' or 'regression'")
    if wrapper_class is None:
        raise ValueError("wrapper_class must be provided")
    if model is None:
        raise ValueError("model must be provided")
    if scorer is None:
        raise ValueError("scorer must be provided")
    
    # Check for target column in each DataFrame
    target_cols_all = []
    for i, df in enumerate(dataframes):
        target_cols = [col for col in df.columns if col.lower() == "target"]
        if not target_cols:
            raise ValueError(f"No 'target' column found in DataFrame {i}")
        target_cols_all.append(target_cols)
    
    # Merge DataFrames horizontally
    logging.info("Merging DataFrames...")
    try:
        # Verify target consistency
        for i, (df, target_cols) in enumerate(zip(dataframes[1:], target_cols_all[1:])):
            for col in target_cols:
                if not df[col].equals(dataframes[0][target_cols_all[0][0]]):
                    raise ValueError(f"Target column {col} in DataFrame {i+1} differs from DataFrame 0")
        
        # Horizontal concatenation
        dfs_to_concat = []
        seen_features = set()
        for i, (df, target_cols) in enumerate(zip(dataframes, target_cols_all)):
            df_copy = df.copy()
            df_copy = df_copy.rename(columns={target_cols[0]: "target"})
            if len(target_cols) > 1:
                df_copy = df_copy.drop(columns=target_cols[1:])
            # Drop overlapping features from subsequent DataFrames
            features = [col for col in df_copy.columns if col != 'target' and col not in seen_features]
            seen_features.update(features)
            dfs_to_concat.append(df_copy[features])
            logging.info(f"DataFrame {i} columns: {list(df_copy.columns)}")
            logging.info(f"DataFrame {i} NaNs: {df_copy.isna().sum().to_dict()}")
        
        merged_df = pd.concat(dfs_to_concat, axis=1)
        merged_df['target'] = dataframes[0]['target']
        logging.info(f"Merged DataFrame shape: {merged_df.shape}")
        logging.info(f"NaNs in merged_df: {merged_df.isna().sum().to_dict()}")
    except Exception as e:
        raise ValueError(f"Error merging DataFrames: {e}")
    
    if merged_df is None or merged_df.empty:
        raise ValueError("Merged DataFrame is empty")
    
    # Separate features and target
    X = merged_df.drop(columns=["target"])
    y = merged_df["target"]
    
    # Check for NaNs in X
    logging.info(f"NaNs in X before imputation: {X.isna().sum().sum()}")
    
    # Impute NaNs (unlikely with horizontal concatenation)
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    logging.info(f"NaNs in X after imputation: {X.isna().sum().sum()}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    logging.info(f"NaNs in X_scaled: {X_scaled.isna().sum().sum()}")
    
    # Check for zero variance features
    variances = X_scaled.var()
    zero_variance_cols = variances[variances == 0].index
    if len(zero_variance_cols) > 0:
        logging.warning(f"Zero variance features detected: {zero_variance_cols.tolist()}")
        X_scaled = X_scaled.drop(columns=zero_variance_cols)
        X = X.drop(columns=zero_variance_cols)
    
    # Initialize result dictionary
    selected_dfs = {}
    
    # Determine default model and scorer if not provided
    if model is None:
        model = LogisticRegression() if problem_type == "classification" else LinearRegression()
    if scorer is None:
        scorer = "accuracy" if problem_type == "classification" else "neg_mean_squared_error"
    
    # Wrap PyTorch model if provided
    logging.info(f"Model type: {type(model)}, Model class: {model.__class__.__name__}, Problem type: {problem_type}")
    if isinstance(model, nn.Module):
        criterion = nn.MSELoss() if problem_type == "regression" else nn.CrossEntropyLoss()
        model = PyTorchWrapper(model, criterion=criterion, optimizer=optim.Adam, lr=0.001, epochs=10, device='cpu')
        logging.info(f"Wrapped PyTorch model with criterion={criterion}")
    
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
        # Chi-Square Test
        selector = SelectKBest(score_func=chi2, k=min(top_n, X.shape[1]))
        selector.fit(X_scaled, y)
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
    logging.info(f"Before RFE: Model={model}, Type={type(model)}, Scorer={scorer}")
    # Clone and test fit to check if model will have importance attribute after fit
    try:
        estimator_clone = clone(model)
        small_X = X_scaled.iloc[:20].values  # Convert to NumPy
        small_y = y.iloc[:20].values if isinstance(y, pd.Series) else y[:20]
        estimator_clone.fit(small_X, small_y)
        has_importance = hasattr(estimator_clone, 'coef_') or hasattr(estimator_clone, 'feature_importances_')
        logging.info(f"Has importance attributes: {has_importance}")
    except Exception as e:
        logging.warning(f"Failed to test fit model {model.__class__.__name__}: {e}")
        has_importance = False
    
    try:
        if has_importance:
            logging.info(f"Using native importance for RFE with {model.__class__.__name__}")
            rfe = RFE(estimator=model, n_features_to_select=top_n)
        else:
            logging.info(f"Using permutation importance wrapper for RFE with {model.__class__.__name__}")
            try:
                wrapped_model = ImportanceWrapper(estimator=model, scorer=scorer, n_repeats=5, random_state=42)
                logging.info(f"Created ImportanceWrapper with estimator={model}, scorer={scorer}")
            except Exception as e:
                logging.error(f"Failed to create ImportanceWrapper: {e}")
                raise
            rfe = RFE(estimator=wrapped_model, n_features_to_select=top_n)
        
        rfe.fit(X_scaled.values, y.values if isinstance(y, pd.Series) else y)
        rfe_features = X.columns[rfe.support_].tolist()
        selected_dfs["RFE"] = pd.concat([X[rfe_features], y], axis=1)
    except Exception as e:
        logging.warning(f"RFE failed for {model.__class__.__name__}: {e}. Skipping RFE.")
    
    # 3. Embedded Methods
    logging.info("Applying embedded methods...")
    
    if problem_type == "classification":
        # Lasso-like (L1 penalty)
        lasso_model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
        lasso_model.fit(X_scaled, y)
        lasso_features = X.columns[lasso_model.coef_[0] != 0].tolist()
        if len(lasso_features) > top_n:
            lasso_features = lasso_features[:top_n]
        selected_dfs["Lasso"] = pd.concat([X[lasso_features], y], axis=1)
        
        # Ridge-like (L2 penalty)
        ridge_model = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000)
        ridge_model.fit(X_scaled, y)
        ridge_coef = pd.Series(np.abs(ridge_model.coef_[0]), index=X.columns)
        ridge_features = ridge_coef.nlargest(top_n).index.tolist()
        selected_dfs["Ridge"] = pd.concat([X[ridge_features], y], axis=1)
        
        # ElasticNet-like
        elastic_model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000)
        elastic_model.fit(X_scaled, y)
        elastic_features = X.columns[elastic_model.coef_[0] != 0].tolist()
        if len(elastic_features) > top_n:
            elastic_features = elastic_features[:top_n]
        selected_dfs["ElasticNet"] = pd.concat([X[elastic_features], y], axis=1)
    else:
        # Lasso
        lasso_model = Lasso(alpha=0.1)
        lasso_model.fit(X_scaled, y)
        lasso_features = X.columns[lasso_model.coef_ != 0].tolist()
        if len(lasso_features) > top_n:
            lasso_features = lasso_features[:top_n]
        selected_dfs["Lasso"] = pd.concat([X[lasso_features], y], axis=1)
        
        # Ridge
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_scaled, y)
        ridge_coef = pd.Series(np.abs(ridge_model.coef_), index=X.columns)
        ridge_features = ridge_coef.nlargest(top_n).index.tolist()
        selected_dfs["Ridge"] = pd.concat([X[ridge_features], y], axis=1)
        
        # ElasticNet
        elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
        elastic_model.fit(X_scaled, y)
        elastic_features = X.columns[elastic_model.coef_ != 0].tolist()
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
    
    def objective_function(solution, X, y, model, scorer):
        """Objective function for metaheuristic feature selection."""
        if hasattr(solution, "contents"):  # Handle ctypes pointer
            solution = np.ctypeslib.as_array(solution, shape=(X.shape[1],))
        
        if isinstance(solution, np.ndarray):
            solution = solution.tolist()
        
        # Convert solution to binary (select features where value > 0.5)
        selected = [i for i, val in enumerate(solution) if val > 0.5]
        if not selected:
            return float('inf') if "neg_" in str(scorer) else -float('inf')
        
        selected_features = X.columns[selected].tolist()
        X_subset = X[selected_features].values  # Convert to NumPy array
        logging.info(f"Objective function: X_subset shape={X_subset.shape}, selected features={selected_features}")
        
        try:
            scores = cross_val_score(model, X_subset, y, cv=2, scoring=scorer, error_score='raise')
            mean_score = scores.mean()
            return -mean_score if "neg_" not in str(scorer) else mean_score
        except Exception as e:
            logging.error(f"Error in metaheuristic optimization: {e}")
            return float('inf') if "neg_" not in str(scorer) else -float('inf')
    
    # Initialize optimizer
    optimizer = wrapper_class(
        dim=X.shape[1],
        bounds=[(0, 1)] * X.shape[1],
        population_size=wrapper_population_size,
        max_iter=wrapper_max_iter,
        method=wrapper_method,
        **wrapper_kwargs
    )
    
    try:
        import time
        start_time = time.time()
        optimizer.optimize(lambda sol: objective_function(sol, X, y, model, scorer))
        logging.info(f"Metaheuristic optimization took {time.time() - start_time:.2f} seconds")
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
    
    return selected_dfs
