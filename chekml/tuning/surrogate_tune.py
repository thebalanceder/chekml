import random
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import math
from sklearn.utils._param_validation import StrOptions
from sklearn.linear_model import Ridge

def evaluate_model(model, config, loss_fn, X_train, y_train, X_val, y_val):
    """Train the model with the given configuration and evaluate it."""
    model_copy = deepcopy(model)
    model_copy.set_params(**config)
    model_copy.fit(X_train, y_train)
    predictions = model_copy.predict(X_val)
    return loss_fn(y_val, predictions)

def evaluate_candidate(model, base_config, param_name, candidate, loss_fn, X_train, y_train, X_val, y_val):
    config_candidate = base_config.copy()
    config_candidate[param_name] = candidate
    loss = evaluate_model(model, config_candidate, loss_fn, X_train, y_train, X_val, y_val)
    return candidate, loss

def select_best(candidate_losses, current_best, best_loss):
    """
    Given a dictionary candidate_losses mapping candidate value -> loss,
    if multiple candidates have the same loss as best_loss, select the candidate
    with the smallest value.
    """
    candidates_equal = [cand for cand, loss in candidate_losses.items() if abs(loss - best_loss) < 1e-12]
    if candidates_equal:
        return min(candidates_equal)
    else:
        return current_best

def fine_tune_integer(model, param_name, init_best, loss_fn, X_train, y_train, X_val, y_val, base_config, 
                      max_iter=10, tol=1e-4):
    best_value = init_best
    current_config = base_config.copy()
    current_config[param_name] = best_value
    best_loss = evaluate_model(model, current_config, loss_fn, X_train, y_train, X_val, y_val)
    prev_loss = best_loss

    exp_factor = 2.0
    direction = 1
    
    if best_value == 0:
        low = 0
        high = 5
    else:
        low = best_value + 1
        high = best_value * 3

    for iteration in range(max_iter):
        if low > high:
            break

        candidates = list(range(low, high + 1))
        if not candidates:
            break

        candidate_losses = {}
        for candidate in candidates:
            config_candidate = base_config.copy()
            config_candidate[param_name] = candidate
            loss = evaluate_model(model, config_candidate, loss_fn, X_train, y_train, X_val, y_val)
            candidate_losses[candidate] = loss

        if not candidate_losses:
            break

        current_best = min(candidate_losses, key=candidate_losses.get)
        current_loss = candidate_losses[current_best]
        current_best = select_best(candidate_losses, current_best, current_loss)
        current_loss = candidate_losses[current_best]

        print(f"Iteration {iteration}: Range=({low}, {high}), Best candidate: {current_best} with loss {current_loss:.6f}")

        if (current_loss < best_loss - tol) or (abs(current_loss - best_loss) < tol and current_best < best_value):
            best_loss = current_loss
            best_value = current_best
        else:
            direction *= -1
            exp_factor = max(1.1, exp_factor / 2)

        if best_value == 0:
            low = 0
            high = 5
        else:
            if direction > 0:
                low = best_value + 1
                high = int(best_value + exp_factor * best_value)
            else:
                high = best_value - 1
                low = max(1, int(best_value - exp_factor * best_value))

        if abs(prev_loss - best_loss) < tol:
            break
        prev_loss = best_loss

    return best_value, best_loss
    
def fine_tune_float(model, param_name, init_best, loss_fn, X_train, y_train, X_val, y_val, base_config, 
                    max_iter=10, tol=1e-6, n_jobs=-1):
    best_value = init_best
    current_config = base_config.copy()
    current_config[param_name] = best_value
    best_loss = evaluate_model(model, current_config, loss_fn, X_train, y_train, X_val, y_val)
    prev_loss = best_loss
    
    print(f"Starting fine-tuning for float hyperparameter '{param_name}' with initial value {best_value:.4f} and loss {best_loss:.6f}")
    
    if best_value < 0.9:
        step = 0.0001
        for iteration in range(max_iter):
            candidates = [round(best_value + i * step, 4) for i in range(-10, 11) if i != 0]
            results = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_candidate)(model, base_config, param_name, candidate, loss_fn, X_train, y_train, X_val, y_val)
                for candidate in candidates
            )
            candidate_losses = {cand: loss for cand, loss in results}
            current_best = min(candidate_losses, key=candidate_losses.get)
            current_loss = candidate_losses[current_best]
            current_best = select_best(candidate_losses, current_best, current_loss)
            current_loss = candidate_losses[current_best]
            
            print(f"Iteration {iteration}: around {best_value:.4f}, best candidate: {current_best:.4f} with loss {current_loss:.6f}")
            
            if current_loss < best_loss - tol:
                best_loss = current_loss
                best_value = current_best
                step /= 2
            else:
                print("No significant improvement; converged in <0.9 regime.")
                break
                
            if abs(prev_loss - best_loss) < tol:
                break
            prev_loss = best_loss
    else:
        exp_factor = 2.0
        direction = 1
        low = best_value * 2
        high = best_value * 5
        
        for iteration in range(max_iter):
            candidates = [round(low + i*(high-low)/9, 0) for i in range(10)]
            results = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_candidate)(model, base_config, param_name, candidate, loss_fn, X_train, y_train, X_val, y_val)
                for candidate in candidates
            )
            candidate_losses = {cand: loss for cand, loss in results}
            current_best = min(candidate_losses, key=candidate_losses.get)
            current_loss = candidate_losses[current_best]
            current_best = select_best(candidate_losses, current_best, current_loss)
            current_loss = candidate_losses[current_best]
            
            print(f"Phase 1 Iteration {iteration}: Range=({low:.2f}, {high:.2f}), best candidate: {current_best:.2f} with loss {current_loss:.6f}")
            
            if current_loss < best_loss - tol:
                best_loss = current_loss
                best_value = current_best
            else:
                direction *= -1
                exp_factor = max(1.1, exp_factor/2)
                print("Phase 1: No improvement; reversing direction and reducing exp factor.")
            
            if direction > 0:
                low = best_value * 2
                high = best_value + exp_factor * best_value
            else:
                high = best_value / 2
                low = max(0.1, best_value - exp_factor * best_value)
            
            if abs(prev_loss - best_loss) < tol:
                print("Phase 1: Convergence reached at coarse search level.")
                break
            prev_loss = best_loss
        
        print(f"Starting Phase 2 zoom: best coarse candidate = {best_value:.2f}, loss = {best_loss:.6f}")
        step = 0.0001
        for iteration in range(max_iter):
            candidates = [round(best_value + i*step, 4) for i in range(-10, 11) if i != 0]
            results = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_candidate)(model, base_config, param_name, candidate, loss_fn, X_train, y_train, X_val, y_val)
                for candidate in candidates
            )
            candidate_losses = {cand: loss for cand, loss in results}
            current_best = min(candidate_losses, key=candidate_losses.get)
            current_loss = candidate_losses[current_best]
            current_best = select_best(candidate_losses, current_best, current_loss)
            current_loss = candidate_losses[current_best]
            
            print(f"Phase 2 Iteration {iteration}: around {best_value:.4f}, best candidate: {current_best:.4f} with loss {current_loss:.6f}")
            
            if (current_loss < best_loss - tol) or (abs(current_loss - best_loss) < tol and current_best < best_value):
                best_loss = current_loss
                best_value = current_best
                step /= 2
            else:
                print("Phase 2: No significant improvement; converged at fine resolution.")
                break
                
            if abs(prev_loss - best_loss) < tol:
                break
            prev_loss = best_loss

    return best_value, best_loss
    
def sample_configuration(schema):
    """Randomly sample a hyperparameter configuration from the schema."""
    config = {}
    for hp, values in schema.items():
        config[hp] = random.choice(values)
    # Enforce Ridge-specific constraints
    if isinstance(schema.get('solver'), list):
        # Set positive=True for lbfgs, positive=False for other solvers
        if config['solver'] == 'lbfgs':
            config['positive'] = True
        elif config['solver'] in ['sag', 'saga', 'lsqr', 'sparse_cg', 'svd', 'cholesky']:
            config['positive'] = False
    return config

def evaluate_model(model, config, loss_fn, X_train, y_train, X_val, y_val):
    """Train the model with a given configuration and evaluate it."""
    model_copy = deepcopy(model)
    model_copy.set_params(**config)
    model_copy.fit(X_train, y_train)
    predictions = model_copy.predict(X_val)
    loss = loss_fn(y_val, predictions)
    return loss

def collect_data(model, schema, loss_fn, X_train, y_train, X_val, y_val, iterations=30):
    """Collect hyperparameter configurations and their corresponding loss values."""
    data = []
    for i in range(iterations):
        config = sample_configuration(schema)
        loss = evaluate_model(model, config, loss_fn, X_train, y_train, X_val, y_val)
        data_point = {**config, 'loss': loss}
        data.append(data_point)
        print(f"Iteration {i+1}: config={config}, loss={loss}")
    return pd.DataFrame(data)

def train_surrogate(df, target='loss'):
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    # Ensure max_iter is treated as numeric, not categorical
    X = df.drop(columns=[target])
    y = df[target]
    # Convert max_iter to numeric, filling None with a large integer (e.g., 1000)
    if 'max_iter' in X.columns:
        X['max_iter'] = X['max_iter'].fillna(1000).astype(int)
    X_encoded = pd.get_dummies(X, columns=categorical_cols)
    # Ensure no NaNs in X_encoded
    X_encoded = X_encoded.fillna(0)
    surrogate = RandomForestRegressor(n_estimators=100)
    surrogate.fit(X_encoded, y)
    return surrogate, categorical_cols

def predict_with_surrogate(surrogate, config, categorical_cols):
    df_config = pd.DataFrame([config])
    # Handle max_iter in the config
    if 'max_iter' in df_config.columns:
        df_config['max_iter'] = df_config['max_iter'].fillna(1000).astype(int)
    df_config_encoded = pd.get_dummies(df_config, columns=categorical_cols)
    for col in surrogate.feature_names_in_:
        if col not in df_config_encoded.columns:
            df_config_encoded[col] = 0
    df_config_encoded = df_config_encoded[surrogate.feature_names_in_]
    return surrogate.predict(df_config_encoded)[0]

def optimize_hyperparameters(surrogate, schema, categorical_cols, num_candidates=1000):
    best_config = None
    best_pred = float('inf')
    for _ in range(num_candidates):
        config = sample_configuration(schema)
        pred_loss = predict_with_surrogate(surrogate, config, categorical_cols)
        if pred_loss < best_pred:
            best_pred = pred_loss
            best_config = config
    return best_config, best_pred

def extract_hyperparameter_schema(model):
    # Hardcode schema for Ridge to prioritize positive=False and remove None from max_iter
    if isinstance(model, Ridge):
        return {
            'alpha': [0, 0.5, 0.9, 1],
            'fit_intercept': [True, False],
            'copy_X': [True, False],
            'max_iter': [1, 2, 3, 4, 5, 1000],  # Replace None with 1000
            'tol': [0.0001, 0.001, 0.01],
            'solver': ['sag', 'saga', 'lsqr', 'lbfgs', 'cholesky', 'auto', 'sparse_cg', 'svd'],
            'positive': [False],  # Prioritize positive=False as requested
            'random_state': [42]
        }
    
    categorical_options = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'covariance_type': ['spherical', 'tied', 'diag', 'full'],
        'solver': ['sag', 'saga', 'lsqr', 'lbfgs', 'cholesky', 'auto', 'sparse_cg', 'svd']
    }
    
    schema = {}
    params = model.get_params()
    param_constraints = getattr(model, "_parameter_constraints", {})
    
    for param, value in params.items():
        lower_name = param.lower()
        if lower_name == 'n_jobs':
            schema[param] = [-1]
        elif isinstance(value, bool):
            schema[param] = [True, False]
        elif isinstance(value, str):
            if lower_name in categorical_options:
                schema[param] = categorical_options[lower_name]
            elif lower_name in param_constraints:
                options = param_constraints[lower_name]
                if isinstance(options, (StrOptions, set)):
                    schema[param] = list(options)
                elif isinstance(options, (list, tuple)):
                    schema[param] = list(options)
                else:
                    schema[param] = [value]
            else:
                schema[param] = [value]
        elif isinstance(value, int):
            schema[param] = [1, 2, 3, 4, 5]
        elif value is None:
            schema[param] = [1, 2, 3, 4, 5, 1000]  # Replace None with 1000
        elif isinstance(value, float):
            schema[param] = [0, 0.5, 0.9, 1]
        else:
            schema[param] = [value]
    
    return schema

def custom_hyperparameter_tuning(model, loss_fn, X_train, y_train, X_val, y_val, max_iter=10, surrogate_iters=30, num_candidates=1000):
    schema = extract_hyperparameter_schema(model)
    df_data = collect_data(model, schema, loss_fn, X_train, y_train, X_val, y_val, iterations=surrogate_iters)
    surrogate, categorical_cols = train_surrogate(df_data, target='loss')
    coarse_config, surrogate_loss = optimize_hyperparameters(surrogate, schema, categorical_cols, num_candidates=num_candidates)
    print("\nBest configuration predicted by the surrogate model:")
    print(coarse_config)
    print("Predicted loss:", surrogate_loss)
    
    overall_loss = evaluate_model(model, coarse_config, loss_fn, X_train, y_train, X_val, y_val)
    print("Actual loss with coarse configuration:", overall_loss)
    
    final_config = coarse_config.copy()
    for param, values in schema.items():
        initial_value = final_config[param]
        if isinstance(initial_value, bool):
            print(f"Skipping fine-tuning for boolean parameter '{param}' with value {initial_value}")
            continue
        elif isinstance(initial_value, int):
            print(f"Fine-tuning integer parameter '{param}' starting at {initial_value}")
            tuned_value, tuned_loss = fine_tune_integer(model, param, initial_value, loss_fn,
                                                        X_train, y_train, X_val, y_val, final_config, max_iter=max_iter)
            final_config[param] = tuned_value
            overall_loss = tuned_loss
        elif isinstance(initial_value, float):
            print(f"Fine-tuning float parameter '{param}' starting at {initial_value}")
            tuned_value, tuned_loss = fine_tune_float(model, param, initial_value, loss_fn,
                                                      X_train, y_train, X_val, y_val, final_config, max_iter=max_iter)
            final_config[param] = tuned_value
            overall_loss = tuned_loss
        else:
            print(f"Skipping fine-tuning for parameter '{param}' of type {type(initial_value)}")

    print("\nFinal tuned configuration:", final_config)
    print("Final loss:", overall_loss)
    return final_config, overall_loss

if __name__ == "__main__":
    from sklearn.linear_model import Ridge
    from sklearn.datasets import fetch_california_housing
    from sklearn.metrics import mean_squared_error

    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    def loss_fn(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    model = Ridge(random_state=42)
    final_config, final_loss = custom_hyperparameter_tuning(model, loss_fn, X_train, y_train, X_val, y_val,
                                                            max_iter=10, surrogate_iters=30, num_candidates=1000)