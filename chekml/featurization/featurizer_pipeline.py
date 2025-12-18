import pandas as pd
import numpy as np
import os
import subprocess
import sysconfig
import logging
import signal
from contextlib import contextmanager
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression
import torch
import torch.nn as nn
import torch.optim as optim
from chekml.featurization.IF.slow.inequality_based_featurization import InequalityFeaturizerSlow
from chekml.featurization.IRF.slow.information_repurposed_featurization import InformationRepurposedFeaturizerSlow
from chekml.featurization.MhF.MhF import MetaheuristicFeaturizer, PyTorchWrapper
from chekml.MetaheuristicOptimization.CIntegration_cffi1.wrapper import Wrapper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Timeout context manager for Unix-like systems
@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Operation timed out")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# PyTorch neural network for MetaheuristicFeaturizer
class SimpleNN(nn.Module):
    def __init__(self, input_dim=10, hidden_dim1=50, hidden_dim2=30, output_dim=1):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.output(x)
        return x

# Function to compile Cython module dynamically
def compile_cython_module(cython_file, c_file, output_so):
    try:
        python_include = sysconfig.get_path('include')
        numpy_include = np.get_include()
        logging.info(f"Compiling Cython module: {output_so}")
        subprocess.run([
            "cython", "-3", "--force", cython_file
        ], check=True)
        subprocess.run([
            "gcc", "-shared", "-pthread", "-fPIC", "-O3",
            "-o", output_so, cython_file.replace(".pyx", ".c"), c_file,
            "-lm", "-I", numpy_include, "-I", python_include,
            "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"
        ], check=True)
        logging.info(f"Successfully compiled {output_so}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Compilation failed: {e}")
        raise RuntimeError(f"Compilation failed: {e}")

class FeaturizerPipeline:
    def __init__(self, base_dir=None, wrapper_class=None, timeout_seconds=60):
        """
        Initialize the FeaturizerPipeline.

        Parameters:
        - base_dir (str, optional): Base directory for dynamic file paths. Defaults to script directory.
        - wrapper_class (class, optional): Wrapper class for MetaheuristicFeaturizer. Defaults to Wrapper from chekml.
        - timeout_seconds (int, default: 60): Timeout for InformationRepurposedFeaturizer to prevent hanging.
        """
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.wrapper_class = wrapper_class or Wrapper
        self.timeout_seconds = timeout_seconds
        self.ineq_cython_file = os.path.join(self.base_dir, "IF", "v2", "ineq_cython.pyx")
        self.inequalities_c_file = os.path.join(self.base_dir, "IF", "v2", "inequalities.c")
        self.ineq_so_file = os.path.join(self.base_dir, "IF", "v2", "ineq_cython.so")
        self.report = []
        self.all_features = {}  # Track all features and their scores

    def run(self, df, top_n=5, criterion="mutual_information", exclude_features=None, output_csv="best_features.csv", output_report="pipeline_report.txt"):
        """
        Run the feature engineering pipeline, selecting best features from all original and processed features.

        Parameters:
        - df (pandas.DataFrame): Input DataFrame with 'target' column.
        - top_n (int, default: 5): Number of top features to select and save.
        - criterion (str, default: "mutual_information"): Criterion for selecting best features ("mutual_information" or "cross_val_score").
        - exclude_features (list, optional): List of feature names to exclude from best features selection.
        - output_csv (str, default: "best_features.csv"): Path to save top N features.
        - output_report (str, default: "pipeline_report.txt"): Path to save summary report.

        Returns:
        - dict: Dictionary containing output DataFrame, feature scores, and selected features.
        """
        self.report = []
        self.all_features = {}
        exclude_features = exclude_features or []
        logging.info(f"Starting FeaturizerPipeline with wrapper: {self.wrapper_class.__name__}")
        self.report.append(f"# Featurizer Pipeline Report\n")
        self.report.append(f"- Wrapper class: {self.wrapper_class.__name__}\n")
        self.report.append(f"- Excluded features: {exclude_features}\n")

        # Validate input
        if 'target' not in df.columns:
            raise ValueError("DataFrame must contain 'target' column")
        if df.isna().any().any():
            raise ValueError("Input DataFrame contains NaN values")
        if criterion not in ["mutual_information", "cross_val_score"]:
            raise ValueError("criterion must be 'mutual_information' or 'cross_val_score'")
        invalid_excludes = [f for f in exclude_features if f not in df.columns and not f.startswith('f_')]
        if invalid_excludes:
            logging.warning(f"Some excluded features not in input DataFrame: {invalid_excludes}")

        # Initialize feature scores with original features
        original_features = [col for col in df.columns if col != 'target' and col not in exclude_features]
        y = df['target']
        if criterion == "mutual_information":
            mi_scores = mutual_info_regression(df[original_features], y)
            self.all_features.update(dict(zip(original_features, mi_scores)))
        else:  # cross_val_score
            scorer = make_scorer(mean_squared_error, greater_is_better=False)
            for col in original_features:
                scores = cross_val_score(LinearRegression(), df[[col]], y, cv=5, scoring=scorer)
                self.all_features[col] = -scores.mean()  # Convert to positive

        # Step 1: InequalityFeaturizer
        logging.info("\n=== Running InequalityFeaturizer ===")
        self.report.append("## InequalityFeaturizer\n")
        if not os.path.exists(self.ineq_so_file):
            compile_cython_module(self.ineq_cython_file, self.inequalities_c_file, self.ineq_so_file)

        try:
            ineq_featurizer = InequalityFeaturizerSlow()
            ineq_df = ineq_featurizer.featurize(
                df,
                level=2,
                stage=3,
                csv_path='ineq_features.csv',
                report_path='ineq_mi_report.txt'
            )
            # Update feature scores for new features
            new_features = [col for col in ineq_df.columns if col.startswith('f_') and col not in exclude_features]
            if new_features:
                if criterion == "mutual_information":
                    mi_scores = mutual_info_regression(ineq_df[new_features], y)
                    self.all_features.update(dict(zip(new_features, mi_scores)))
                else:
                    for col in new_features:
                        scores = cross_val_score(LinearRegression(), ineq_df[[col]], y, cv=5, scoring=scorer)
                        self.all_features[col] = -scores.mean()
            self.report.append(f"- Output shape: {ineq_df.shape}\n")
            self.report.append(f"- New features: {new_features[:5]}\n")
        except Exception as e:
            logging.error(f"InequalityFeaturizer failed: {e}")
            raise

        # Step 2: InformationRepurposedFeaturizer
        logging.info("\n=== Running InformationRepurposedFeaturizer ===")
        self.report.append("## InformationRepurposedFeaturizer\n")
        custom_models = [
            ("linear", LinearRegression()),
            ("decision_tree", DecisionTreeRegressor(random_state=42))
        ]
        custom_metrics = {
            "pearson": (lambda x, y: np.corrcoef(x, y)[0, 1], "maximize"),
            "spearman": (lambda x, y: np.corrcoef(np.argsort(x), np.argsort(y))[0, 1], "maximize")
        }
        custom_loss = {
            "identity": (lambda x: x, "maximize")
        }
        try:
            with timeout(self.timeout_seconds):
                irf_df, metric_scores_df, feature_mi, trained_models = InformationRepurposedFeaturizerSlow(
                    df=ineq_df,
                    models=custom_models,
                    loss_functions=custom_loss,
                    metrics=custom_metrics,
                    prediction_mode="top_n",
                    top_n=3,
                    score_key="spearman",
                    level=1,
                    save_models=False,
                    save_results_file="irf_results.txt",
                    n_jobs=1
                )
            # Update feature scores for new features
            new_features = [col for col in irf_df.columns if not col.startswith('feature') and col != 'target' and col not in exclude_features]
            if new_features:
                if criterion == "mutual_information":
                    mi_scores = mutual_info_regression(irf_df[new_features], y)
                    self.all_features.update(dict(zip(new_features, mi_scores)))
                else:
                    for col in new_features:
                        scores = cross_val_score(LinearRegression(), irf_df[[col]], y, cv=5, scoring=scorer)
                        self.all_features[col] = -scores.mean()
            self.report.append(f"- Output shape: {irf_df.shape}\n")
            self.report.append(f"- New features: {new_features[:3]}\n")
            self.report.append(f"- Top MI scores: {sorted([(k, f'{v:.4f}') for k, v in feature_mi.items() if v > 0.1], key=lambda x: x[1], reverse=True)[:5]}\n")
        except TimeoutError:
            logging.error("InformationRepurposedFeaturizer timed out")
            irf_df = ineq_df
            self.report.append("- Failed: Timed out\n")
        except Exception as e:
            logging.error(f"InformationRepurposedFeaturizer failed: {e}")
            irf_df = ineq_df
            self.report.append(f"- Failed: {e}\n")

        # Step 3: MetaheuristicFeaturizer
        logging.info("\n=== Running MetaheuristicFeaturizer ===")
        self.report.append("## MetaheuristicFeaturizer\n")
        custom_scorer = make_scorer(mean_squared_error, greater_is_better=False)
        pytorch_model = PyTorchWrapper(
            model=SimpleNN(input_dim=irf_df.shape[1]-1),
            criterion=nn.MSELoss(),
            optimizer=optim.Adam,
            lr=0.001,
            epochs=10,
            batch_size=32,
            device='cpu'
        )
        models = {
            "LinearRegression": LinearRegression(),
            "PyTorchNN": pytorch_model
        }
        selected_dfs = {}
        for model_name, model in models.items():
            logging.info(f"Running MetaheuristicFeaturizer with {model_name}...")
            self.report.append(f"### {model_name}\n")
            try:
                result = MetaheuristicFeaturizer(
                    dataframes=[irf_df],
                    top_n=top_n,
                    problem_type="regression",
                    model=model,
                    scorer=custom_scorer,
                    wrapper_class=self.wrapper_class,
                    wrapper_method="DISO",
                    wrapper_population_size=10,
                    wrapper_max_iter=10
                )
                selected_dfs[model_name] = result
                for method, df in result.items():
                    self.report.append(f"- {method} shape: {df.shape}\n")
                    self.report.append(f"- Features: {list(df.columns[:-1])}\n")
            except Exception as e:
                logging.error(f"MetaheuristicFeaturizer with {model_name} failed: {e}")
                self.report.append(f"- Failed: {e}\n")

        # Select best features from all features
        logging.info("\n=== Selecting Best Features ===")
        self.report.append("## Best Features Selection\n")
        try:
            # Use the most comprehensive DataFrame (irf_df) for all features
            X_final = irf_df.drop('target', axis=1)
            y_final = irf_df['target']
            # Exclude specified features
            valid_features = [(f, s) for f, s in self.all_features.items() if f not in exclude_features]
            top_features = sorted(valid_features, key=lambda x: x[1], reverse=(criterion == "mutual_information"))[:top_n]
            
            # Save best features to CSV
            best_df = pd.concat([X_final[[f[0] for f in top_features]], y_final], axis=1)
            best_df.to_csv(output_csv, index=False)
            self.report.append(f"- Criterion: {criterion}\n")
            self.report.append(f"- Top {top_n} features: {[f[0] for f in top_features]}\n")
            self.report.append(f"- Scores: {[f'{f[1]:.4f}' for f in top_features]}\n")
            self.report.append(f"- Saved to: {output_csv}\n")
        except Exception as e:
            logging.error(f"Best features selection failed: {e}")
            self.report.append(f"- Failed: {e}\n")

        # Evaluate final features
        logging.info("\n=== Evaluating Final Features ===")
        self.report.append("## Final Evaluation\n")
        try:
            scores = cross_val_score(LinearRegression(), X_final[[f[0] for f in top_features]], y_final, cv=5, scoring=custom_scorer)
            self.report.append(f"- 5-fold CV MSE: {-scores.mean():.4f} (+/- {2 * scores.std():.4f})\n")
        except Exception as e:
            logging.error(f"Final evaluation failed: {e}")
            self.report.append(f"- Failed: {e}\n")

        # Save report
        with open(output_report, 'w') as f:
            f.write("".join(self.report))

        return {
            "final_df": best_df,
            "feature_scores": self.all_features,
            "selected_features": [f[0] for f in top_features]
        }

if __name__ == "__main__":
    # Example usage with custom wrapper
    np.random.seed(42)
    torch.manual_seed(42)
    from sklearn.datasets import make_regression

    # Custom wrapper example
    class CustomWrapper:
        def __init__(self, dim, bounds, population_size, max_iter, method="DISO"):
            self.dim = dim
            self.bounds = bounds
            self.population_size = population_size
            self.max_iter = max_iter
            self.method = method
            self.best_solution = np.random.uniform(0, 1, dim)
            self.best_score = float('inf')

        def optimize(self, objective_function):
            for _ in range(self.max_iter):
                solution = np.random.uniform(0, 1, self.dim)
                score = objective_function(solution)
                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = solution

        def get_best_solution(self):
            return self.best_solution, self.best_score

        def free(self):
            pass

    X, y = make_regression(n_samples=200, n_features=10, n_informative=5, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature{i+1}' for i in range(10)])
    df['target'] = y
    pipeline = FeaturizerPipeline(wrapper_class=CustomWrapper)
    result = pipeline.run(
        df,
        top_n=5,
        criterion="mutual_information",
        exclude_features=['feature1', 'feature2'],
        output_csv="best_features.csv",
        output_report="pipeline_report.txt"
    )
    logging.info(f"Pipeline completed. Best features: {result['selected_features']}")
