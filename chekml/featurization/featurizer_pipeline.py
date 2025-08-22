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
import torch
import torch.nn as nn
import torch.optim as optim
from chekml.featurization.IF.v2.inequality_based_featurization import InequalityFeaturizer
from chekml.featurization.IRF.v4.information_repurposed_featurization import InformationRepurposedFeaturizer
from chekml.featurization.MhF.MhF import MetaheuristicFeaturizer, PyTorchWrapper
from chekml.featurization.MetaheuristicOptimization.CIntegration_cffi1.wrapper import Wrapper

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
    def __init__(self, base_dir=None, timeout_seconds=60):
        """
        Initialize the FeaturizerPipeline.

        Parameters:
        - base_dir (str, optional): Base directory for dynamic file paths. Defaults to script directory.
        - timeout_seconds (int, default: 60): Timeout for InformationRepurposedFeaturizer to prevent hanging.
        """
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.timeout_seconds = timeout_seconds
        self.ineq_cython_file = os.path.join(self.base_dir, "IF", "v2", "ineq_cython.pyx")
        self.inequalities_c_file = os.path.join(self.base_dir, "IF", "v2", "inequalities.c")
        self.ineq_so_file = os.path.join(self.base_dir, "IF", "v2", "ineq_cython.so")
        self.report = []

    def run(self, df, top_n=5, criterion="mutual_information", output_csv="best_features.csv", output_report="pipeline_report.txt"):
        """
        Run the feature engineering pipeline.

        Parameters:
        - df (pandas.DataFrame): Input DataFrame with 'target' column.
        - top_n (int, default: 5): Number of top features to select and save.
        - criterion (str, default: "mutual_information"): Criterion for selecting best features ("mutual_information" or "cross_val_score").
        - output_csv (str, default: "best_features.csv"): Path to save top N features.
        - output_report (str, default: "pipeline_report.txt"): Path to save summary report.

        Returns:
        - dict: Dictionary containing output DataFrame, feature scores, and selected features.
        """
        self.report = []
        logging.info("Starting FeaturizerPipeline")
        self.report.append("# Featurizer Pipeline Report\n")

        # Validate input
        if 'target' not in df.columns:
            raise ValueError("DataFrame must contain 'target' column")
        if df.isna().any().any():
            raise ValueError("Input DataFrame contains NaN values")
        if criterion not in ["mutual_information", "cross_val_score"]:
            raise ValueError("criterion must be 'mutual_information' or 'cross_val_score'")

        # Step 1: InequalityFeaturizer
        logging.info("\n=== Running InequalityFeaturizer ===")
        self.report.append("## InequalityFeaturizer\n")
        if not os.path.exists(self.ineq_so_file):
            compile_cython_module(self.ineq_cython_file, self.inequalities_c_file, self.ineq_so_file)

        try:
            ineq_featurizer = InequalityFeaturizer()
            ineq_df = ineq_featurizer.featurize(
                df,
                level=2,
                stage=3,
                csv_path='ineq_features.csv',
                report_path='ineq_mi_report.txt'
            )
            self.report.append(f"- Output shape: {ineq_df.shape}\n")
            self.report.append(f"- New features: {[col for col in ineq_df.columns if col.startswith('f_')][:5]}\n")
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
                irf_df, metric_scores_df, feature_mi, trained_models = InformationRepurposedFeaturizer(
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
            self.report.append(f"- Output shape: {irf_df.shape}\n")
            self.report.append(f"- New features: {[col for col in irf_df.columns if not col.startswith('feature') and col != 'target'][:3]}\n")
            self.report.append(f"- Top MI scores: {sorted([(k, f'{v:.4f}') for k, v in feature_mi.items() if v > 0.1], key=lambda x: x[1], reverse=True)[:5]}\n")
        except TimeoutError:
            logging.error("InformationRepurposedFeaturizer timed out")
            irf_df = ineq_df
            feature_mi = {}
            self.report.append("- Failed: Timed out\n")
        except Exception as e:
            logging.error(f"InformationRepurposedFeaturizer failed: {e}")
            irf_df = ineq_df
            feature_mi = {}
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
                    wrapper_class=Wrapper,
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

        # Select best features based on criterion
        logging.info("\n=== Selecting Best Features ===")
        self.report.append("## Best Features Selection\n")
        try:
            final_df = list(selected_dfs["LinearRegression"].values())[0]  # Use LinearRegression's first method
            X_final = final_df.drop('target', axis=1)
            y_final = final_df['target']
            if criterion == "mutual_information":
                from sklearn.feature_selection import mutual_info_regression
                mi_scores = mutual_info_regression(X_final, y_final)
                feature_scores = dict(zip(X_final.columns, mi_scores))
                top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            else:  # cross_val_score
                feature_scores = {}
                for col in X_final.columns:
                    scores = cross_val_score(LinearRegression(), X_final[[col]], y_final, cv=5, scoring=custom_scorer)
                    feature_scores[col] = -scores.mean()  # Negative MSE to positive
                top_features = sorted(feature_scores.items(), key=lambda x: x[1])[:top_n]

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
            scores = cross_val_score(LinearRegression(), X_final, y_final, cv=5, scoring=custom_scorer)
            self.report.append(f"- 5-fold CV MSE: {-scores.mean():.4f} (+/- {2 * scores.std():.4f})\n")
        except Exception as e:
            logging.error(f"Final evaluation failed: {e}")
            self.report.append(f"- Failed: {e}\n")

        # Save report
        with open(output_report, 'w') as f:
            f.write("".join(self.report))

        return {
            "final_df": best_df,
            "feature_scores": feature_scores,
            "selected_features": [f[0] for f in top_features]
        }

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    torch.manual_seed(42)
    X, y = make_regression(n_samples=200, n_features=10, n_informative=5, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature{i+1}' for i in range(10)])
    df['target'] = y
    pipeline = FeaturizerPipeline()
    result = pipeline.run(df, top_n=5, criterion="mutual_information")
    logging.info(f"Pipeline completed. Best features: {result['selected_features']}")
