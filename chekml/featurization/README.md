## InequalityFeaturizer

## All Available Parameters

- Method: `add_inequality(name, source_code)`:
  - `name`: String, the name of the user-defined inequality.
  - `source_code`: String, the Python source code for the inequality (stored for reference, requires manual C implementation).

- Method: `delete_inequality(name)`:
  - `name`: String, the name of the user-defined inequality to remove.

- Method: `delete_all_inequalities()`:
  - No parameters.
  - Removes all user-defined inequalities.

- Method: `print_inequalities()`:
  - No parameters.
  - Prints all default and user-defined inequalities.

- Method: `featurize(df, level=1, stage=1, csv_path=None, report_path=None)`:
  - `df`: Pandas DataFrame, the input data containing features and a target column.
  - `level`: Integer, the maximum number of features to combine when generating inequality-based features (default: 1).
  - `stage`: Integer, the number of top inequalities to select for each feature combination based on their average absolute value (default: 1).
  - `csv_path`: String or None, path to save the resulting DataFrame as a CSV file (must end with .csv, default: None).
  - `report_path`: String or None, path to save the mutual information scores report as a text file (must end with .txt, default: None).

### Algorithm of InequalityFeaturizer
- The InequalityFeaturizer is designed to create new features for a dataset by applying mathematical inequalities to combinations of input features, implemented efficiently using C and Cython for performance. Here’s how the algorithm works:

### Initialization:
- The class initializes with a predefined list of inequalities (e.g., arithmetic mean (am), geometric mean (gm), harmonic mean (hm), etc.) defined in inequalities.c.
- It sets up paths for the C implementation (inequalities.c) and Cython interface (ineq_cython.pyx).

### Adding/Deleting Inequalities:
- Users can add custom inequalities via `add_inequality`, but the Python source code is only stored for reference. The actual implementation must be manually added to `inequalities.c` and `inequalities.h`, followed by recompilation.
- `delete_inequality` and `delete_all_inequalities` remove user defined inequalities from the internal dictionary, but manual removal from C files is required.

### Featurization Process (featurize method):
- Input Validation: Ensures the input DataFrame has no NaN values and contains a target column.
- Data Preparation: Extracts features (excluding target) and converts them to a NumPy array for C compatibility.

### Feature Combination:
- Generates combinations of features up to the specified level (e.g., level=2 means pairs of features).
- For each combination, applies all defined inequalities (from inequalities[] in inequalities.c).

### Inequality Computation:
- Each inequality function (e.g., am, gm) takes a subset of features and computes a value for each row.
- Functions handle edge cases (e.g., using `MIN_VALUE` to avoid division by zero).
- Special cases for two inputs are implemented for some inequalities (e.g., log_mean, seiffert).

### Feature Selection:
- Computes the average absolute value of each inequality’s output across rows.
- Sorts inequalities by this value and selects the top stage inequalities for each combination.

### Output:
- Creates new features in the DataFrame with names like `f_<feature_indices>_<inequality_name>` (e.g., `f_0_1_am` for arithmetic mean of features 0 and 1).
- Computes mutual information scores between each feature (original and new) and the target using `mutual_info_regression`.
- Optionally saves the new DataFrame to `csv_path` and mutual information scores to `report_path`.

### C/Cython Integration:
- The compute_features function in inequalities.c handles the core computation, called via the Cython wrapper compute_features_cython.
- The Cython module optimizes memory management and data transfer between Python and C.

### Compilation:
- The build.sh script compiles the Cython file (ineq_cython.pyx) to C and then compiles the C code with inequalities.c into a shared library (ineq_cython.so).
- This requires cython, gcc, NumPy, and Python development headers.

### Mutual Information:
-After generating new features, the algorithm computes mutual information scores to quantify the relevance of each feature to the target, aiding in feature selection or analysis.

### Usage
- The InequalityFeaturizer class allows you to:
  - Apply default inequalities (e.g., arithmetic mean, geometric mean, etc.) to feature combinations.
  - Add or delete user-defined inequalities (requires manual C implementation).
  - Generate new features and compute mutual information scores.

### Example
```python
pythonimport pandas as pd
import numpy as np
from inequality_based_featurization import InequalityFeaturizer

# Create sample data
np.random.seed(42)
data = pd.DataFrame({
    'A': np.abs(np.random.randn(100)),
    'B': np.abs(np.random.randn(100)),
    'C': np.abs(np.random.randn(100))
})
data['target'] = 0.5 * data['A'] + 0.5 * data['C'] + np.random.randn(100) * 0.1

# Initialize featurizer
featurizer = InequalityFeaturizer()

# Add a custom inequality (requires manual C implementation)
custom_ineq = """
def custom_ineq(x):
    return np.max(x) - np.min(x)
"""
featurizer.add_inequality("custom_ineq", custom_ineq)

# Print all inequalities
featurizer.print_inequalities()

# Delete the custom inequality
featurizer.delete_inequality("custom_ineq")

# Featurize the data
result_df = featurizer.featurize(
    df=data,
    level=2,  # Combine up to pairs of features
    stage=3,  # Select top 3 inequalities per combination
    csv_path='output_features.csv',  # Save new DataFrame
    report_path='mi_report.txt'      # Save mutual information scores
)

# View results
print("\nFirst few rows of resulting DataFrame:")
print(result_df.head())
```
### Parameters

featurize(df, level=1, stage=1, csv_path=None, report_path=None):

df: Input DataFrame with features and a target column.
level: Maximum number of features to combine (e.g., 1 for single features, 2 for pairs).
stage: Number of top inequalities to select per combination.
csv_path: Path to save the resulting DataFrame (CSV).
report_path: Path to save mutual information scores (TXT).



Adding Custom Inequalities
To add a custom inequality:

Define the Python source code for reference.
Manually implement the inequality in inequalities.c as a C function.
Update the inequalities[] array in inequalities.c and inequalities.h.
Recompile using ./build.sh.

Example C implementation for custom_ineq:
cdouble custom_ineq(double* x, int n) {
    double min_x = x[0], max_x = x[0];
    for (int i = 1; i < n; i++) {
        min_x = fmin(min_x, x[i]);
        max_x = fmax(max_x, x[i]);
    }
    return max_x - min_x;
}
Update inequalities[] in inequalities.c:
cInequality inequalities[] = {
    // Existing inequalities...
    {"custom_ineq", custom_ineq}
};
Output

New Features: Added to the DataFrame with names like f_<feature_indices>_<inequality_name> (e.g., f_0_1_am).
Mutual Information Scores: Printed and optionally saved to report_path.
CSV Output: New DataFrame saved to csv_path if specified.

Available Parameters for InformationRepurposedFeaturizer
Based on the provided code, the InformationRepurposedFeaturizer function accepts the following parameters:

Function: InformationRepurposedFeaturizer:

df: Pandas DataFrame, the input data containing features and a target column.
models: List of tuples, each containing a model name (string) and a scikit-learn-compatible model instance (e.g., [("linear", LinearRegression()), ("rf", RandomForestRegressor())]). Default includes LinearRegression, RandomForestRegressor, GradientBoostingRegressor, and XGBRegressor.
loss_functions: Dictionary of loss functions, where each key is a metric name and the value is a tuple of a transformation function and its optimization direction ("maximize" or "minimize") (e.g., {"pearson": (lambda x: x, "maximize")}).
metrics: Dictionary of evaluation metrics, where each key is a metric name and the value is a tuple of a metric function and its optimization direction (e.g., {"pearson": (lambda x, y: pearsonr(x, y)[0], "maximize")}).
prediction_mode: String, one of "all", "top_n", or "specific_metrics". Determines which predictions are included in the output:

"all": Include all generated predictions.
"top_n": Include the top top_n predictions based on score_key.
"specific_metrics": Include predictions for specified metrics in specific_metrics.


top_n: Integer, number of top predictions to select when prediction_mode="top_n". Default is 10.
score_key: String, the metric used to rank predictions when prediction_mode="top_n". Default is "mutual_information".
specific_metrics: List of strings, metrics to include when prediction_mode="specific_metrics". Default is None.
level: Integer, maximum number of features to combine (e.g., 1 for single features, 2 for pairs). Default is 1.
save_models: Boolean, whether to save trained models to disk using cloudpickle. Default is False.
save_results_file: String or None, path to save the results (mutual information scores, DataFrame preview, etc.) as a text file. Default is None.
n_jobs: Integer, number of parallel jobs for computation. Default is 1.


Helper Functions:

custom_digitize(data, bins):

data: NumPy array, data to bin.
bins: Integer, number of bins.


compute_mic(x, y, max_bins=10):

x, y: NumPy arrays, input data for computing Maximal Information Coefficient (MIC).
max_bins: Integer, maximum number of bins for MIC computation (default: 10).


compute_copula_measure(x, y), compute_distance_correlation(x, y), compute_hsic(x, y), compute_energy_distance_correlation(x, y):

x, y: NumPy arrays, input data for respective metric computations.




Default Metrics (from default_metrics):

Includes metrics like pearson, spearman, kendall, mutual_information, normalized_mutual_information, maximal_information_coefficient, distance_correlation, hsic, copula, gini_correlation, fisher_score, relief_score, anova_f_statistic, chi_square_score, partial_correlation, information_gain, gain_ratio.
Each metric is a tuple of a function and its optimization direction ("maximize" or "minimize").



Algorithm of InformationRepurposedFeaturizer
The InformationRepurposedFeaturizer generates new features by training machine learning models on combinations of input features, evaluating their predictions using various statistical metrics, and selecting the most informative features. It leverages OpenCL-accelerated metrics (via metrics.cpp) for performance. Here’s how the algorithm works:

Initialization:

Validates the input DataFrame, ensuring it has a target column and no NaN values (handled by SimpleImputer if present).
Sets up default or user-provided models, loss functions, and metrics.
Initializes StandardScaler for feature scaling and SimpleImputer for handling missing values.


Feature Combination:

Generates all possible combinations of features up to the specified level (e.g., level=2 means pairs of features).
For each combination, prepares the feature subset as input for model training.


Model Training and Prediction:

For each feature combination, trains each model with each loss function.
Applies the loss function to the target variable before training to transform it as needed.
Generates predictions for each model and feature combination.


Metric Evaluation:

Computes mutual information between each original feature and the target using mutual_info_regression.
Evaluates predictions using the specified metrics (e.g., Pearson correlation, MIC, HSIC) against the target.
Metrics like compute_mic, compute_distance_correlation, etc., are computed using OpenCL kernels in metrics.cpp for efficiency, with CPU fallback if OpenCL fails.
Stores metric scores and predictions for later selection.


Parallel Processing:

Uses joblib.Parallel to parallelize model training and metric computation across n_jobs processes, improving performance for large datasets or many combinations.


Feature Selection:

Based on prediction_mode:

"all": Includes all generated predictions in the output DataFrame.
"top_n": Selects the top top_n predictions based on the score_key metric (e.g., highest mutual information).
"specific_metrics": Includes predictions for the specified metrics in specific_metrics.


Adds selected predictions as new columns in the output DataFrame with names like <feature_combo>_<model>_<metric>_<loss_function>.


Model Saving (Optional):

If save_models=True, saves trained models to disk using cloudpickle in the saved_models directory.


Output:

Returns:

result_df: DataFrame with original features, target, and selected prediction-based features.
metric_scores_df: DataFrame with scores for each metric across predictions.
feature_mi: Dictionary of mutual information scores for original features.
trained_models: Dictionary of trained models (if save_models=True).


Optionally saves results (mutual information, DataFrame preview, metric scores) to save_results_file.


OpenCL Integration:

The metrics.cpp file provides OpenCL-accelerated implementations for complex metrics (e.g., MIC, distance correlation, HSIC) using kernels for distance matrix computation, matrix centering, Gaussian kernels, and reduction sums.
Initializes an OpenCL context, preferring Intel platforms and GPU devices, and falls back to CPU if needed.

Usage

The InformationRepurposedFeaturizer generates new features by:





Training models on feature combinations.



Evaluating predictions with metrics like Pearson correlation, MIC, HSIC, etc.



Selecting the most informative predictions based on the specified mode.

Example

import pandas as pd
import numpy as np
from information_repurposed_featurization import InformationRepurposedFeaturizer
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from scipy.stats import pearsonr, spearmanr

# Create sample data
np.random.seed(42)
data = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'feature3': np.random.randn(100),
    'target': np.random.randn(100) + np.random.randn(100) * 0.5
})

# Define custom models
custom_models = [
    ('decision_tree', DecisionTreeRegressor(random_state=42)),
    ('xg_boost', XGBRegressor(random_state=42, n_estimators=50))
]

# Define custom loss functions
custom_loss = {
    'pearson': (lambda x: x, 'maximize'),
    'spearman': (lambda x: np.argsort(x), 'maximize'),
    'custom_metric': (lambda x: np.log1p(np.abs(x)), 'maximize')
}

# Define custom metrics
custom_metrics = {
    'pearson': (lambda x, y: pearsonr(x, y)[0], 'maximize'),
    'spearman': (lambda x, y: spearmanr(x, y)[0], 'maximize'),
    'custom_metric': (lambda x, y: np.mean((x - y) ** 2), 'minimize')
}

# Run featurizer
result_df, metric_scores_df, feature_mi, trained_models = InformationRepurposedFeaturizer(
    df=data,
    models=custom_models,
    loss_functions=custom_loss,
    metrics=custom_metrics,
    prediction_mode='top_n',
    top_n=3,
    score_key='spearman',
    level=2,
    save_models=True,
    save_results_file='model_results.txt',
    n_jobs=4
)

# Print results
print("Feature Mutual Information with Target:")
for feature, mi in feature_mi.items():
    print(f"{feature}: {mi:.4f}")
print("\nPredictions DataFrame:")
print(result_df.head())
print("\nMetric Scores DataFrame:")
print(metric_scores_df)
print("\nNumber of trained models:", len(trained_models))

Available Parameters for MetaheuristicFeaturizer
Based on the provided code in MhF.py and runMhF.py, the MetaheuristicFeaturizer function accepts the following parameters:

Function: MetaheuristicFeaturizer:

dataframes: List of Pandas DataFrames, each containing features and a target column. The DataFrames may have overlapping or distinct features but must share the same target.
top_n: Integer, the number of top features to select for each feature selection method (default: 10).
problem_type: String, either "classification" or "regression", specifying the type of machine learning problem.
model: A scikit-learn-compatible model instance or a PyTorch model wrapped in PyTorchWrapper. Examples include LinearRegression, DecisionTreeRegressor, MLPRegressor, or a custom PyTorch model like SimpleNN.
scorer: A scikit-learn scorer object (e.g., created with make_scorer), used to evaluate model performance during feature selection. For regression, it could be mean_squared_error with greater_is_better=False; for classification, it could be accuracy_score.
wrapper_class: A class (e.g., Wrapper from chekml.MetaheuristicOptimization.CIntegration_cffi1.wrapper), providing the metaheuristic optimization method (e.g., Differential Evolution, Particle Swarm Optimization).
wrapper_method: String, the specific metaheuristic method to use (e.g., "DISO" for Differential Evolution-based optimization).
wrapper_population_size: Integer, the population size for the metaheuristic algorithm.
wrapper_max_iter: Integer, the maximum number of iterations for the metaheuristic algorithm.
wrapper_kwargs: Dictionary, additional keyword arguments for the wrapper_class initialization (default: {}).


Helper Class: PyTorchWrapper:

model: A PyTorch model (e.g., SimpleNN) implementing the nn.Module interface.
criterion: A PyTorch loss function (e.g., nn.MSELoss() for regression).
optimizer: A PyTorch optimizer class (e.g., optim.Adam, default: optim.Adam).
lr: Float, learning rate for the optimizer (default: 0.001).
epochs: Integer, number of training epochs (default: 10).
batch_size: Integer, batch size for training (default: 32).
device: String, device for PyTorch computations (e.g., "cpu" or "cuda", default: "cpu").



Algorithm of MetaheuristicFeaturizer
The MetaheuristicFeaturizer is designed to select the most informative features from multiple DataFrames using a combination of traditional feature selection methods and a metaheuristic optimization approach. It supports both regression and classification tasks and integrates PyTorch models for advanced neural network-based feature selection. Here’s how the algorithm works:

Initialization:

Validates that all input DataFrames contain a target column and handles missing values using SimpleImputer (mean strategy).
Concatenates all DataFrames, keeping unique features and the shared target column.
Applies StandardScaler to standardize features for methods requiring scaled inputs (e.g., Lasso, Ridge).


Feature Selection Methods:

Variance Threshold: Removes features with variance below a threshold (default: 0.0) to eliminate low-variance features.
SelectKBest:

For classification: Uses chi2, f_classif, or mutual_info_classif to select the top top_n features based on statistical tests.
For regression: Uses f_regression or mutual_info_regression to select the top top_n features.


Recursive Feature Elimination (RFE):

Uses a base estimator (e.g., LogisticRegression for classification, LinearRegression for regression) to iteratively eliminate features, selecting the top top_n.


Permutation Importance:

Fits a model (e.g., RandomForest) and computes feature importance based on the impact of permuting each feature on model performance, selecting the top top_n.


Lasso:

For classification: Uses LogisticRegression with l1 penalty.
For regression: Uses Lasso with alpha=0.1, selecting features with non-zero coefficients (up to top_n).


Ridge:

Uses Ridge (regression) or LogisticRegression with l2 penalty (classification), selecting the top top_n features based on coefficient magnitudes.


ElasticNet:

Uses ElasticNet (regression) or LogisticRegression with elasticnet penalty (classification), selecting features with non-zero coefficients (up to top_n).


RandomForest:

Fits a RandomForestClassifier or RandomForestRegressor and selects the top top_n features based on feature importances.


XGBoost:

Fits an XGBClassifier or XGBRegressor and selects the top top_n features based on feature importances.




Metaheuristic Feature Selection:

Uses the provided wrapper_class (e.g., Wrapper) and wrapper_method (e.g., "DISO") to perform feature selection via metaheuristic optimization.
Defines an objective function that:

Converts the metaheuristic solution (a vector of values between 0 and 1) to a binary mask (features with values > 0.5 are selected).
Evaluates the selected feature subset using cross-validation (cross_val_score) with the provided model and scorer.
Returns a score to minimize (negative for maximization scorers, positive for minimization scorers like neg_mean_squared_error).


Runs the metaheuristic optimizer with wrapper_population_size and wrapper_max_iter, selecting the top top_n features from the best solution.
Frees the optimizer’s resources after completion.


Output:

Returns a dictionary (selected_dfs) where each key is a method name (e.g., "VarianceThreshold", "Metaheuristic") and the value is a DataFrame containing the selected features and the target column.
Each DataFrame includes up to top_n features selected by the respective method.


PyTorch Integration:

For PyTorch models (e.g., SimpleNN), the PyTorchWrapper class adapts the model to the scikit-learn interface.
Dynamically adjusts the input layer to match the number of features.
Trains the model using the specified criterion, optimizer, lr, epochs, and batch_size, with early stopping based on validation loss.
Supports both CPU and GPU (cuda) computations.

Usage
The MetaheuristicFeaturizer selects the most informative features from multiple DataFrames using methods like Variance Threshold, SelectKBest, RFE, Permutation Importance, Lasso, Ridge, ElasticNet, RandomForest, XGBoost, and a metaheuristic approach (e.g., Differential Evolution via Wrapper).
Example
pythonimport pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from chekml.MetaheuristicOptimization.CIntegration_cffi1.wrapper import Wrapper
from MhF import MetaheuristicFeaturizer
import torch
import torch.nn as nn

# Define a PyTorch neural network
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
        x = self.relu2(self.layer Primative(self.layer2(x)))
        x = self.output(x)
        return x

# Generate sample data
X, y = make_regression(n_samples=200, n_features=10, n_informative=5, noise=0.1, random_state=42)
df1 = pd.DataFrame(X[:, :6], columns=[f'feature{i+1}' for i in range(6)])
df1['target'] = y
df2 = pd.DataFrame(X[:, 4:], columns=[f'feature{i+5}' for i in range(6)])
df2['target'] = y
dataframes = [df1, df2]

# Define custom scorer
custom_scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Define models
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=5, random_state=42),
    "MLPRegressor": MLPRegressor(
        hidden_layer_sizes=(50, 30), activation='relu', solver='adam',
        alpha=0.001, max_iter=200, tol=1e-3, early_stopping=True,
        validation_fraction=0.1, random_state=42
    ),
    "SimpleNN": SimpleNN(input_dim=10)
}

# Run MetaheuristicFeaturizer
for model_name, model in models.items():
    print(f"\nRunning MetaheuristicFeaturizer with {model_name}...\n")
    selected_dfs = MetaheuristicFeaturizer(
        dataframes=dataframes,
        top_n=3,
        problem_type="regression",
        model=model,
        scorer=custom_scorer,
        wrapper_class=Wrapper,
        wrapper_method="DISO",
        wrapper_population_size=10,
        wrapper_max_iter=20
    )
    
    # Print results
    for method, df in selected_dfs.items():
        print(f"{model_name} - {method} selected features (shape: {df.shape}):\n{df.head()}\n")
Parameters

MetaheuristicFeaturizer(dataframes, top_n=10, problem_type="regression", model=None, scorer=None, wrapper_class=None, wrapper_method="DISO", wrapper_population_size=10, wrapper_max_iter=20, wrapper_kwargs={}):

dataframes: List of DataFrames with features and a shared target column.
top_n: Number of top features to select per method.
problem_type: "classification" or "regression".
model: Scikit-learn model or PyTorch model (wrapped in PyTorchWrapper).
scorer: Scikit-learn scorer (e.g., make_scorer(mean_squared_error, greater_is_better=False)).
wrapper_class: Metaheuristic wrapper class (e.g., Wrapper).
wrapper_method: Metaheuristic method (e.g., "DISO").
wrapper_population_size: Population size for metaheuristic optimization.
wrapper_max_iter: Maximum iterations for metaheuristic optimization.
wrapper_kwargs: Additional arguments for the wrapper class.


PyTorchWrapper(model, criterion=None, optimizer=optim.Adam, lr=0.001, epochs=10, batch_size=32, device='cpu'):

model: PyTorch nn.Module (e.g., SimpleNN).
criterion: PyTorch loss function (e.g., nn.MSELoss()).
optimizer: PyTorch optimizer class (e.g., optim.Adam).
lr: Learning rate.
epochs: Number of training epochs.
batch_size: Batch size for training.
device: "cpu" or "cuda".



Output

selected_dfs: Dictionary where keys are feature selection method names (e.g., "VarianceThreshold", "Metaheuristic") and values are DataFrames containing the selected features and target column.

Notes

Ensure all DataFrames have a target column with consistent values.
Missing values are handled by SimpleImputer (mean strategy).
PyTorch models require a compatible criterion (e.g., nn.MSELoss for regression).
The Wrapper class must be properly configured and compiled (refer to its documentation).
Metaheuristic optimization can be computationally intensive; adjust wrapper_population_size and wrapper_max_iter based on available resources.

Available Parameters for FeaturizerPipeline
Based on the provided code in featurizer_pipeline.py and example.py, the FeaturizerPipeline class and its methods accept the following parameters:

Class Initialization (__init__):

base_dir: String or None, the base directory for dynamic file paths (e.g., for Cython and C files). Defaults to the directory of the script.
wrapper_class: Class or None, the wrapper class for metaheuristic optimization (e.g., Wrapper from chekml.MetaheuristicOptimization.CIntegration_cffi1 or a custom class like CustomWrapper). Defaults to Wrapper.
timeout_seconds: Integer, the timeout duration (in seconds) for InformationRepurposedFeaturizer to prevent hanging (default: 60).


Method: run(df, top_n, criterion, exclude_features, output_csv, output_report):

df: Pandas DataFrame, the input data containing features and a target column.
top_n: Integer, the number of top features to select based on the specified criterion (default: 5).
criterion: String, the metric used to rank features for final selection. Options are "mutual_information" (default) or "cross_val_score".
exclude_features: List of strings or None, feature names to exclude from the final selection (default: None).
output_csv: String, the file path to save the final DataFrame with selected features and the target (default: "best_features.csv").
output_report: String, the file path to save the pipeline report (default: "pipeline_report.txt").


Helper Class: CustomWrapper (Optional, as shown in example.py):

dim: Integer, the dimensionality of the solution space (number of features).
bounds: List of tuples, the bounds for each dimension (e.g., [(0, 1)] * dim).
population_size: Integer, the population size for the metaheuristic algorithm.
max_iter: Integer, the maximum number of iterations for the metaheuristic algorithm.
method: String, the metaheuristic method to use (e.g., "DISO").


Helper Class: SimpleNN (Used in MetaheuristicFeaturizer):

input_dim: Integer, the number of input features (default: 10).
hidden_dim1: Integer, the number of units in the first hidden layer (default: 50).
hidden_dim2: Integer, the number of units in the second hidden layer (default: 30).
output_dim: Integer, the number of output units (default: 1, for regression).


Helper Function: compile_cython_module:

cython_file: String, path to the Cython file (e.g., ineq_cython.pyx).
c_file: String, path to the C file (e.g., inequalities.c).
output_so: String, path to the output shared object file (e.g., ineq_cython.so).



Algorithm of FeaturizerPipeline
The FeaturizerPipeline integrates three feature engineering methods—InequalityFeaturizer, InformationRepurposedFeaturizer, and MetaheuristicFeaturizer—to generate and select the most informative features for a regression task. It combines their outputs, ranks features based on a criterion, and produces a final DataFrame with the top features. Here’s how the algorithm works:

Initialization:

Sets up file paths for Cython and C files (ineq_cython.pyx, inequalities.c, ineq_cython.so) based on base_dir.
Initializes the metaheuristic wrapper class (defaults to Wrapper or uses a custom class like CustomWrapper).
Configures logging for detailed output and error tracking.
Defines a PyTorch neural network (SimpleNN) for use in MetaheuristicFeaturizer.


Input Validation and Preprocessing:

Ensures the input DataFrame has a target column and no NaN values (handled by SimpleImputer with mean strategy).
Standardizes features using StandardScaler for methods requiring normalized inputs.


Dynamic Compilation:

Checks if the Cython module (ineq_cython.so) exists; if not, compiles ineq_cython.pyx and inequalities.c using compile_cython_module to enable InequalityFeaturizer.


InequalityFeaturizer:

Applies the InequalityFeaturizer to generate features based on mathematical inequalities (e.g., arithmetic mean, geometric mean) applied to feature combinations.
Parameters: level=2, stage=3, csv_path=None, report_path=None.
Computes mutual information scores for generated features against the target.
Stores feature scores in self.all_features and updates the report.


InformationRepurposedFeaturizer:

Runs within a timeout context (default: 60 seconds) to prevent hanging.
Generates features by training models (LinearRegression, DecisionTreeRegressor) on feature combinations and evaluating predictions using metrics like mutual information, Pearson correlation, etc.
Parameters: models=[("linear", LinearRegression()), ("dt", DecisionTreeRegressor())], prediction_mode="top_n", top_n=5, score_key="mutual_information", level=2, save_models=False, save_results_file=None, n_jobs=2.
Stores feature scores and updates the report.


MetaheuristicFeaturizer:

Applies feature selection using multiple methods (VarianceThreshold, SelectKBest, RFE, Permutation Importance, Lasso, Ridge, ElasticNet, RandomForest, XGBoost) and a metaheuristic approach.
Uses the provided wrapper_class (e.g., CustomWrapper) and wrapper_method="DISO" for metaheuristic optimization.
Evaluates feature subsets using cross-validation with a model (LinearRegression, DecisionTreeRegressor, or SimpleNN).
Parameters: top_n=5, problem_type="regression", scorer=make_scorer(mean_squared_error, greater_is_better=False), wrapper_population_size=10, wrapper_max_iter=20.
Stores selected features and updates the report.


Feature Selection:

Combines features from all three featurizers, excluding those in exclude_features.
Ranks features based on criterion:

"mutual_information": Selects the top top_n features with the highest mutual information scores.
"cross_val_score": Selects the top top_n features based on cross-validation scores using LinearRegression.


Creates a final DataFrame (best_df) with the selected features and target.


Evaluation and Output:

Evaluates the final feature set using 5-fold cross-validation with LinearRegression, reporting the mean squared error (MSE).
Saves the final DataFrame to output_csv and the pipeline report (including shapes, features, scores, and errors) to output_report.
Returns a dictionary with:

final_df: The final DataFrame with selected features and target.
feature_scores: Dictionary of all features and their scores.
selected_features: List of the top top_n feature names.

Usage
The FeaturizerPipeline integrates three featurizers to generate and select features:

InequalityFeaturizer: Creates features using mathematical inequalities (e.g., arithmetic mean, geometric mean).
InformationRepurposedFeaturizer: Generates features by training models on feature combinations and evaluating predictions.
MetaheuristicFeaturizer: Selects features using statistical methods and metaheuristic optimization (e.g., Differential Evolution).

Example
pythonimport pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from featurizer_pipeline import FeaturizerPipeline

# Custom wrapper for metaheuristic optimization
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

# Generate sample data
X, y = make_regression(n_samples=200, n_features=10, n_informative=5, noise=0.1, random_state=42)
df = pd.DataFrame(X, columns=[f'feature{i+1}' for i in range(10)])
df['target'] = y

# Run pipeline
pipeline = FeaturizerPipeline(wrapper_class=CustomWrapper)
result = pipeline.run(
    df,
    top_n=5,
    criterion="mutual_information",
    exclude_features=['feature1', 'feature2'],
    output_csv="best_features.csv",
    output_report="pipeline_report.txt"
)

# Print results
print("Best features:", result["selected_features"])
print("\nFinal DataFrame head:")
print(result["final_df"].head())
print("\nFeature scores:", {k: f"{v:.4f}" for k, v in result["feature_scores"].items()})
Parameters

FeaturizerPipeline(base_dir=None, wrapper_class=None, timeout_seconds=60):

base_dir: Directory for Cython and C files (default: script directory).
wrapper_class: Metaheuristic wrapper class (default: Wrapper from chekml).
timeout_seconds: Timeout for InformationRepurposedFeaturizer (default: 60).


run(df, top_n=5, criterion="mutual_information", exclude_features=None, output_csv="best_features.csv", output_report="pipeline_report.txt"):

df: Input DataFrame with features and a target column.
top_n: Number of top features to select.
criterion: "mutual_information" or "cross_val_score" for ranking features.
exclude_features: List of feature names to exclude.
output_csv: Path to save the final DataFrame.
output_report: Path to save the pipeline report.



Output

result: Dictionary containing:

final_df: DataFrame with the top top_n features and target.
feature_scores: Dictionary of all features and their scores (e.g., mutual information).
selected_features: List of the top top_n feature names.


output_csv: CSV file with the final DataFrame.
output_report: Text file with a detailed report, including shapes, selected features, scores, and errors.

Notes

Ensure the input DataFrame has a target column.
Missing values are handled by SimpleImputer (mean strategy).
The pipeline requires ineq_cython.pyx and inequalities.c in the specified base_dir for InequalityFeaturizer.
The wrapper_class must implement optimize, get_best_solution, and free methods (see CustomWrapper example).
The pipeline uses a timeout for InformationRepurposedFeaturizer to prevent hanging on large datasets.
PyTorch models require compatible hardware (CPU or GPU) and a defined criterion (e.g., nn.MSELoss).
