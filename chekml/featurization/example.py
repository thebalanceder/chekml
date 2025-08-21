import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from chekml.featurization.IF import InequalityFeaturizer
from chekml.featurization.IRF import InformationRepurposedFeaturizer
from chekml.featurization.MhF import MetaheuristicFeaturizer
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Set random seed for reproducibility
np.random.seed(42)

# Create synthetic DataFrames for classification and regression
def create_synthetic_dataframes(n_samples=100, problem_type="classification"):
    if problem_type == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=10,
            n_informative=6,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=10,
            n_informative=6,
            noise=0.1,
            random_state=42
        )
    
    # Create two DataFrames with overlapping features
    df1 = pd.DataFrame(X[:, :8], columns=[f"feature{i+1}" for i in range(8)])
    df1["target"] = y
    
    df2 = pd.DataFrame(X[:, 4:], columns=[f"feature{i+5}" for i in range(6)])
    df2["target"] = y
    
    return [df1, df2]

# Demonstration of InequalityFeaturizer
def demo_inequality_featurizer(df):
    print("\n=== InequalityFeaturizer Demonstration ===")
    try:
        featurizer = InequalityFeaturizer()
        
        # Add a custom inequality (for demonstration, requires manual C implementation)
        custom_ineq = """
def custom_ineq(x):
    return np.max(x) - np.min(x)
"""
        featurizer.add_inequality("custom_ineq", custom_ineq)
        
        # Print all inequalities
        print("\nAvailable Inequalities:")
        featurizer.print_inequalities()
        
        # Featurize the DataFrame
        result_df = featurizer.featurize(
            df,
            level=2,
            stage=3,
            csv_path="inequality_features.csv",
            report_path="inequality_mi_report.txt"
        )
        
        print("\nFirst few rows of InequalityFeaturizer result:")
        print(result_df.head())
        
        # Clean up
        featurizer.delete_all_inequalities()
        print("\nAfter deleting custom inequalities:")
        featurizer.print_inequalities()
        
        return result_df
    except Exception as e:
        print(f"InequalityFeaturizer Error: {e}")
        return None

# Demonstration of InformationRepurposedFeaturizer
def demo_information_repurposed_featurizer(df, problem_type="regression"):
    print("\n=== InformationRepurposedFeaturizer Demonstration ===")
    try:
        custom_models = [
            ("decision_tree", DecisionTreeRegressor(random_state=42)),
            ("xg_boost", XGBRegressor(random_state=42, n_estimators=50))
        ]
        
        custom_metrics = {
            "pearson": (lambda x, y: np.corrcoef(x, y)[0, 1], "maximize"),
            "spearman": (lambda x, y: np.corrcoef(np.argsort(x), np.argsort(y))[0, 1], "maximize")
        }
        
        result_df, metric_scores_df, feature_mi, trained_models = InformationRepurposedFeaturizer(
            df=df,
            models=custom_models,
            metrics=custom_metrics,
            prediction_mode="top_n",
            top_n=3,
            score_key="spearman",
            level=2,
            save_models=False,
            save_results_file="irf_results.txt",
            n_jobs=2
        )
        
        print("\nFeature Mutual Information with Target:")
        for feature, mi in feature_mi.items():
            print(f"{feature}: {mi:.4f}")
        
        print("\nFirst few rows of InformationRepurposedFeaturizer result:")
        print(result_df.head())
        
        print("\nMetric Scores DataFrame (first few rows):")
        print(metric_scores_df.head())
        
        return result_df
    except Exception as e:
        print(f"InformationRepurposedFeaturizer Error: {e}")
        return None

# Demonstration of MetaheuristicFeaturizer
def demo_metaheuristic_featurizer(dfs, problem_type="classification"):
    print("\n=== MetaheuristicFeaturizer Demonstration ===")
    try:
        result = MetaheuristicFeaturizer(
            dataframes=dfs,
            top_n=3,
            problem_type=problem_type
        )
        
        print("\nMetaheuristicFeaturizer Results:")
        for method, selected_df in result.items():
            print(f"\n{method}:")
            print(f"Shape: {selected_df.shape}")
            print("Selected Features:", [col for col in selected_df.columns if col != "target"])
            print(selected_df.head())
        
        return result
    except Exception as e:
        print(f"MetaheuristicFeaturizer Error: {e}")
        return None

# Main execution
if __name__ == "__main__":
    # Generate synthetic data for classification
    print("Generating synthetic classification data...")
    classification_dfs = create_synthetic_dataframes(n_samples=100, problem_type="classification")
    
    # Generate synthetic data for regression
    print("\nGenerating synthetic regression data...")
    regression_dfs = create_synthetic_dataframes(n_samples=100, problem_type="regression")
    
    # Print info about input DataFrames
    print("\nClassification DataFrames:")
    for i, df in enumerate(classification_dfs):
        print(f"\nDataFrame {i}:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(df.head())
    
    print("\nRegression DataFrame (using first DataFrame):")
    print(f"Shape: {regression_dfs[0].shape}")
    print(f"Columns: {list(regression_dfs[0].columns)}")
    print(regression_dfs[0].head())
    
    # Run demonstrations
    demo_inequality_featurizer(regression_dfs[0])
    demo_information_repurposed_featurizer(regression_dfs[0], problem_type="regression")
    demo_metaheuristic_featurizer(classification_dfs, problem_type="classification")
