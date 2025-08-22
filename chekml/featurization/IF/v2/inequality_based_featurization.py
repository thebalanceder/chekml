import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import io
import os
import subprocess
import sys
import sysconfig
import warnings
warnings.filterwarnings('ignore')

try:
    from IF.v2.ineq_cython import compute_features_cython
except ImportError:
    print("Cython module not found. Please compile ineq_cython.pyx first.")
    compute_features_cython = None

class InequalityFeaturizer:
    def __init__(self):
        self.user_inequalities = {}  # name: source_code (for reference only)
        self.c_file_path = os.path.join(os.path.dirname(__file__),"inequalities.c")
        self.cython_file_path = os.path.join(os.path.dirname(__file__),"ineq_cython.pyx")
        self.default_inequalities = [
            "am", "gm", "hm", "qm", "pm3", "pm_neg1", "lehmer2", "lehmer05",
            "log_mean", "identric", "heronian", "contra_hm", "rms",
            "pm4", "pm2", "pm_neg2", "lehmer3", "lehmer_neg1", "centroidal",
            "seiffert", "neuman_sandor", "log_mean_gen", "stolarsky2",
            "pm6", "pm_neg3", "lehmer4", "lehmer_neg2", "exp_mean",
            "quad_entropy", "wgm", "hyperbolic", "stolarsky3", "midrange"
        ]
    
    def add_inequality(self, name, source_code):
        """Add a user-defined inequality (requires manual C implementation)."""
        warnings.warn(
            "Automatic Python-to-C translation is not implemented. "
            "Manually add the C implementation for '{}' to '{}' and update 'inequalities.h'. "
            "The Python source will be stored for reference."
            .format(name, self.c_file_path)
        )
        self.user_inequalities[name] = source_code
        self._compile_cython()
    
    def delete_inequality(self, name):
        """Delete a user-defined inequality."""
        if name in self.user_inequalities:
            del self.user_inequalities[name]
            warnings.warn(
                "Removed '{}' from user-defined inequalities. "
                "Manually remove its C implementation from '{}' and 'inequalities.h', then recompile."
                .format(name, self.c_file_path)
            )
            self._compile_cython()
        else:
            raise ValueError(f"Inequality '{name}' not found in user-defined inequalities.")
    
    def print_inequalities(self):
        """Print all default and user-defined inequalities."""
        print("\nDefault Inequalities:")
        for ineq in self.default_inequalities:
            print(f"  - {ineq}")
        
        print("\nUser-Defined Inequalities:")
        if not self.user_inequalities:
            print("  None")
        else:
            for name, source in self.user_inequalities.items():
                print(f"  - {name}:")
                # Indent source code lines for readability
                for line in source.strip().split('\n'):
                    print(f"      {line}")
    
    def delete_all_inequalities(self):
        """Delete all user-defined inequalities."""
        if self.user_inequalities:
            removed = list(self.user_inequalities.keys())
            self.user_inequalities.clear()
            warnings.warn(
                f"Removed all user-defined inequalities: {removed}. "
                f"Manually remove their C implementations from '{self.c_file_path}' "
                "and 'inequalities.h', then recompile."
            )
            self._compile_cython()
        else:
            print("No user-defined inequalities to delete.")
    
    def _compile_cython(self):
        """Compile the Cython module."""
        try:
            # Cythonize the .pyx file
            subprocess.run([
                "cython", "-3", "--force", self.cython_file_path
            ], check=True)
            # Compile the generated C file with gcc
            python_include = sysconfig.get_path('include')
            numpy_include = np.get_include()
            subprocess.run([
                "gcc", "-shared", "-pthread", "-fPIC", "-O3",
                "-o", "ineq_cython.so", "ineq_cython.c", "inequalities.c",
                "-lm", "-I", numpy_include, "-I", python_include,
                "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"
            ], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Compilation failed: {e}")
    
    def featurize(self, df, level=1, stage=1, csv_path=None, report_path=None):
        """Perform inequality-based featurization."""
        if df.isna().any().any():
            raise ValueError("Input DataFrame contains NaN values")
        if 'target' not in df.columns:
            raise ValueError("DataFrame must contain 'target' column")
        
        if csv_path is not None:
            if not isinstance(csv_path, str) or not csv_path.endswith('.csv'):
                raise ValueError("csv_path must be a string ending with '.csv'")
        if report_path is not None:
            if not isinstance(report_path, str) or not report_path.endswith('.txt'):
                raise ValueError("report_path must be a string ending with '.txt'")
        
        features = [col for col in df.columns if col != 'target']
        new_df = df.copy()
        
        # Convert DataFrame to NumPy array
        data = df[features].to_numpy()
        cols = np.arange(len(features), dtype=np.int32)
        
        # Call Cython function
        if compute_features_cython is None:
            raise RuntimeError("Cython module not available.")
        
        output, new_names = compute_features_cython(data, cols, level, stage)
        
        # Add new features to DataFrame
        for i, name in enumerate(new_names):
            new_df[f"f_{name}"] = output[:, i]
        
        # Compute mutual information
        X = new_df.drop('target', axis=1)
        y = new_df['target']
        X = X.fillna(X.mean())
        
        mi_scores = mutual_info_regression(X, y)
        mi_dict = dict(zip(X.columns, mi_scores))
        
        output_str = io.StringIO()
        output_str.write("\nMutual Information Scores:\n")
        for feature, score in mi_dict.items():
            output_str.write(f"{feature}: {score:.4f}\n")
        report_content = output_str.getvalue()
        print(report_content, end='')
        output_str.close()
        
        if report_path is not None:
            try:
                with open(report_path, 'w') as f:
                    f.write(report_content)
            except Exception as e:
                print(f"Error saving report to {report_path}: {e}")
        
        if csv_path is not None:
            try:
                new_df.to_csv(csv_path, index=False)
            except Exception as e:
                print(f"Error saving DataFrame to {csv_path}: {e}")
        
        return new_df

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_df = pd.DataFrame({
        'A': np.abs(np.random.randn(100)),
        'B': np.abs(np.random.randn(100)),
        'C': np.abs(np.random.randn(100))
    })
    sample_df['target'] = 0.5 * sample_df['A'] + 0.5 * sample_df['C'] + np.random.randn(100) * 0.1
    
    featurizer = InequalityFeaturizer()
    
    # Add a user-defined inequality (requires manual C implementation)
    user_ineq = """
def custom_ineq(x):
    return np.max(x) - np.min(x)
"""
    featurizer.add_inequality("custom_ineq", user_ineq)
    
    # Print all inequalities
    featurizer.print_inequalities()
    
    # Delete all user-defined inequalities
    featurizer.delete_all_inequalities()
    
    # Print inequalities again to verify deletion
    featurizer.print_inequalities()
    
    result_df = featurizer.featurize(
        sample_df, 
        level=2, 
        stage=3, 
        csv_path='output_features.csv', 
        report_path='mi_report.txt'
    )
    print("\nFirst few rows of resulting DataFrame:")
    print(result_df.head())
