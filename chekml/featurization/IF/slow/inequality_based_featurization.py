import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import io
import os
import itertools
import warnings
warnings.filterwarnings('ignore')

class InequalityFeaturizer:
    def __init__(self):
        self.inequalities = {}  # name: func (Python functions)
        self.user_sources = {}  # name: source_code (for user-defined, for reference/printing)
        self._init_default_inequalities()
    
    def _init_default_inequalities(self):
        MIN_VALUE = 1e-10
        
        def am(x):
            return np.mean(x)
        
        def gm(x):
            x = np.maximum(x, MIN_VALUE)
            return np.exp(np.mean(np.log(x)))
        
        def hm(x):
            x = np.maximum(x, MIN_VALUE)
            return len(x) / np.sum(1.0 / x)
        
        def qm(x):
            return np.sqrt(np.mean(x ** 2))
        
        def pm3(x):
            x = np.maximum(x, MIN_VALUE)
            return np.mean(x ** 3) ** (1.0 / 3.0)
        
        def pm_neg1(x):
            x = np.maximum(x, MIN_VALUE)
            return len(x) / np.sum(1.0 / x)
        
        def lehmer2(x):
            x = np.maximum(x, MIN_VALUE)
            num = np.sum(x ** 2)
            denom = np.sum(x)
            return num / denom
        
        def lehmer05(x):
            x = np.maximum(x, MIN_VALUE)
            num = np.sum(np.sqrt(x))
            denom = np.sum(1.0 / np.sqrt(x))
            return num / denom
        
        def log_mean(x):
            if len(x) == 2 and x[0] != x[1]:
                a, b = sorted(np.maximum(x, MIN_VALUE))
                return (b - a) / np.log(b / a)
            else:
                return np.mean(x)
        
        def identric(x):
            if len(x) == 2 and x[0] != x[1]:
                a, b = sorted(np.maximum(x, MIN_VALUE))
                return a ** (b / (b - a)) * b ** (a / (a - b))
            else:
                x = np.maximum(x, MIN_VALUE)
                return np.exp(np.mean(np.log(x)) - 1.0)
        
        def heronian(x):
            if len(x) == 2:
                a, b = sorted(np.maximum(x, MIN_VALUE))
                return (a + np.sqrt(a * b) + b) / 3.0
            else:
                return np.mean(x)
        
        def contra_hm(x):
            sum_sq = np.sum(x ** 2)
            sum_x = np.sum(x)
            n = len(x)
            return (sum_sq / n) / (sum_x / n)
        
        def rms(x):
            return np.sqrt(np.mean(x ** 2))
        
        def pm4(x):
            x = np.maximum(x, MIN_VALUE)
            return np.mean(x ** 4) ** (1.0 / 4.0)
        
        def pm2(x):
            x = np.maximum(x, MIN_VALUE)
            return np.mean(x ** 2) ** (1.0 / 2.0)
        
        def pm_neg2(x):
            x = np.maximum(x, MIN_VALUE)
            return (len(x) / np.sum(x ** -2)) ** (1.0 / 2.0)
        
        def lehmer3(x):
            x = np.maximum(x, MIN_VALUE)
            num = np.sum(x ** 3)
            denom = np.sum(x ** 2)
            return num / denom
        
        def lehmer_neg1(x):
            x = np.maximum(x, MIN_VALUE)
            num = np.sum(1.0 / x)
            denom = np.sum(1.0 / (x ** 2))
            return num / denom
        
        def centroidal(x):
            weights = np.arange(1, len(x) + 1)
            return np.sum(weights * x) / np.sum(weights)
        
        def seiffert(x):
            if len(x) == 2 and x[0] != x[1]:
                a, b = sorted(np.maximum(x, MIN_VALUE))
                return (a - b) / (2.0 * np.arcsin((a - b) / (a + b)))
            else:
                return np.mean(x)
        
        def neuman_sandor(x):
            if len(x) == 2 and x[0] != x[1]:
                a, b = sorted(np.maximum(x, MIN_VALUE))
                return (a - b) / (2.0 * np.arcsinh((a - b) / (a + b)))
            else:
                return np.mean(x)
        
        def log_mean_gen(x):
            n = len(x)
            if n > 1:
                sum_val = 0.0
                count = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        a, b = sorted(np.maximum([x[i], x[j]], MIN_VALUE))
                        sum_val += a if a == b else (b - a) / np.log(b / a)
                        count += 1
                return sum_val / count
            else:
                return x[0]
        
        def stolarsky2(x):
            if len(x) == 2 and x[0] != x[1]:
                a, b = sorted(np.maximum(x, MIN_VALUE))
                return ((b ** 2 - a ** 2) / (2.0 * (b - a))) ** 1.0
            else:
                return np.mean(x)
        
        def pm6(x):
            x = np.maximum(x, MIN_VALUE)
            return np.mean(x ** 6) ** (1.0 / 6.0)
        
        def pm_neg3(x):
            x = np.maximum(x, MIN_VALUE)
            return (len(x) / np.sum(x ** -3)) ** (1.0 / 3.0)
        
        def lehmer4(x):
            x = np.maximum(x, MIN_VALUE)
            num = np.sum(x ** 4)
            denom = np.sum(x ** 3)
            return num / denom
        
        def lehmer_neg2(x):
            x = np.maximum(x, MIN_VALUE)
            num = np.sum(1.0 / (x ** 2))
            denom = np.sum(1.0 / (x ** 3))
            return num / denom
        
        def exp_mean(x):
            return np.log(np.mean(np.exp(x)))
        
        def quad_entropy(x):
            x = np.maximum(x, MIN_VALUE)
            sum_x = np.sum(x)
            p = x / sum_x
            return -np.sum(p ** 2 * np.log(np.maximum(p, MIN_VALUE)))
        
        def wgm(x):
            x = np.maximum(x, MIN_VALUE)
            sum_x = np.sum(x)
            w = x / sum_x
            return np.exp(np.sum(w * np.log(x)))
        
        def hyperbolic(x):
            if len(x) == 2 and x[0] != x[1]:
                a, b = sorted(np.maximum(x, MIN_VALUE))
                return (a + b) / (2.0 * np.cosh((a - b) / (a + b)))
            else:
                return np.mean(x)
        
        def stolarsky3(x):
            if len(x) == 2 and x[0] != x[1]:
                a, b = sorted(np.maximum(x, MIN_VALUE))
                return ((b ** 3 - a ** 3) / (3.0 * (b - a))) ** (1.0 / 2.0)
            else:
                return np.mean(x)
        
        def midrange(x):
            return (np.min(x) + np.max(x)) / 2.0
        
        default_funcs = {
            "am": am, "gm": gm, "hm": hm, "qm": qm,
            "pm3": pm3, "pm_neg1": pm_neg1, "lehmer2": lehmer2, "lehmer05": lehmer05,
            "log_mean": log_mean, "identric": identric, "heronian": heronian,
            "contra_hm": contra_hm, "rms": rms, "pm4": pm4, "pm2": pm2,
            "pm_neg2": pm_neg2, "lehmer3": lehmer3, "lehmer_neg1": lehmer_neg1,
            "centroidal": centroidal, "seiffert": seiffert, "neuman_sandor": neuman_sandor,
            "log_mean_gen": log_mean_gen, "stolarsky2": stolarsky2, "pm6": pm6,
            "pm_neg3": pm_neg3, "lehmer4": lehmer4, "lehmer_neg2": lehmer_neg2,
            "exp_mean": exp_mean, "quad_entropy": quad_entropy, "wgm": wgm,
            "hyperbolic": hyperbolic, "stolarsky3": stolarsky3, "midrange": midrange
        }
        
        self.inequalities = default_funcs
    
    def add_inequality(self, name, source_code):
        """Add a user-defined inequality as a Python function."""
        if name in self.inequalities:
            raise ValueError(f"Inequality '{name}' already exists.")
        
        # Execute the source code to define the function
        local_dict = {}
        try:
            exec(source_code, {"np": np}, local_dict)
            func = local_dict.get(name)
            if not callable(func):
                raise ValueError(f"Source code must define a function named '{name}'.")
        except Exception as e:
            raise ValueError(f"Error compiling user-defined inequality '{name}': {e}")
        
        self.inequalities[name] = func
        self.user_sources[name] = source_code
        print(f"Successfully added user-defined inequality '{name}'.")
    
    def delete_inequality(self, name):
        """Delete a user-defined inequality."""
        if name in self.user_sources:
            del self.inequalities[name]
            del self.user_sources[name]
            print(f"Successfully deleted user-defined inequality '{name}'.")
        else:
            raise ValueError(f"User-defined inequality '{name}' not found.")
    
    def print_inequalities(self):
        """Print all default and user-defined inequalities."""
        print("\nDefault Inequalities:")
        default_names = [name for name in self.inequalities if name not in self.user_sources]
        for name in default_names:
            print(f"  - {name}")
        
        print("\nUser-Defined Inequalities:")
        if not self.user_sources:
            print("  None")
        else:
            for name, source in self.user_sources.items():
                print(f"  - {name}:")
                # Indent source code lines for readability
                for line in source.strip().split('\n'):
                    print(f"      {line}")
    
    def delete_all_inequalities(self):
        """Delete all user-defined inequalities."""
        if self.user_sources:
            removed = list(self.user_sources.keys())
            for name in removed:
                del self.inequalities[name]
            self.user_sources.clear()
            print(f"Removed all user-defined inequalities: {removed}.")
        else:
            print("No user-defined inequalities to delete.")
    
    def compute_features(self, data, cols, level, stage):
        """Python implementation of compute_features."""
        rows, num_cols = data.shape
        output = []
        output_names = []
        
        class Result:
            def __init__(self, name, value):
                self.name = name
                self.value = value
        
        def compare_results(a, b):
            return 1 if a.value > b.value else -1 if a.value < b.value else 0
        
        for r in range(1, level + 1):
            for comb in itertools.combinations(range(num_cols), r):
                combo_cols = [cols[i] for i in comb]
                results = []
                
                for name, func in self.inequalities.items():
                    temp = np.zeros(rows)
                    for j in range(rows):
                        x = data[j, list(comb)]
                        try:
                            temp[j] = func(x)
                        except:
                            temp[j] = np.nan
                    
                    if not np.isnan(temp[0]):
                        avg = np.mean(np.abs(temp))
                        results.append(Result(name, avg))
                
                # Sort descending by value
                results.sort(key=lambda res: res.value, reverse=True)
                
                top_count = min(stage, len(results))
                for i in range(top_count):
                    name = results[i].name
                    func = self.inequalities[name]
                    new_col = np.zeros(rows)
                    for j in range(rows):
                        x = data[j, list(comb)]
                        new_col[j] = func(x)
                    
                    output.append(new_col)
                    
                    # Construct name
                    combo_str = '_'.join(map(str, combo_cols))
                    feature_name = f"{combo_str}_{name}"
                    output_names.append(feature_name)
        
        if output:
            output = np.column_stack(output)
        else:
            output = np.empty((rows, 0))
        
        return output, output_names
    
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
        
        # Call Python compute_features
        output, new_names = self.compute_features(data, cols, level, stage)
        
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
    
    # Add a user-defined inequality
    user_ineq = """
def custom_ineq(x):
    import numpy as np
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
