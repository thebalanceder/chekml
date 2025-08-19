import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.feature_selection import mutual_info_regression
import warnings
import io
import os

warnings.filterwarnings('ignore')

def Inequality_Featurizer(df, level=1, stage=1, csv_path=None, report_path=None):
    """
    Creates new features based on inequality equations and computes mutual information.
    
    Parameters:
    df (pandas.DataFrame): Input dataframe containing features and 'target' column
    level (int): Controls feature combination level (1: single, 2: pairs, etc.)
    stage (int): Number of top inequality equations to use (1: top, 2: top 2, etc.)
    csv_path (str, optional): Path to save the resulting DataFrame as CSV
    report_path (str, optional): Path to save the mutual information report as TXT
    
    Returns:
    pandas.DataFrame: Dataframe with original and new features
    """
    # Validate input DataFrame
    if df.isna().any().any():
        raise ValueError("Input DataFrame contains NaN values")
    if 'target' not in df.columns:
        raise ValueError("DataFrame must contain 'target' column")
    
    # Validate file paths
    if csv_path is not None:
        if not isinstance(csv_path, str) or not csv_path.endswith('.csv'):
            raise ValueError("csv_path must be a string ending with '.csv'")
    if report_path is not None:
        if not isinstance(report_path, str) or not report_path.endswith('.txt'):
            raise ValueError("report_path must be a string ending with '.txt'")
    
    def am(x): return np.mean(x)
    def gm(x): 
        x = np.maximum(x, 1e-10)
        return np.exp(np.mean(np.log(x)))
    def hm(x): 
        x = np.maximum(x, 1e-10)
        return len(x) / np.sum(1 / x)
    def qm(x): return np.sqrt(np.mean(np.square(x)))
    def pm3(x): 
        x = np.maximum(x, 1e-10)
        return np.mean(x**3)**(1/3)
    def pm_neg1(x): 
        x = np.maximum(x, 1e-10)
        return np.mean(x**(-1))**(-1)
    def lehmer2(x): 
        x = np.maximum(x, 1e-10)
        return np.sum(x**2) / np.sum(x)
    def lehmer05(x): 
        x = np.maximum(x, 1e-10)
        return np.sum(x**0.5) / np.sum(x**-0.5)
    def log_mean(x): 
        x = np.maximum(x, 1e-10)
        return np.mean(x) if len(x) != 2 or x[0] == x[1] else (x[1] - x[0]) / np.log(x[1]/x[0])
    def identric(x):
        x = np.maximum(x, 1e-10)
        if len(x) == 2 and x[0] != x[1]:
            return x[0]**(x[1]/(x[1]-x[0])) * x[1]**(x[0]/(x[0]-x[1]))
        return np.exp(np.mean(np.log(x)) - 1)
    def heronian(x): 
        x = np.maximum(x, 1e-10)
        return np.mean(x) if len(x) != 2 else (x[0] + np.sqrt(x[0]*x[1]) + x[1])/3
    def contra_hm(x): 
        x = np.maximum(x, 1e-10)
        return np.mean(x**2) / np.mean(x)
    def rms(x): return np.sqrt(np.mean(x**2))
    def pm4(x): 
        x = np.maximum(x, 1e-10)
        return np.mean(x**4)**(1/4)
    def pm2(x): 
        x = np.maximum(x, 1e-10)
        return np.mean(x**2)**(1/2)
    def pm_neg2(x): 
        x = np.maximum(x, 1e-10)
        return np.mean(x**(-2))**(-1/2)
    def lehmer3(x): 
        x = np.maximum(x, 1e-10)
        return np.sum(x**3) / np.sum(x**2)
    def lehmer_neg1(x): 
        x = np.maximum(x, 1e-10)
        return np.sum(x**-1) / np.sum(x**-2)
    def centroidal(x): 
        x = np.maximum(x, 1e-10)
        weights = np.arange(1, len(x) + 1)
        return np.average(x, weights=weights)
    def seiffert(x): 
        x = np.maximum(x, 1e-10)
        if len(x) == 2 and x[0] != x[1]:
            a, b = x[0], x[1]
            return (a - b) / (2 * np.arcsin((a - b) / (a + b)))
        return np.mean(x)
    def neuman_sandor(x): 
        x = np.maximum(x, 1e-10)
        if len(x) == 2 and x[0] != x[1]:
            a, b = x[0], x[1]
            return (a - b) / (2 * np.arcsinh((a - b) / (a + b)))
        return np.mean(x)
    def log_mean_gen(x): 
        x = np.maximum(x, 1e-10)
        if len(x) > 1:
            pairs = list(combinations(x, 2))
            results = [(b - a) / np.log(b/a) if a != b else np.mean([a, b]) for a, b in pairs]
            return np.mean(results)
        return np.mean(x)
    def stolarsky2(x): 
        x = np.maximum(x, 1e-10)
        if len(x) == 2 and x[0] != x[1]:
            a, b = x[0], x[1]
            return ((b**2 - a**2) / (2 * (b - a)))**(1/1)
        return np.mean(x)
    def pm6(x): 
        x = np.maximum(x, 1e-10)
        return np.mean(x**6)**(1/6)
    def pm_neg3(x): 
        x = np.maximum(x, 1e-10)
        return np.mean(x**(-3))**(-1/3)
    def lehmer4(x): 
        x = np.maximum(x, 1e-10)
        return np.sum(x**4) / np.sum(x**3)
    def lehmer_neg2(x): 
        x = np.maximum(x, 1e-10)
        return np.sum(x**-2) / np.sum(x**-3)
    def exp_mean(x): 
        x = np.maximum(x, 1e-10)
        return np.log(np.mean(np.exp(x)))
    def quad_entropy(x): 
        x = np.maximum(x, 1e-10)
        p = x / np.sum(x)
        return -np.sum(p**2 * np.log(np.maximum(p, 1e-10)))
    def wgm(x): 
        x = np.maximum(x, 1e-10)
        weights = x / np.sum(x)
        return np.exp(np.sum(weights * np.log(x)))
    def hyperbolic(x): 
        x = np.maximum(x, 1e-10)
        if len(x) == 2 and x[0] != x[1]:
            a, b = x[0], x[1]
            return (a + b) / (2 * np.cosh((a - b) / (a + b)))
        return np.mean(x)
    def stolarsky3(x): 
        x = np.maximum(x, 1e-10)
        if len(x) == 2 and x[0] != x[1]:
            a, b = x[0], x[1]
            return ((b**3 - a**3) / (3 * (b - a)))**(1/2)
        return np.mean(x)
    def midrange(x): 
        x = np.maximum(x, 1e-10)
        return (np.max(x) + np.min(x)) / 2
    
    inequalities = [
        ('am', am), ('gm', gm), ('hm', hm), ('qm', qm),
        ('pm3', pm3), ('pm_neg1', pm_neg1), ('lehmer2', lehmer2),
        ('lehmer05', lehmer05), ('log_mean', log_mean),
        ('identric', identric), ('heronian', heronian),
        ('contra_hm', contra_hm), ('rms', rms),
        ('pm4', pm4), ('pm2', pm2), ('pm_neg2', pm_neg2),
        ('lehmer3', lehmer3), ('lehmer_neg1', lehmer_neg1),
        ('centroidal', centroidal), ('seiffert', seiffert),
        ('neuman_sandor', neuman_sandor), ('log_mean_gen', log_mean_gen),
        ('stolarsky2', stolarsky2), ('pm6', pm6), ('pm_neg3', pm_neg3),
        ('lehmer4', lehmer4), ('lehmer_neg2', lehmer_neg2),
        ('exp_mean', exp_mean), ('quad_entropy', quad_entropy),
        ('wgm', wgm), ('hyperbolic', hyperbolic),
        ('stolarsky3', stolarsky3), ('midrange', midrange)
    ]
    
    features = [col for col in df.columns if col != 'target']
    new_df = df.copy()
    
    for r in range(1, level + 1):
        for comb in combinations(features, r):
            comb_name = '_'.join(comb)
            values = df[list(comb)].values
            
            ineq_results = []
            for ineq_name, ineq_func in inequalities:
                try:
                    result = np.apply_along_axis(ineq_func, 1, values)
                    if not np.all(np.isnan(result)):
                        ineq_results.append((ineq_name, result))
                except:
                    continue
            
            avg_values = [(name, np.nanmean(np.abs(res))) for name, res in ineq_results]
            avg_values.sort(key=lambda x: x[1], reverse=True)
            top_ineqs = avg_values[:min(stage, len(avg_values))]
            
            for ineq_name, _ in top_ineqs:
                for name, result in ineq_results:
                    if name == ineq_name:
                        new_feature_name = f"{comb_name}_{ineq_name}"
                        new_df[new_feature_name] = result
                        new_df[new_feature_name] = new_df[new_feature_name].replace([np.inf, -np.inf], np.nan)
                        new_df[new_feature_name] = new_df[new_feature_name].fillna(new_df[new_feature_name].mean())
    
    X = new_df.drop('target', axis=1)
    y = new_df['target']
    X = X.fillna(X.mean())
    
    mi_scores = mutual_info_regression(X, y)
    mi_dict = dict(zip(X.columns, mi_scores))
    
    output = io.StringIO()
    output.write("\nMutual Information Scores:\n")
    for feature, score in mi_dict.items():
        output.write(f"{feature}: {score:.4f}\n")
    report_content = output.getvalue()
    print(report_content, end='')
    output.close()
    
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
    np.random.seed(42)
    sample_df = pd.DataFrame({
        'A': np.abs(np.random.randn(100)),
        'B': np.abs(np.random.randn(100)),
        'C': np.abs(np.random.randn(100))
    })
    sample_df['target'] = 0.5 * sample_df['A'] + 0.5 * sample_df['C'] + np.random.randn(100) * 0.1
    result_df = Inequality_Featurizer(
        sample_df, 
        level=2, 
        stage=3, 
        csv_path='output_features.csv', 
        report_path='mi_report.txt'
    )
    print("\nFirst few rows of resulting DataFrame:")
    print(result_df.head())
