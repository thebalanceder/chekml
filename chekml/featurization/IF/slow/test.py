import pandas as pd
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
