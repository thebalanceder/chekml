import pandas as pd
import numpy as np
from chekml.featurization import InequalityFeaturizerFast

# Create sample data
np.random.seed(42)
data = pd.DataFrame({
    'A': np.abs(np.random.randn(100)),
    'B': np.abs(np.random.randn(100)),
    'C': np.abs(np.random.randn(100))
})
data['target'] = 0.5 * data['A'] + 0.5 * data['C'] + np.random.randn(100) * 0.1

def train_test_split_np(df, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    split = int(len(df) * (1 - test_size))
    return df.iloc[idx[:split]], df.iloc[idx[split:]]

# Train / test split
train_df, test_df = train_test_split_np(
    data,
    test_size=0.2,
)

# Initialize featurizer
featurizer = InequalityFeaturizerFast()

# Fit on training data only
featurizer.fit(train_df, level=2, stage=3, top_k=2)

# Transform test data
test_with_feats = featurizer.transform(test_df)
print(test_with_feats.head())
