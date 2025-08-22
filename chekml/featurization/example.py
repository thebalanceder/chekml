from featurizer_pipeline import FeaturizerPipeline
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

# Custom wrapper (optional)
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

# Run pipeline with custom wrapper and excluded features
pipeline = FeaturizerPipeline(wrapper_class=CustomWrapper)
result = pipeline.run(
    df,
    top_n=5,
    criterion="mutual_information",
    exclude_features=['feature1', 'feature2'],
    output_csv="best_features.csv",
    output_report="pipeline_report.txt"
)

print("Best features:", result["selected_features"])
