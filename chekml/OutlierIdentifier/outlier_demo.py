import numpy as np
from sklearn.model_selection import train_test_split
from outlier_hybrid import OutlierHybrid
from robust_svm import RobustOneClassSVM, OutlierEvaluator

def main():
    # Generate synthetic data for outlier detection
    np.random.seed(42)
    data = np.random.randn(300, 5)
    data[::30] += 5  # Inject anomalies

    # Demo OutlierHybrid
    print("=== OutlierHybrid Demo ===")
    hybrid = OutlierHybrid(contamination=0.1, encoding_dim=10, epochs=50, batch_size=32)
    anomalies, scores = hybrid.detect_outliers(data)
    print("Anomaly Count:", np.sum(anomalies))

    # Demo hybrid regression
    X_train, y_train = np.random.randn(100, 5), np.random.randn(100)
    X_test = np.random.randn(20, 5)
    predictions = hybrid.hybrid_regression(X_train, y_train, X_test)
    print("Hybrid Model Predictions:", predictions[:5])  # Show first 5 predictions

    # Generate synthetic data for OutlierEvaluator
    X = np.random.randn(10000, 8)
    y = np.ones(10000)
    X_outliers = np.random.uniform(low=-5, high=4, size=(50, 8))
    y_outliers = -np.ones(50)
    X = np.vstack([X, X_outliers])
    y = np.hstack([y, y_outliers])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Demo OutlierEvaluator
    print("\n=== OutlierEvaluator Demo ===")
    evaluator = OutlierEvaluator()
    results = evaluator.evaluate_models(X_train, X_test, y_test)
    for model, score in results.items():
        print(f"{model}: {score:.4f}")

if __name__ == "__main__":
    main()
