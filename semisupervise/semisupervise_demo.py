import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from alvc import ALVC
from ssdt import SemiSupervisedDecisionTree

def demo_alvc():
    """Demonstrate ALVC clustering."""
    # Generate synthetic data
    n_samples = 300
    n_features = 2
    n_clusters = 3
    X, y_true = make_blobs(n_samples=n_samples, n_features=n_features, 
                          centers=n_clusters, cluster_std=1.0, random_state=42)

    # Initialize and fit ALVC
    alvc = ALVC(n_clusters=n_clusters)
    alvc.fit(X)

    # Predict clusters
    y_pred = alvc.predict(X)

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50)
    plt.scatter(alvc.cluster_centers_[:, 0], alvc.cluster_centers_[:, 1], 
               c='red', marker='x', s=200, linewidths=3)
    plt.title("ALVC Clustering Demo")
    plt.show()

def demo_ssdt():
    """Demonstrate Semi-Supervised Decision Tree classification."""
    # Generate synthetic data
    X, y = make_classification(n_samples=10000, n_features=50, n_informative=30, 
                            n_redundant=10, n_classes=2, weights=[0.8, 0.2], 
                            class_sep=0.5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Mask labels for semi-supervised learning (10% labeled)
    y_train_semi = y_train.copy()
    num_labeled = int(0.05 * len(y_train))
    y_train_semi[num_labeled:] = -1

    # Train SSDT
    base_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
    semi_tree = SemiSupervisedDecisionTree(base_tree)
    semi_tree.fit(X_train[:num_labeled], y_train[:num_labeled], X_train[num_labeled:])

    # Predict and evaluate
    y_pred = semi_tree.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    print("Semi-Supervised Decision Tree Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")

if __name__ == "__main__":
    print("Running ALVC Demo...")
    demo_alvc()
    print("\nRunning SSDT Demo...")
    demo_ssdt()
