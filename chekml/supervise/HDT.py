import numpy as np
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

class HellingerDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def hellinger_distance(self, y_left, y_right):
        """Compute Hellinger Distance between left and right splits."""
        p = np.bincount(y_left, minlength=2) / len(y_left) if len(y_left) > 0 else np.zeros(2)
        q = np.bincount(y_right, minlength=2) / len(y_right) if len(y_right) > 0 else np.zeros(2)
        return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))  # ✅ Hellinger Distance Formula

    def best_split(self, X, y):
        """Find the best feature and threshold to split using Hellinger Distance."""
        best_score, best_feature, best_threshold = -1, None, None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                score = self.hellinger_distance(y[left_mask], y[right_mask])
                if score > best_score:
                    best_score, best_feature, best_threshold = score, feature, threshold

        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        """Recursively build the Hellinger Decision Tree."""
        if len(set(y)) == 1 or (self.max_depth and depth >= self.max_depth) or len(y) < self.min_samples_split:
            return Counter(y).most_common(1)[0][0]

        feature, threshold = self.best_split(X, y)
        if feature is None:
            return Counter(y).most_common(1)[0][0]

        left_mask, right_mask = X[:, feature] <= threshold, X[:, feature] > threshold
        left_subtree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return {"feature": feature, "threshold": threshold, "left": left_subtree, "right": right_subtree}

    def fit(self, X, y):
        """Train the Hellinger Decision Tree."""
        self.tree = self.build_tree(X, y)

    def predict_sample(self, node, sample):
        """Predict a single sample."""
        if not isinstance(node, dict):
            return node
        feature, threshold = node["feature"], node["threshold"]
        return self.predict_sample(node["left"], sample) if sample[feature] <= threshold else self.predict_sample(node["right"], sample)

    def predict(self, X):
        """Predict multiple samples."""
        return np.array([self.predict_sample(self.tree, sample) for sample in X])
# 📌 2️⃣ Generate an imbalanced dataset (90% in one class)
X, y = make_classification(n_samples=5000, n_features=10, n_informative=5, 
                           n_redundant=2, weights=[0.9, 0.1], random_state=42)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 3️⃣ Train Hellinger Distance Decision Tree
hddt = HellingerDecisionTree(max_depth=10, min_samples_split=5)
hddt.fit(X_train, y_train)
hddt_pred = hddt.predict(X_test)

# 📌 4️⃣ Train Standard Decision Tree (Gini-Based)
gini_tree = DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_split=5, random_state=42)
gini_tree.fit(X_train, y_train)
gini_pred = gini_tree.predict(X_test)

# 📌 5️⃣ Evaluate Models
hddt_acc = accuracy_score(y_test, hddt_pred)
hddt_f1 = f1_score(y_test, hddt_pred, average="weighted")

gini_acc = accuracy_score(y_test, gini_pred)
gini_f1 = f1_score(y_test, gini_pred, average="weighted")

# 📌 6️⃣ Display Results
print("\n🔍 Model Performance on Imbalanced Dataset:")
print(f"Hellinger Distance Decision Tree  - Accuracy: {hddt_acc:.4f}, F1-Score: {hddt_f1:.4f}")
print(f"Gini-Based Decision Tree          - Accuracy: {gini_acc:.4f}, F1-Score: {gini_f1:.4f}")
