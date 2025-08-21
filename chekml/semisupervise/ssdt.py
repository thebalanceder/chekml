import numpy as np

class SemiSupervisedDecisionTree:
    """Semi-Supervised Decision Tree classifier."""
    
    def __init__(self, base_tree, confidence_threshold=0.8):
        """Initialize the SSDT model.
        
        Args:
            base_tree: Base decision tree classifier
            confidence_threshold (float): Threshold for pseudo-labeling confidence
        """
        self.tree = base_tree
        self.confidence_threshold = confidence_threshold

    def pseudo_label(self, X_unlabeled):
        """Assign pseudo-labels to unlabeled data.
        
        Args:
            X_unlabeled (array-like): Unlabeled data
            
        Returns:
            tuple: (X_confident, y_pseudo) where X_confident are samples meeting
                  the confidence threshold and y_pseudo are their pseudo-labels
        """
        predictions = self.tree.predict_proba(X_unlabeled)
        max_probs = np.max(predictions, axis=1)
        pseudo_labels = np.argmax(predictions, axis=1)
        confident_samples = max_probs >= self.confidence_threshold
        return X_unlabeled[confident_samples], pseudo_labels[confident_samples]

    def fit(self, X_labeled, y_labeled, X_unlabeled):
        """Train on labeled and pseudo-labeled data.
        
        Args:
            X_labeled (array-like): Labeled input data
            y_labeled (array-like): Labels for labeled data
            X_unlabeled (array-like): Unlabeled input data
        """
        self.tree.fit(X_labeled, y_labeled)
        X_pseudo, y_pseudo = self.pseudo_label(X_unlabeled)
        if len(X_pseudo) > 0:
            X_combined = np.vstack([X_labeled, X_pseudo])
            y_combined = np.hstack([y_labeled, y_pseudo])
            self.tree.fit(X_combined, y_combined)

    def predict(self, X):
        """Predict new samples.
        
        Args:
            X (array-like): Input data
            
        Returns:
            array: Predicted labels
        """
        return self.tree.predict(X)
