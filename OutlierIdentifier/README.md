 # Algorithm Overview and Usage

 ## 1.Outlier Hybrid
 - Description: `OutlierHybrid` combines multiple outlier detection techniques like Isolation Forest, Local Outlier Factor (LOF) and an Autoencoder into a hybrid model.
 - Parameters:
   - `contamination` (float,default=0.1) :Expected proportion of outliers in the data. Used by Isolation Forset and LOF to set decision thresholds.
   - `encoding_dim` (int,default=10) :Number of units of the autoender's hidden layer. Smaller values reduce complexity but may miss patterns.
   - `epochs` (int,default=50) :Number of training epochs for the autoencoder. More epochs improve training but increase computation time.
   - `batch_size` (int,default=32) :Batch size for autoencoder training. Smaller Batches may improve training stability.
- Algorithm Explanation:
  - Outlier detection:
    - Scales input data using `MinMaxScaler`
    - Applies Isolation Forest to compute anomaly scores based on decision tree isolation.
    - Use LOF to compute scores based on local density deviations
    - Trains an autoencoder to reconstruct the data and computes reconstruction errors.
    - Combines score (0.5*isolation forest + 0.3*LOF + 0.2*autoencoder) and thresholds them to identify anomalies.
  - Hybrid Regression:
    - Trains Linear Regression and Decision Tree Regression models on training data.
    - Averages predicitons from both models for test data.
   
  - Usage Example:
 
**Python**
```python
from chekml.OutlierIdentifier.outlier_hybrid import OutlierHybrid
import numpy as np

data = np.random.randn(300,5)
data[::30] += 5
hybrid = OutlierHybrid(contamination=0.1,encoding_dim=10,epochs=50,batch_size=32)
anomalies, scores=hybrid.detect_outliers(data)
print("Anomaly Count:",np.sum(anomalies))

X_train, y_train = np.random.randn(100,5),np.random.randn(100)
X_test=np.random.randn(20,5)
predictions = hybrid.hybrid_regression(X_train,y_train,X_test)
print("Predictions:",predictions[:5])
```

 ## 2.Robust One Class Support Vector Machine
 - Description: `RobustOneClassSVM` is an enhanced one-class SVM that uses Random Fourier Features (RFF) for kernel approximation and a Huber loss function for robust optimization. It is designed for anomaly detection in high-dimensional datasets and uses a batch-based training approach for scalability.
 - Parameters:
   - `nu` (float,default=0.1) :Upper bound on the fraction of training errors and lower bound on the fraction of support vectors. Controls the trade-off between sensitivity and specificity.
   - `gamma` (float,default=0.1) :Kernel coefficient for the Random Fourier Features. Higher values lead to more complex decision boundaries.
   - `batch_size` (int,default=32) :Number of samples per training batch. Smaller batches reduces memory usage but may slow convergence.
   - `delta` (float, default=1.0) :Threshold for the Huber loss function. Smaller values make the loss more robust to outliers.
   - `rff_dim` (int,default=100) :Dimension of the Random Fourier Features.
 - Algorithm Explanation:
   - Transforms input data into a feature space using Random Fourier Features to approximate an RBF kernel.
   - Optimizes a one-class SVM objective using a Huber loss function to reduce sensitivity to outliers.
   - Trains iteratively over mini-batches, updating weights and bias with gradient descent
   - Predicts anomalies based on whether points lies within the decision boudary
  
   - Usage Example:
  
**Python**
```python
from chekml.OutlierIdentifier.robust_svm import RobustOneClassSVM
import numpy as np

X_train = np.random.randn(1000,8)
rsvm = RobustOneClassSVM(nu=0.1,gamma=0.1,batch_size=32,delta=1.0,rff_dim=100)
rsvm.fit(X_train,epochs=100,lr=0.01)
X_test = np.random.randn(100,8)
predictions=rsvm.predict(X_test)
print("Predictions (1=inlier, 0=outlier):",predictions[:5])
```
 ## 3.Outlier Evaluator
 - Description: `OutlierEvaluator` is a utility class that benchmarks multiple outlier detection algorithms, including `RobustOneClassSVM`, standard `OneClassSVM`, Isolation Forest, LOF, Elliptic Envelope, HBOS, KNN, and COF. It evaluates their accuracy and runtime on a test dataset with ground truth labels.
 - Parameters:
   - None
 - Algorithm Explanation:
   - Takes training and test data along with test labels (1 for inliers, -1 for outliers).
   - Fits each model on the training data and predicts on the test data.
   - Measures accuracy using `accuracy_score` and runtime for each model.
   - Returns a dictionary with model names, their accuracy scores, and execution times.
  
   - Usage Example:
   
**Python**
```python
from chekml.OutlierIdentifier.robust_svm import OutlierEvaluator
from sklearn.model_selection import train_test_split
import numpy as np

# Generate synthetic data
X = np.random.randn(10000, 8)
y = np.ones(10000)
X_outliers = np.random.uniform(low=-5, high=4, size=(50, 8))
y_outliers = -np.ones(50)
X = np.vstack([X, X_outliers])
y = np.hstack([y, y_outliers])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

evaluator = OutlierEvaluator()
results = evaluator.evaluate_models(X_train, X_test, y_test)
for model, score in results.items():
    print(f"{model}: {score:.4f}")
```
