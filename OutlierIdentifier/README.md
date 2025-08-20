 # Algorithm Overview and Usage

 ## 1.OutlierHybrid
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
X_test=np.random
``` 
