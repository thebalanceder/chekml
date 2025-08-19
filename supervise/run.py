import numpy as np
import dask.array as da
from FLR import FLR

# Example usage with NumPy array
X_np = np.random.rand(1000, 10)
y_np = np.random.rand(1000)

# Create and fit the model
model = FLR()
model.fit(X_np, y_np)

# Predict using a NumPy array
predictions_np = model.predict(X_np)
print("Predictions (NumPy):", predictions_np)

# Example usage with Dask array
X_da = da.random.random((1000, 10), chunks=(100, 10))
y_da = da.random.random((1000,), chunks=(100,))

# Predict using a Dask array
predictions_da = model.predict(X_da)
print("Predictions (Dask):", predictions_da)  # No need to call .compute()
