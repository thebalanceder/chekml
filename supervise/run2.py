from FLR import FLR
#from FLR2 import FLR2
import time
import dask.array as da
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Use Dask to create large datasets (lazy evaluation)
number = 50_000_000
X = da.random.random((number, 10), chunks=(1_000_000, 10))  # Chunking prevents RAM overflow
y = 5 + X @ da.array([3, -1, 4, 0.5, 1, -2, 2, -0.3, 0.8, 1.5]) + da.random.normal(0, 0.1, size=number)

# Convert a manageable subset to NumPy (10M rows at a time)
X_np = X[:10_000_000].compute()
y_np = y[:10_000_000].compute()

# ✅ Train Custom Ultra-Optimized Linear Regression (FLR)
model = FLR()
start_time = time.time()
model.fit(X_np, y_np)
print(f"Custom Ultra-Optimized Linear Regression Training Time: {time.time() - start_time:.4f} sec")

# ✅ Train Sklearn Parallel Linear Regression (`n_jobs=-1`)
sklearn_model = LinearRegression(n_jobs=-1)  # Use all CPU cores
start_time = time.time()
sklearn_model.fit(X_np, y_np)
print(f"Sklearn Training Time: {time.time() - start_time:.4f} sec")

# ✅ Train FLR2
#model2 = FLR2()
#start_time = time.time()
#model2.fit(X_np, y_np)
#print(f"Custom Ultra-Optimized Linear Regression 2 Training Time: {time.time() - start_time:.4f} sec")

# ✅ Convert X[:50_000] to NumPy Before Prediction for Accuracy Check
X_test = X[:50_000].compute()  # Convert only 50,000 rows to NumPy for accuracy measurement
y_test = y[:50_000].compute()  # Get actual y values

# 🔥 Get Predictions
y_pred_custom = model.predict(X_test)
y_pred_sklearn = sklearn_model.predict(X_test)
#y_pred2 = model2.predict(X_test)

# ✅ Compute Accuracy Metrics
def evaluate_model(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 Accuracy for {name}:")
    print(f"  ✅ MSE: {mse:.6f}")
    print(f"  ✅ MAE: {mae:.6f}")
    print(f"  ✅ R² Score: {r2:.6f}")

# 🔍 Evaluate Each Model
evaluate_model("Custom FLR", y_test, y_pred_custom)
evaluate_model("Sklearn Linear Regression", y_test, y_pred_sklearn)
#evaluate_model("Custom FLR2", y_test, y_pred2)

# 🖨️ Print First 5 Predictions for Reference
print("\n🔍 First 5 Predictions:")
print(f"Custom FLR: {y_pred_custom[:5]}")
print(f"Sklearn: {y_pred_sklearn[:5]}")
#print(f"Custom FLR2: {y_pred2[:5]}")

