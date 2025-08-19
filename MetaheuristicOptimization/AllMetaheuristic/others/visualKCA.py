import numpy as np
import matplotlib.pyplot as plt
from key_cutting_algorithm import KeyCuttingAlgorithm

# Define Rastrigin function (2D)
def rastrigin_function(x):
    """Computes the Rastrigin function value."""
    x = np.array(x)  # Convert list to NumPy array
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# Define search space bounds (key cutting depths in millimeters)
bounds = [(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)]  # 4D space

# Run Key Cutting Optimization Algorithm with Rastrigin function
kco = KeyCuttingAlgorithm(num_keys=20, dim=4, bounds=bounds)
best_solution, best_value, history = kco.optimize(rastrigin_function)

# Extract search history
iterations = [h[0] for h in history]
values = [h[2] for h in history]

# Plot optimization history
plt.figure(figsize=(10, 6))
plt.plot(iterations, values, marker="o", linestyle="dashed", color="blue", alpha=0.7, label="Optimization Progress")

plt.xlabel("Iteration")
plt.ylabel("Best Rastrigin Function Value")
plt.title("Key Cutting Optimization on Rastrigin Function")
plt.legend()
plt.savefig("KCO.png")

print(f"Best solution found: {best_solution}, Objective Value: {best_value}")

