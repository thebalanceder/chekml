import numpy as np
import matplotlib.pyplot as plt
from bacterial_colony_optimization import BacterialColonyOptimization

# Define Rastrigin function (2D)
def rastrigin_function(x):
    """Computes the Rastrigin function value."""
    x = np.array(x)  # Convert list to NumPy array
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# Define search space bounds
bounds = [(-5.12, 5.12), (-5.12, 5.12)]  # 2D space

# Run Bacterial Colony Optimization Algorithm for Rastrigin function
bco = BacterialColonyOptimization(num_bacteria=20, dim=2, bounds=bounds)
best_solution, best_value, history = bco.optimize(rastrigin_function)

# Extract search history
x_vals = [h[1][0] for h in history]
y_vals = [h[1][1] for h in history]

# Create a grid for contour plot
X = np.linspace(bounds[0][0], bounds[0][1], 100)
Y = np.linspace(bounds[1][0], bounds[1][1], 100)
X, Y = np.meshgrid(X, Y)
Z = np.array([[rastrigin_function([x, y]) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

# Plot
plt.figure(figsize=(10, 6))
plt.contourf(X, Y, Z, levels=50, cmap="viridis")
plt.colorbar(label="Objective Value")

# Plot search path
plt.plot(x_vals, y_vals, marker="o", markersize=3, color="red", linestyle="dashed", alpha=0.7, label="Search Path")
plt.scatter(best_solution[0], best_solution[1], color="white", marker="*", s=100, label="Best Solution")

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Bacterial Colony Optimization on Rastrigin Function")
plt.legend()
plt.savefig("BCO.png")

print(f"Best solution found: {best_solution}, Objective Value: {best_value}")

