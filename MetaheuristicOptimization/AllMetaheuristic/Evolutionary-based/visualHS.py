import numpy as np
import matplotlib.pyplot as plt
from harmony_search import HarmonySearch

# Define Rastrigin function (2D)
def rastrigin_function(x):
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Define search space bounds
bounds = [(-5.12, 5.12), (-5.12, 5.12)]  # 2D

# Run Harmony Search
hs = HarmonySearch(objective_function=rastrigin_function, dim=2, bounds=bounds)
best_solution, best_value, history = hs.optimize()

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
plt.title("Harmony Search on Rastrigin Function")
plt.legend()
plt.savefig("HS.png")

print(f"Best solution found: {best_solution}, Objective Value: {best_value}")

