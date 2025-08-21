import numpy as np
import matplotlib.pyplot as plt
from coa_algorithm import CoyoteOptimization
import time

# Define challenging benchmark functions
def schwefel_function(x):
    return 418.9829 * len(x) - sum([xi * np.sin(np.sqrt(abs(xi))) for xi in x])

def levy_function(x):
    w = [1 + (xi - 1) / 4 for xi in x]
    term1 = np.sin(np.pi * w[0])**2
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    term2 = sum([(wi - 1)**2 * (1 + 10 * np.sin(np.pi * wi + 1)**2) for wi in w[:-1]])
    return term1 + term2 + term3

def michalewicz_function(x, m=10):
    return -sum([np.sin(xi) * (np.sin((i + 1) * xi**2 / np.pi))**(2 * m) for i, xi in enumerate(x)])

def shubert_function(x):
    return np.prod([sum([j * np.cos((j + 1) * xi + j) for j in range(1, 6)]) for xi in x])

# Define search space bounds
dim = 2
bounds = np.array([[-5] * dim, [5] * dim])  # Shape: (2, dim)

# List of functions to test
benchmark_functions = {
    "Schwefel": schwefel_function,
    "Levy": levy_function,
    "Michalewicz": michalewicz_function,
    "Shubert": shubert_function
}

execution_times = {}

# Run COA on each benchmark function
for name, func in benchmark_functions.items():
    print(f"\nOptimizing {name} function...")

    # Initialize optimizer
    coa = CoyoteOptimization(objective_function=func, dim=2, bounds=bounds, n_packs=20, n_coy=5, max_nfeval=2000)

    # Measure execution time
    start_time = time.time()
    best_solution, best_value, history = coa.optimize()
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times[name] = execution_time

    # Extract search history
    x_vals = [h[1][0] for h in history]
    y_vals = [h[1][1] for h in history]

    # Create a grid for contour plot
    X = np.linspace(bounds[0, 0], bounds[1, 0], 100)
    Y = np.linspace(bounds[0, 1], bounds[1, 1], 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.array([[func([x, y]) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

    # Compute density of coyote positions using a 2D histogram
    hist, xedges, yedges = np.histogram2d(x_vals, y_vals, bins=50, range=[[bounds[0, 0], bounds[1, 0]], [bounds[0, 1], bounds[1, 1]]])
    hist = hist.T  # Transpose for correct orientation

    # Plot
    plt.figure(figsize=(10, 6))

    # Plot contour of objective function with vibrant colors
    contour = plt.contourf(X, Y, Z, levels=150, cmap="jet")  # Use jet colormap for high contrast
    plt.colorbar(contour, label="Objective Value")

    # Overlay density heatmap of coyote positions
    plt.imshow(hist, extent=[bounds[0, 0], bounds[1, 0], bounds[0, 1], bounds[1, 1]], origin='lower', cmap='YlOrRd', alpha=0.5)
    plt.colorbar(label="Solution Density")

    # Plot search path
    plt.plot(x_vals, y_vals, marker="o", markersize=3, color="cyan", linestyle="dashed", alpha=0.7, label="Search Path")
    plt.scatter(best_solution[0], best_solution[1], color="white", marker="*", s=100, label="Best Solution")

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(f"COA on {name} Function")
    plt.legend()
    plt.savefig(f"COA_{name}.png")

    print(f"Best solution found: {best_solution}, Objective Value: {best_value}")
    print(f"Execution Time: {execution_time:.4f} seconds")

# Display execution times
print("\nExecution Time Summary:")
for name, time_taken in execution_times.items():
    print(f"{name} Function: {time_taken:.4f} seconds")
