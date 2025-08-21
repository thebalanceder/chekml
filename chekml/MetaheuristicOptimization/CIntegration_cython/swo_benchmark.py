import numpy as np
import matplotlib.pyplot as plt
from swo_algorithm import SpiderWaspOptimizer  # Import the Cythonized module
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
bounds = [(-5, 5), (-5, 5)]  # 2D

# List of functions to test
benchmark_functions = {
    "Schwefel": schwefel_function,
    "Levy": levy_function,
    "Michalewicz": michalewicz_function,
    "Shubert": shubert_function
}

execution_times = {}

# Run SWO on each benchmark function
for name, func in benchmark_functions.items():
    print(f"\nOptimizing {name} function...")

    # Initialize optimizer
    swo = SpiderWaspOptimizer(objective_function=func, dim=2, bounds=bounds, population_size=100, max_iter=1000)

    # Measure execution time
    start_time = time.time()
    best_solution, best_value, convergence_curve = swo.optimize()
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times[name] = execution_time

    # Extract search history from positions over iterations
    # Since SWO doesn't store full history, we'll use the best solution and convergence curve
    x_vals = [best_solution[0]]  # Using best solution for simplicity
    y_vals = [best_solution[1]]

    # Create a grid for contour plot
    X = np.linspace(bounds[0][0], bounds[0][1], 100)
    Y = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.array([[func([x, y]) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

    # Plot
    plt.figure(figsize=(10, 6))
    plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(label="Objective Value")

    # Plot best solution (no full path available from SWO)
    plt.scatter(best_solution[0], best_solution[1], color="white", marker="*", s=100, label="Best Solution")

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(f"SWO on {name} Function")
    plt.legend()
    plt.savefig(f"SWO_{name}.png")

    # Plot convergence curve
    plt.figure(figsize=(10, 6))
    plt.semilogy(convergence_curve, '-<', markersize=8, linewidth=1.5, label="SWO")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness Obtained So Far")
    plt.title(f"Convergence Curve for {name} Function")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"SWO_{name}_convergence.png")

    print(f"Best solution found: {best_solution}, Objective Value: {best_value}")
    print(f"Execution Time: {execution_time:.4f} seconds")

# Display execution times
print("\nExecution Time Summary:")
for name, time_taken in execution_times.items():
    print(f"{name} Function: {time_taken:.4f} seconds")
