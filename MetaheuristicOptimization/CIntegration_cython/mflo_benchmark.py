import numpy as np
import matplotlib.pyplot as plt
from mflo_algorithm import MothFlameOptimizer
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
bounds = np.array([[-5, 5], [-5, 5]], dtype=np.float64)  # 2D, converted to NumPy array

# List of functions to test
benchmark_functions = {
    "Schwefel": schwefel_function,
    "Levy": levy_function,
    "Michalewicz": michalewicz_function,
    "Shubert": shubert_function
}

execution_times = {}

# Run MFO on each benchmark function
for name, func in benchmark_functions.items():
    print(f"\nOptimizing {name} function...")

    # Initialize optimizer
    mfo = MothFlameOptimizer(objective_function=func, dim=2, bounds=bounds, population_size=30, max_iter=100)

    # Measure execution time
    start_time = time.time()
    best_solution, best_value, convergence = mfo.optimize()
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times[name] = execution_time

    # Create a grid for contour plot
    X = np.linspace(bounds[0, 0], bounds[0, 1], 100)
    Y = np.linspace(bounds[1, 0], bounds[1, 1], 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.array([[func([x, y]) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

    # Plot
    plt.figure(figsize=(10, 6))
    plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(label="Objective Value")

    # Plot best solution
    plt.scatter(best_solution[0], best_solution[1], color="white", marker="*", s=100, label="Best Solution")

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(f"MFO on {name} Function")
    plt.legend()
    plt.savefig(f"MFO_{name}.png")
    plt.close()  # Close the figure to free memory

    print(f"Best solution found: {best_solution}, Objective Value: {best_value}")
    print(f"Execution Time: {execution_time:.4f} seconds")

# Display execution times
print("\nExecution Time Summary:")
for name, time_taken in execution_times.items():
    print(f"{name} Function: {time_taken:.4f} seconds")
