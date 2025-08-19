import numpy as np
import matplotlib.pyplot as plt
from bso_algorithm import buffalo_swarm_optimization
import time

# Define challenging benchmark functions
def schwefel_function(x):
    return  418.9829 * len(x) - sum([xi * np.sin(np.sqrt(abs(xi))) for xi in x])

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
lower_bound = -5
upper_bound = 5
num_variables = 2  # 2D

# List of functions to test
benchmark_functions = {
    "Schwefel": schwefel_function,
    "Levy": levy_function,
    "Michalewicz": michalewicz_function,
    "Shubert": shubert_function
}

execution_times = {}

# Run BSO on each benchmark function
for name, func in benchmark_functions.items():
    print(f"\nOptimizing {name} function...")

    # Measure execution time
    start_time = time.time()
    best_solution, best_fitness, history = buffalo_swarm_optimization(
        objective_function=func,
        num_buffaloes=50,
        num_iterations=100,
        num_variables=num_variables,
        lower_bound=lower_bound,
        upper_bound=upper_bound
    )
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times[name] = execution_time

    # Extract search path from history
    x_vals = [h[1][0] for h in history]
    y_vals = [h[1][1] for h in history]

    # Create a grid for contour plot
    X = np.linspace(lower_bound, upper_bound, 100)
    Y = np.linspace(lower_bound, upper_bound, 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.array([[func([x, y]) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

    # Plot
    plt.figure(figsize=(10, 6))
    plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(label="Objective Value")

    # Plot search path
    plt.plot(x_vals, y_vals, marker="o", markersize=3, color="cyan", linestyle="dashed", alpha=0.7, label="Search Path")
    plt.scatter(best_solution[0], best_solution[1], color="white", marker="*", s=100, label="Best Solution")

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(f"BSO on {name} Function")
    plt.legend()
    plt.savefig(f"BSO_{name}.png")

    print(f"Best solution found: {best_solution}, Objective Value: {best_fitness}")
    print(f"Execution Time: {execution_time:.4f} seconds")

# Display execution times
print("\nExecution Time Summary:")
for name, time_taken in execution_times.items():
    print(f"{name} Function: {time_taken:.4f} seconds")
