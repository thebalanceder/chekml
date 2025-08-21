import numpy as np
import time
import matplotlib.pyplot as plt
from gwca_optimizer import GreatWallConstructionOptimizer  # Updated import

# Define challenging benchmark functions
def schwefel_function(x):
    return 418.9829 * len(x) - sum([xi * np.sin(np.sqrt(abs(xi))) for xi in x])

def levy_function(x):
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    term2 = sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    return term1 + term2 + term3

def michalewicz_function(x, m=10):
    return -sum(np.sin(x) * (np.sin(((np.arange(len(x)) + 1) * x**2) / np.pi))**(2 * m))

def shubert_function(x):
    return np.prod([
        sum(j * np.cos((j + 1) * xi + j) for j in range(1, 6))  
        for xi in x
    ])

# Define search space bounds
bounds = [(-5, 5), (-5, 5)]  # 2D case

# List of functions to test
benchmark_functions = {
    "Schwefel": schwefel_function,
    "Levy": levy_function,
    "Michalewicz": michalewicz_function,
    "Shubert": shubert_function
}

execution_times = {}
best_solutions = {}

# Start total timer
total_start_time = time.time()

# Run GWCA on each benchmark function
for name, func in benchmark_functions.items():
    print(f"\nOptimizing {name} function...")

    # Initialize optimizer
    gwca = GreatWallConstructionOptimizer(objective_function=func, dim=2, bounds=np.array(bounds), population_size=50, max_iter=100)

    # Measure execution time
    start_time = time.time()
    best_solution, best_value, history = gwca.optimize()
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times[name] = execution_time

    # Extract search history
    x_vals = [h[0] for h in history]  # Iterations
    y_vals = [h[1] for h in history]  # Best values

    best_solutions[name] = {"solution": best_solution, "fitness": best_value}

    # Create a grid for contour plot
    X = np.linspace(bounds[0][0], bounds[0][1], 100)
    Y = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.array([[func(np.array([x, y])) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

    # Plot contour map of the benchmark function
    plt.figure(figsize=(10, 6))
    plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(label="Objective Value")

    # Plot best solution
    plt.scatter(best_solution[0], best_solution[1], color="red", marker="*", s=200, edgecolors="black", linewidth=1.5, label="Best Solution")

    plt.annotate(f"({best_solution[0]:.2f}, {best_solution[1]:.2f})", 
                 (best_solution[0], best_solution[1]), 
                 textcoords="offset points", xytext=(10,10), ha='right', fontsize=10, color="white",
                 bbox=dict(facecolor='black', alpha=0.6, edgecolor='white'))

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(f"GWCA Optimization on {name} Function\nExecution Time: {execution_time:.4f} sec")
    plt.legend()
    plt.savefig(f"GWCA_{name}.png")

    print(f"Best solution found: {best_solution}, Objective Value: {best_value}")
    print(f"Execution Time: {execution_time:.4f} seconds")

# End total timer
total_end_time = time.time()
total_execution_time = total_end_time - total_start_time

# Display execution times and best solutions
print("\n=== Optimization Summary ===")
for name, data in best_solutions.items():
    print(f"{name} Function:")
    print(f"  Best Solution: {data['solution']}")
    print(f"  Best Fitness: {data['fitness']:.6f}")
    print(f"  Execution Time: {execution_times[name]:.4f} seconds\n")

# Print total execution time
print(f"Total Optimization Time: {total_execution_time:.4f} seconds")


