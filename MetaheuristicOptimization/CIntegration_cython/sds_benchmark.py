import numpy as np
import matplotlib.pyplot as plt
from sds_algorithm import StochasticDiffusionSearch
import time

# Define challenging benchmark functions with decomposable components
def schwefel_function(x):
    # Decompose into components: each term in the sum
    components = [-xi * np.sin(np.sqrt(abs(xi))) for xi in x]
    return components  # Return list of component values

def levy_function(x):
    w = [1 + (xi - 1) / 4 for xi in x]
    term1 = np.sin(np.pi * w[0])**2
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    term2 = [(wi - 1)**2 * (1 + 10 * np.sin(np.pi * wi + 1)**2) for wi in w[:-1]]
    components = [term1] + term2 + [term3]
    return components  # Return list of component values

def michalewicz_function(x, m=10):
    # Decompose into components: each term in the sum
    components = [-np.sin(xi) * (np.sin((i + 1) * xi**2 / np.pi))**(2 * m) for i, xi in enumerate(x)]
    return components  # Return list of component values

def shubert_function(x):
    # Decompose into components: each product term
    components = [sum([j * np.cos((j + 1) * xi + j) for j in range(1, 6)]) for xi in x]
    return components  # Return list of component values

# Wrapper to compute full objective value for evaluation
def full_objective(components):
    return sum(components)

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

# Run SDS on each benchmark function
for name, func in benchmark_functions.items():
    print(f"\nOptimizing {name} function...")

    # Initialize optimizer
    sds = StochasticDiffusionSearch(
        objective_function=func,
        dim=2,
        bounds=bounds,
        population_size=1000,
        max_iter=100,
        mutation_rate=0.08,
        mutation_scale=4.0,
        cluster_threshold=0.33,
        context_sensitive=True
    )

    # Measure execution time
    start_time = time.time()
    best_solution, best_value, history = sds.optimize()
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times[name] = execution_time

    # Extract search history
    x_vals = [h[1][0] for h in history]
    y_vals = [h[1][1] for h in history]

    # Create a grid for contour plot
    X = np.linspace(bounds[0][0], bounds[0][1], 100)
    Y = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.array([[full_objective(func([x, y])) for x, y in zip(row_x, row_y)] 
                  for row_x, row_y in zip(X, Y)])

    # Plot
    plt.figure(figsize=(10, 6))
    plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(label="Objective Value")

    # Plot search path
    plt.plot(x_vals, y_vals, marker="o", markersize=3, color="cyan", linestyle="dashed", 
             alpha=0.7, label="Search Path")
    plt.scatter(best_solution[0], best_solution[1], color="white", marker="*", s=100, 
                label="Best Solution")

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(f"SDS on {name} Function")
    plt.legend()
    plt.savefig(f"SDS_{name}.png")

    print(f"Best solution found: {best_solution}, Objective Value: {best_value}")
    print(f"Execution Time: {execution_time:.4f} seconds")

# Display execution times
print("\nExecution Time Summary:")
for name, time_taken in execution_times.items():
    print(f"{name} Function: {time_taken:.4f} seconds")
