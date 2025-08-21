import numpy as np
import matplotlib.pyplot as plt
from tfwo_algorithm import TurbulentFlowWaterOptimizer
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

# Run TFWO on each benchmark function
for name, func in benchmark_functions.items():
    print(f"\nOptimizing {name} function...")

    # Initialize optimizer
    tfwo = TurbulentFlowWaterOptimizer(objective_function=func, dim=2, bounds=bounds)

    # Track search history
    x_vals = []
    y_vals = []
    best_costs = np.zeros(tfwo.max_iter)
    mean_costs = np.zeros(tfwo.max_iter)
    best_value = float('inf')
    best_solution = None

    # Measure execution time
    start_time = time.time()

    # Run optimization loop manually
    tfwo.initialize_whirlpools()
    for iter in range(tfwo.max_iter):
        tfwo.effects_of_whirlpools(iter)
        tfwo.pseudocode6()
        # Evaluate current best solution
        best_cost = float('inf')
        best_position = None
        # Sample positions to approximate whirlpool behavior
        for _ in range(tfwo.n_whirlpools):
            candidate_position = np.random.uniform(
                tfwo.bounds[:, 0], tfwo.bounds[:, 1], tfwo.dim
            )
            cost = tfwo.objective_function(candidate_position)
            if cost < best_cost:
                best_cost = cost
                best_position = candidate_position.copy()
        # Store history
        x_vals.append(best_position[0])
        y_vals.append(best_position[1])
        best_costs[iter] = best_cost
        # Approximate mean costs
        whirlpool_costs = np.array([tfwo.objective_function(
            np.random.uniform(tfwo.bounds[:, 0], tfwo.bounds[:, 1], tfwo.dim)
        ) for _ in range(tfwo.n_whirlpools)])
        mean_costs[iter] = np.mean(whirlpool_costs)
        # Update global best
        if best_cost < best_value:
            best_value = best_cost
            best_solution = best_position.copy()
        print(f"Iter {iter + 1}: Best Cost = {best_cost}")

    end_time = time.time()
    execution_time = end_time - start_time
    execution_times[name] = execution_time

    # Create a grid for contour plot
    X = np.linspace(bounds[0][0], bounds[0][1], 100)
    Y = np.linspace(bounds[1][0], bounds[1][1], 100)
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
    plt.title(f"TFWO on {name} Function")
    plt.legend()
    plt.savefig(f"TFWO_{name}.png")

    print(f"Best solution found: {best_solution}, Objective Value: {best_value}")
    print(f"Execution Time: {execution_time:.4f} seconds")

# Display execution times
print("\nExecution Time Summary:")
for name, time_taken in execution_times.items():
    print(f"{name} Function: {time_taken:.4f} seconds")
