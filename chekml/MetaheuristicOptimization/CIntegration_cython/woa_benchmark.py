import numpy as np
import matplotlib.pyplot as plt
from woa_algorithm import WhaleOptimizationAlgorithm
import time

# --- Benchmark functions ---
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

# --- Search space bounds ---
bounds = [(-5, 5), (-5, 5)]  # 2D problem

# --- Benchmark functions dictionary ---
benchmark_functions = {
    "Schwefel": schwefel_function,
    "Levy": levy_function,
    "Michalewicz": michalewicz_function,
    "Shubert": shubert_function
}

execution_times = {}

# --- Optimization loop ---
for name, func in benchmark_functions.items():
    print(f"\nOptimizing {name} function...")

    # Initialize WOA
    woa = WhaleOptimizationAlgorithm(objective_function=func, dim=2, bounds=bounds)

    # Measure execution time
    start_time = time.time()
    best_solution, best_value, search_history = woa.optimize()
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times[name] = execution_time

    # --- Contour plot preparation ---
    X = np.linspace(bounds[0][0], bounds[0][1], 100)
    Y = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.array([[func([x, y]) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

    plt.figure(figsize=(10, 6))
    plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(label="Objective Value")

    # --- Filtered search history plot (in-bounds only) ---
    for agent_idx in range(woa.population_size):
        x_vals, y_vals = [], []
        for t in range(len(search_history)):
            x, y = search_history[t][agent_idx]
            if bounds[0][0] <= x <= bounds[0][1] and bounds[1][0] <= y <= bounds[1][1]:
                x_vals.append(x)
                y_vals.append(y)
        if x_vals and y_vals:
            plt.plot(x_vals, y_vals, marker="o", markersize=3, color="cyan", linestyle="dashed",
                     alpha=0.3, label="Search Path" if agent_idx == 0 else "")

    # --- Plot best solution ---
    plt.scatter(best_solution[0], best_solution[1], color="white", marker="*", s=100, label="Best Solution")

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(f"WOA on {name} Function")
    plt.legend()
    plt.savefig(f"WOA_{name}_with_history.png")
    plt.close()

    # --- Logging results ---
    print(f"Best solution found: {best_solution}, Objective Value: {best_value}")
    print(f"Execution Time: {execution_time:.4f} seconds")

# --- Execution summary ---
print("\nExecution Time Summary:")
for name, time_taken in execution_times.items():
    print(f"{name} Function: {time_taken:.4f} seconds")

