import numpy as np
import matplotlib.pyplot as plt
from ica_algorithm import ImperialistCompetitiveAlgorithm
import time
import traceback

# Define benchmark functions
def schwefel_function(x):
    x = np.asarray(x)
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def levy_function(x):
    x = np.asarray(x)
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    return term1 + term2 + term3

def michalewicz_function(x, m=10):
    x = np.asarray(x)
    i = np.arange(1, len(x) + 1)
    return -np.sum(np.sin(x) * (np.sin(i * x**2 / np.pi))**(2 * m))

def shubert_function(x):
    x = np.asarray(x)
    sums = [np.sum([j * np.cos((j + 1) * xi + j) for j in range(1, 6)]) for xi in x]
    return np.prod(sums)

# Define search space bounds
bounds = [(-5, 5), (-5, 5)]  # 2D

# Test only Schwefel initially
benchmark_functions = {
    "Schwefel": schwefel_function,
     "Levy": levy_function,
     "Michalewicz": michalewicz_function,
     "Shubert": shubert_function
}

execution_times = {}

# Run ICA on each benchmark function
for name, func in benchmark_functions.items():
    print(f"\nOptimizing {name} function...")
    try:
        print("Initializing ICA...")
        ica = ImperialistCompetitiveAlgorithm(
            objective_function=func,
            dim=2,
            bounds=bounds,
            num_countries=100,
            num_initial_imperialists=8,
            max_decades=100
        )

        print("Starting optimization...")
        start_time = time.time()
        best_solution, best_cost, history = ica.optimize()
        end_time = time.time()
        execution_time = end_time - start_time
        execution_times[name] = execution_time

        print("Optimization completed. Preparing plot data...")
        x_vals = [h[1][0] for h in history]
        y_vals = [h[1][1] for h in history]

        print("Creating contour plot...")
        X = np.linspace(bounds[0][0], bounds[0][1], 100)
        Y = np.linspace(bounds[1][0], bounds[1][1], 100)
        X, Y = np.meshgrid(X, Y)
        Z = np.array([[func(np.array([x, y])) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

        print("Rendering plot...")
        plt.figure(figsize=(10, 6))
        plt.contourf(X, Y, Z, levels=50, cmap="viridis")
        plt.colorbar(label="Objective Value")

        plt.plot(x_vals, y_vals, marker="o", markersize=3, color="cyan", linestyle="dashed", alpha=0.7, label="Search Path")
        plt.scatter(best_solution[0], best_solution[1], color="white", marker="*", s=100, label="Best Solution")

        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title(f"ICA on {name} Function")
        plt.legend()
        print(f"Saving plot to ICA_{name}.png...")
        plt.savefig(f"ICA_{name}.png")
        plt.close()

        print(f"Best solution found: {best_solution}, Objective Value: {best_cost}")
        print(f"Execution Time: {execution_time:.4f} seconds")

    except Exception as e:
        print(f"Error during {name} optimization: {str(e)}")
        traceback.print_exc()

print("\nExecution Time Summary:")
for name, time_taken in execution_times.items():
    print(f"{name} Function: {time_taken:.4f} seconds")
