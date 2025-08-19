import numpy as np
import time
import matplotlib.pyplot as plt
from wrapper import Wrapper  # ✅ Using our Wrapper class

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

# ✅ Define search space bounds
bounds = [(-5, 5), (-5, 5)]  # 2D case

# ✅ List of benchmark functions
benchmark_functions = {
    "Schwefel": schwefel_function,
    "Levy": levy_function,
    "Michalewicz": michalewicz_function,
    "Shubert": shubert_function
}

execution_times = {}

# ✅ Run optimization on each benchmark function
for name, func in benchmark_functions.items():
    print(f"\n🔹 Optimizing {name} function...")

    # ✅ Initialize optimizer using our Wrapper
    optimizer = Wrapper(dim=2, population_size=3, max_iter=100, bounds=bounds, method="SOA")

    # ✅ Measure execution time
    start_time = time.time()
    optimizer.optimize(func)  # Run optimization
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times[name] = execution_time

    # ✅ Get best solution
    best_solution, best_value = optimizer.get_best_solution()

    # ✅ Create a grid for contour plot
    X = np.linspace(bounds[0][0], bounds[0][1], 100)
    Y = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.array([[func(np.array([x, y])) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

    # ✅ Plot contour map
    plt.figure(figsize=(10, 6))
    plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(label="Objective Value")

    # ✅ Plot best solution
    plt.scatter(best_solution[0], best_solution[1], color="white", marker="*", s=100, label="Best Solution")

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(f"Optimization on {name} Function\nExecution Time: {execution_time:.4f} sec")
    plt.legend()
    plt.savefig(f"{name}_Optimization.png")

    print(f"🏆 Best solution: {best_solution}, Objective Value: {best_value}")
    print(f"⏳ Execution Time: {execution_time:.4f} seconds")

    # ✅ Free memory used by optimizer
    optimizer.free()

# ✅ Display execution times summary
print("\n📊 Execution Time Summary:")
for name, time_taken in execution_times.items():
    print(f"{name} Function: {time_taken:.4f} seconds")
