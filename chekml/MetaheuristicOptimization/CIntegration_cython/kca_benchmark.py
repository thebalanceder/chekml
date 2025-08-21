import numpy as np
import matplotlib.pyplot as plt
import time
import kca_algorithm  # Import the Cythonized module

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

# Run KCA on each benchmark function
for name, func in benchmark_functions.items():
    print(f"\nOptimizing {name} function...")

    # Initialize optimizer
    def binary_fitness(key, _):
        # Map binary key to continuous space: each dimension uses 10 bits
        num_bits = 10
        x = []
        for i in range(2):  # 2D problem
            bits = key[i * num_bits:(i + 1) * num_bits]
            decimal = sum(bit * (2 ** j) for j, bit in enumerate(bits))
            scaled = -5 + (decimal / 1023) * 10  # Map [0, 1023] to [-5, 5]
            x.append(scaled)
        return func(x)

    kca = kca_algorithm.KeyCuttingAlgorithm(
        objective_function=binary_fitness,
        num_buses=2,  # Treat as 2 "buses" for 2D
        num_branches=18,  # 20 bits total (10 per dimension)
        population_size=20,
        max_iter=50,
        probability_threshold=0.5
    )

    # Measure execution time
    start_time = time.time()
    best_solution, best_fitness, history = kca.optimize(use_kca1=True)  # Use KCA1
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times[name] = execution_time

    # Extract search history (map binary to continuous for plotting)
    x_vals = []
    y_vals = []
    for _, key, _ in history:
        num_bits = 10
        x = []
        for i in range(2):
            bits = key[i * num_bits:(i + 1) * num_bits]
            decimal = sum(bit * (2 ** j) for j, bit in enumerate(bits))
            scaled = -5 + (decimal / 1023) * 10
            x.append(scaled)
        x_vals.append(x[0])
        y_vals.append(x[1])

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
    
    # Plot best solution
    best_x = []
    num_bits = 10
    for i in range(2):
        bits = best_solution[i * num_bits:(i + 1) * num_bits]
        decimal = sum(bit * (2 ** j) for j, bit in enumerate(bits))
        scaled = -5 + (decimal / 1023) * 10
        best_x.append(scaled)
    plt.scatter(best_x[0], best_x[1], color="white", marker="*", s=100, label="Best Solution")

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(f"KCA on {name} Function")
    plt.savefig(f"KCA_{name}.png")

    print(f"Best solution found: {best_x}, Objective Value: {best_fitness}")
    print(f"Execution Time: {execution_time:.4f} seconds")

# Display execution times
print("\nExecution Time Summary:")
for name, time_taken in execution_times.items():
    print(f"{name} Function: {time_taken:.4f} seconds")
