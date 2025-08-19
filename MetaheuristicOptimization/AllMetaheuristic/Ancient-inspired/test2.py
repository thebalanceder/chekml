import numpy as np
import matplotlib.pyplot as plt
from dujiangyan_irrigation_optimizer import DujiangyanIrrigationOptimizer # Import your Rust-based module
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

# Run Dujiangyan Irrigation Optimizer on each benchmark function
for name, func in benchmark_functions.items():
    print(f"\nOptimizing {name} function...")

    # Initialize optimizer from dujiangyan_irrigation_optimizer
    diso = DujiangyanIrrigationOptimizer(
        objective_function=func,
        dim=2,
        bounds=bounds,
        population_size=20, # Adjust as needed
        max_iter=100, # Adjust as needed
        diversion_factor=0.5, # Adjust as needed
        flow_adjustment=0.5, # Adjust as needed
        water_density=1.0, # Adjust as needed
        fluid_distribution=1.0, # Adjust as needed
        centrifugal_resistance=1.0, # Adjust as needed
        bottleneck_ratio=0.5, # Adjust as needed
        elimination_ratio=0.5, # Adjust as needed
    )

    # Measure execution time
    start_time = time.time()
    best_solution, best_value = diso.optimize()
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times[name] = execution_time

    # Extract search history (if available)
    # Assuming diso.optimize() returns history, which isn't the case in your Rust code.
    # If no history is returned, the plotting logic below may not be fully applicable.
    # For now, you may need to adapt it based on how you want to visualize the search process.

    # Create a grid for contour plot
    X = np.linspace(bounds[0][0], bounds[0][1], 100)
    Y = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.array([[func([x, y]) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

    # Plot
    plt.figure(figsize=(10, 6))
    plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(label="Objective Value")

    # Assuming we can plot the search path if available
    # You would need to modify the Rust code to return the history for plotting
    # For now, I'm using a placeholder, adjust it accordingly.
    x_vals = [best_solution[0]] # Placeholder for x values
    y_vals = [best_solution[1]] # Placeholder for y values

    # Plot search path (This part assumes you get some path, replace if not)
    plt.plot(x_vals, y_vals, marker="o", markersize=3, color="cyan", linestyle="dashed", alpha=0.7, label="Search Path")
    plt.scatter(best_solution[0], best_solution[1], color="white", marker="*", s=100, label="Best Solution")

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(f"DISO on {name} Function")
    plt.legend()
    plt.savefig(f"DISO_{name}.png")

    print(f"Best solution found: {best_solution}, Objective Value: {best_value}")
    print(f"Execution Time: {execution_time:.4f} seconds")

# Display execution times
print("\nExecution Time Summary:")
for name, time_taken in execution_times.items():
    print(f"{name} Function: {time_taken:.4f} seconds")

