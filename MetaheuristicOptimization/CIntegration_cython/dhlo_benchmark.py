import numpy as np
import matplotlib.pyplot as plt
from dhlo_algorithm import DynamicHuntingLeadershipOptimizer
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

# Modified DHL optimizer to track position history
class DynamicHuntingLeadershipOptimizerWithHistory(DynamicHuntingLeadershipOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.position_history = []  # Store best solution positions per iteration

    def optimize(self):
        """Run the Dynamic Hunting Leadership Optimization and track position history."""
        self.initialize_population()
        self.current_num_leaders = self.num_leaders
        self.leader_history = np.zeros((self.num_leaders, self.max_iter))

        for iter_idx in range(self.max_iter):
            # Update leaders
            leader_history = self.update_leaders(iter_idx)
            self.leader_history[:, iter_idx] = leader_history

            # Update number of leaders
            self.update_num_leaders(iter_idx, leader_history)

            # Update positions
            self.update_positions(iter_idx)

            # Update best solution and record position
            if self.leaders_fit[0] < self.best_value:
                self.best_value = self.leaders_fit[0]
                self.best_solution = self.leaders_pos[0].copy()
            self.position_history.append(self.best_solution.copy())  # Store clipped position

            # Record convergence
            self.convergence_curve.append(self.best_value)

            print(f"Iteration {iter_idx + 1}: Best Value = {self.best_value}, Num Leaders = {self.current_num_leaders}")

        return self.best_solution, self.best_value, self.convergence_curve, self.num_leaders_history, self.position_history

# Run DHL_V4 on each benchmark function
for name, func in benchmark_functions.items():
    print(f"\nOptimizing {name} function with DHL_V4...")

    # Initialize optimizer
    dhl = DynamicHuntingLeadershipOptimizerWithHistory(
        objective_function=func,
        dim=2,
        bounds=bounds,
        population_size=30,
        max_iter=200,
        num_leaders=30,
        variant='V4',
        tolerance=5
    )

    # Measure execution time
    start_time = time.time()
    best_solution, best_value, convergence_curve, num_leaders_history, position_history = dhl.optimize()
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times[name] = execution_time

    # Extract search history and ensure positions are within bounds
    x_vals = [pos[0] for pos in position_history]
    y_vals = [pos[1] for pos in position_history]
    # Clip positions to bounds for plotting
    x_vals = np.clip(x_vals, bounds[0][0], bounds[0][1])
    y_vals = np.clip(y_vals, bounds[1][0], bounds[1][1])

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
    plt.title(f"DHL_V4 on {name} Function")
    plt.legend()
    plt.savefig(f"DHL_V4_{name}.png")
    plt.close()

    print(f"Best solution found: {best_solution}, Objective Value: {best_value}")
    print(f"Execution Time: {execution_time:.4f} seconds")

# Display execution times
print("\nExecution Time Summary for DHL_V4:")
for name, time_taken in execution_times.items():
    print(f"{name} Function: {time_taken:.4f} seconds")
