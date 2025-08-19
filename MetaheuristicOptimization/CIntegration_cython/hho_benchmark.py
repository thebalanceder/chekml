import numpy as np
import matplotlib.pyplot as plt
from hho_algorithm import harris_hawks_optimization
import time

# _____________________________________________________
# Main paper:
# Harris hawks optimization: Algorithm and applications
# Ali Asghar Heidari, Seyedali Mirjalili, Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, Huiling Chen
# Future Generation Computer Systems, 
# DOI: https://doi.org/10.1016/j.future.2019.02.028
# _____________________________________________________

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
lb = -5  # Lower bound
ub = 5   # Upper bound
dim = 2  # Dimensions
bounds = [(lb, ub)] * dim  # For compatibility with plotting

# List of functions to test
benchmark_functions = {
    "Schwefel": schwefel_function,
    "Levy": levy_function,
    "Michalewicz": michalewicz_function,
    "Shubert": shubert_function
}

execution_times = {}

# Run HHO on each benchmark function
for name, func in benchmark_functions.items():
    print(f"\nOptimizing {name} function...")

    # Parameters for HHO
    population_size = 30
    max_iterations = 100

    # Measure execution time
    start_time = time.time()
    best_solution, best_value, convergence = harris_hawks_optimization(
        population_size, max_iterations, lb, ub, dim, func
    )
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times[name] = execution_time

    # Simulate search history (HHO doesn't store full history, so we use convergence curve)
    # For visualization, we'll generate a pseudo-path by running HHO with fewer iterations
    pseudo_history = []
    temp_X = np.random.uniform(lb, ub, (population_size, dim))
    Rabbit_Location = np.zeros(dim)
    Rabbit_Energy = float('inf')
    
    for t in range(max_iterations):
        for i in range(population_size):
            temp_X[i, :] = np.clip(temp_X[i, :], lb, ub)
            fitness = func(temp_X[i, :])
            if fitness < Rabbit_Energy:
                Rabbit_Energy = fitness
                Rabbit_Location = temp_X[i, :].copy()
        pseudo_history.append((t, Rabbit_Location.copy(), Rabbit_Energy))
        
        E1 = 2 * (1 - (t / max_iterations))
        for i in range(population_size):
            E0 = 2 * np.random.rand() - 1
            Escaping_Energy = E1 * E0
            if abs(Escaping_Energy) >= 1:
                q = np.random.rand()
                rand_hawk_index = np.random.randint(0, population_size)
                X_rand = temp_X[rand_hawk_index, :]
                if q < 0.5:
                    temp_X[i, :] = X_rand - np.random.rand() * abs(X_rand - 2 * np.random.rand() * temp_X[i, :])
                else:
                    temp_X[i, :] = (Rabbit_Location - np.mean(temp_X, axis=0)) - \
                                   np.random.rand() * ((ub - lb) * np.random.rand() + lb)
            else:
                r = np.random.rand()
                if r >= 0.5 and abs(Escaping_Energy) < 0.5:
                    temp_X[i, :] = Rabbit_Location - Escaping_Energy * abs(Rabbit_Location - temp_X[i, :])
                elif r >= 0.5 and abs(Escaping_Energy) >= 0.5:
                    Jump_strength = 2 * (1 - np.random.rand())
                    temp_X[i, :] = (Rabbit_Location - temp_X[i, :]) - \
                                   Escaping_Energy * abs(Jump_strength * Rabbit_Location - temp_X[i, :])
                elif r < 0.5:
                    Jump_strength = 2 * (1 - np.random.rand())
                    X1 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - temp_X[i, :])
                    temp_X[i, :] = X1 if func(X1) < func(temp_X[i, :]) else temp_X[i, :]

    # Extract search history
    x_vals = [h[1][0] for h in pseudo_history]
    y_vals = [h[1][1] for h in pseudo_history]

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
    plt.title(f"HHO on {name} Function")
    plt.legend()
    plt.savefig(f"HHO_{name}.png")

    print(f"Best solution found: {best_solution}, Objective Value: {best_value}")
    print(f"Execution Time: {execution_time:.4f} seconds")

# Display execution times
print("\nExecution Time Summary:")
for name, time_taken in execution_times.items():
    print(f"{name} Function: {time_taken:.4f} seconds")
