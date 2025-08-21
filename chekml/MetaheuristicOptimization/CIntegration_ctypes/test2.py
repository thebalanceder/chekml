import numpy as np
import time
import matplotlib.pyplot as plt
from wrapper import Wrapper

# ---- Define Benchmark Functions ----
def schwefel_function(position_ptr):
    x = np.ctypeslib.as_array(position_ptr, shape=(2,))
    return 418.9829 * len(x) - sum([xi * np.sin(np.sqrt(abs(xi))) for xi in x])

def levy_function(position_ptr):
    x = np.ctypeslib.as_array(position_ptr, shape=(2,))
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    term2 = sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    return term1 + term2 + term3

def michalewicz_function(position_ptr, m=10):
    x = np.ctypeslib.as_array(position_ptr, shape=(2,))
    return -sum(np.sin(x) * (np.sin(((np.arange(len(x)) + 1) * x**2) / np.pi))**(2 * m))

def shubert_function(position_ptr):
    x = np.ctypeslib.as_array(position_ptr, shape=(2,))
    return np.prod([
        sum(j * np.cos((j + 1) * xi + j) for j in range(1, 6))  
        for xi in x
    ])

# ---- Define Search Space ----
bounds = [(-5, 5), (-5, 5)]  

# ---- List of Benchmark Functions ----
benchmark_functions = {
    "Schwefel": schwefel_function,
    "Levy": levy_function,
    "Michalewicz": michalewicz_function,
    "Shubert": shubert_function
}

execution_times = {}

# ---- Run Optimization on Each Function ----
for name, func in benchmark_functions.items():
    print(f"\n🔄 Optimizing {name} function...")

    # Initialize optimizer
    wrapper = Wrapper(dim=2, population_size=50, max_iter=100, bounds=bounds, method="DISO")

    # Measure execution time
    start_time = time.time()
    history = wrapper.optimize(func)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times[name] = execution_time

    # Extract final best solution
    best_solution, best_value = wrapper.get_best_solution()

    print(f"✅ Final Best solution for {name}: {best_solution}, Objective Value: {best_value}")
    print(f"⏳ Execution Time: {execution_time:.4f} seconds")

    # Free optimizer memory
    wrapper.free()

    # ---- Generate Top-Down View ----
    print(f"📊 Generating top-down view of {name} function...")
    
    # Create meshgrid
    X = np.linspace(bounds[0][0], bounds[0][1], 100)
    Y = np.linspace(bounds[1][0], bounds[1][1], 100)
    XX, YY = np.meshgrid(X, Y)

    # Compute function values for contour plot
    ZZ = np.array([[func(np.array([x, y], dtype=np.float64)) for x in X] for y in Y])

    # Plot contour map
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(XX, YY, ZZ, levels=50, cmap="viridis")
    plt.colorbar(contour)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title(f"Top-Down View of {name} Function")

    # Plot best solution found
    plt.scatter(best_solution[0], best_solution[1], color="red", marker="x", s=100, label="Best Solution")
    plt.legend()

    # Save figure
    plt.savefig(f"{name.lower()}_topdown.png")
    print(f"📊 Saved plot: {name.lower()}_topdown.png")

    # Show plot (optional)
    # plt.show()
    
# ---- Execution Time Summary ----
print("\n📊 Execution Time Summary:")
for name, time_taken in execution_times.items():
    print(f"⏱️ {name} Function: {time_taken:.4f} seconds")

