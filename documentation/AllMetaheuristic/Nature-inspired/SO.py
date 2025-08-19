import numpy as np

class SpiralOptimizationAlgorithm:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, c=0.1, alpha=1):
        self.obj_func = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)
        self.pop_size = population_size
        self.max_iter = max_iter
        self.c = c
        self.alpha = alpha  # Not used but kept for consistency
        self.history = []

    def optimize(self):
        # Initialize population
        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]
        population = lb + np.random.rand(self.pop_size, self.dim) * (ub - lb)
        fitness = np.apply_along_axis(self.obj_func, 1, population)

        # Identify best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for _ in range(self.max_iter):
            # Spiral movement
            r = np.random.rand(self.pop_size, 1)
            theta = 2 * np.pi * np.random.rand(self.pop_size, 1)
            direction = np.hstack((np.cos(theta), np.sin(theta)))

            step = self.c * r * direction
            new_position = population + step

            # Keep within bounds
            new_position = np.clip(new_position, lb, ub)

            # Evaluate new fitness
            new_fitness = np.apply_along_axis(self.obj_func, 1, new_position)

            # Update best solution
            min_idx = np.argmin(new_fitness)
            if new_fitness[min_idx] < best_fitness:
                best_fitness = new_fitness[min_idx]
                best_solution = new_position[min_idx].copy()

            # Sort and select top individuals
            sorted_indices = np.argsort(new_fitness)
            population = new_position[sorted_indices]

            # Save history (for plotting)
            self.history.append((best_fitness, best_solution.copy()))

        return best_solution, best_fitness, self.history
