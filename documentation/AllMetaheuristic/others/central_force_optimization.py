import numpy as np

class CentralForceOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, alpha=0.1):
        """
        Initialize the Central Force Optimization (CFO) algorithm.

        Parameters:
        - objective_function: Function to optimize (takes a 1D NumPy array, returns a scalar).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension (same for all dimensions).
        - population_size: Number of individuals in the population.
        - max_iter: Maximum number of iterations.
        - alpha: Learning rate for position updates.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [lower, upper]
        self.population_size = population_size
        self.max_iter = max_iter
        self.alpha = alpha

        self.population = None  # Population of individuals (solutions)
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_population(self):
        """Generate initial population randomly within bounds."""
        lower, upper = self.bounds
        self.population = lower + (upper - lower) * np.random.rand(self.population_size, self.dim)

    def evaluate_population(self):
        """Compute fitness values for the population."""
        return np.array([self.objective_function(individual) for individual in self.population])

    def update_positions(self):
        """Update population positions toward the center of mass."""
        # Compute center of mass (mean of population)
        center_of_mass = np.mean(self.population, axis=0)

        # Update each individual's position
        for i in range(self.population_size):
            direction = center_of_mass - self.population[i, :]
            self.population[i, :] += self.alpha * direction

            # Enforce bounds
            self.population[i, :] = np.clip(self.population[i, :], self.bounds[0], self.bounds[1])

    def optimize(self):
        """Run the Central Force Optimization algorithm."""
        self.initialize_population()
        for generation in range(self.max_iter):
            # Evaluate fitness
            fitness = self.evaluate_population()

            # Sort population by fitness
            sorted_indices = np.argsort(fitness)
            fitness = fitness[sorted_indices]
            self.population = self.population[sorted_indices]

            # Update best solution
            if fitness[0] < self.best_value:
                self.best_solution = self.population[0].copy()
                self.best_value = fitness[0]

            # Update positions
            self.update_positions()

            self.history.append((generation, self.best_solution.copy(), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
