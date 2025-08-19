import numpy as np

class PhototropicOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, step_size=0.1):
        """
        Initialize the Phototropic Optimization Algorithm (POA) optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of individuals in the population.
        - max_iter: Maximum number of iterations.
        - step_size: Step size for movement towards the best solution.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.step_size = step_size

        self.population = None  # Population of solutions
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_population(self):
        """Generate initial population randomly within bounds"""
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                           (self.population_size, self.dim))

    def evaluate_population(self):
        """Compute fitness values for the population"""
        return np.array([self.objective_function(individual) for individual in self.population])

    def update_positions(self):
        """Update each individual's position towards the best solution"""
        for i in range(self.population_size):
            # Calculate direction towards the best solution
            direction = self.best_solution - self.population[i, :]
            
            # Normalize direction (handle zero norm case)
            norm_direction = np.linalg.norm(direction)
            if norm_direction != 0:
                direction = direction / norm_direction
            
            # Update position
            self.population[i, :] = self.population[i, :] + self.step_size * direction
            
            # Ensure the new position is within bounds
            self.population[i, :] = np.clip(self.population[i, :], self.bounds[:, 0], self.bounds[:, 1])

    def optimize(self):
        """Run the Phototropic Optimization Algorithm"""
        self.initialize_population()
        
        for iteration in range(self.max_iter):
            # Evaluate fitness for each individual
            fitness = self.evaluate_population()
            
            # Find the best individual in the population
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.population[min_idx].copy()
                self.best_value = fitness[min_idx]
            
            # Update positions towards the best solution
            self.update_positions()
            
            # Store history
            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")
        
        return self.best_solution, self.best_value, self.history

# Example objective function (Sphere function)
def sphere_function(x):
    """Sphere function for testing the optimizer"""
    return np.sum(x**2)
