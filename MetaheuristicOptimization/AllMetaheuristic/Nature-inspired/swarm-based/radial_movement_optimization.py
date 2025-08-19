import numpy as np

class RadialMovementOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, alpha=0.1):
        """
        Initialize the Radial Movement Optimization (RMO) algorithm.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of solutions in the population.
        - max_iter: Maximum number of iterations.
        - alpha: Learning rate for position updates.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.alpha = alpha

        self.population = None  # Population of solutions
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_population(self):
        """ Generate initial population randomly within bounds """
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                            (self.population_size, self.dim))

    def evaluate_population(self):
        """ Compute fitness values for the population """
        return np.array([self.objective_function(individual) for individual in self.population])

    def update_reference_point(self):
        """ Calculate the mean of the population as the reference point """
        return np.mean(self.population, axis=0)

    def update_positions(self, reference_point):
        """ Update each individual's position towards the reference point """
        for i in range(self.population_size):
            direction = reference_point - self.population[i, :]
            self.population[i, :] = self.population[i, :] + self.alpha * direction
            # Ensure new position is within bounds
            self.population[i, :] = np.clip(self.population[i, :], self.bounds[:, 0], self.bounds[:, 1])

    def optimize(self):
        """ Run the Radial Movement Optimization algorithm """
        self.initialize_population()
        
        for iteration in range(self.max_iter):
            # Evaluate fitness for each individual
            fitness = self.evaluate_population()
            
            # Sort population based on fitness
            sorted_indices = np.argsort(fitness)
            fitness = fitness[sorted_indices]
            self.population = self.population[sorted_indices, :]
            
            # Update best solution if a better one is found
            if fitness[0] < self.best_value:
                self.best_solution = self.population[0, :].copy()
                self.best_value = fitness[0]
            
            # Update reference point
            reference_point = self.update_reference_point()
            
            # Update population positions
            self.update_positions(reference_point)
            
            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")
        
        return self.best_solution, self.best_value, self.history

# Example objective function (Sphere function)
def sphere_function(x):
    return np.sum(x ** 2)
