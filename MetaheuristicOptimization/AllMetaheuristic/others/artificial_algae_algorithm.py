import numpy as np

class ArtificialAlgaeAlgorithm:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, step_size=0.1):
        """
        Initialize the Artificial Algae Algorithm (AAA) optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of algae (solutions).
        - max_iter: Maximum number of iterations.
        - step_size: Step size for movement towards best solution.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.step_size = step_size

        self.population = None  # Population of algae (solutions)
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_population(self):
        """ Generate initial algae population randomly """
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                           (self.population_size, self.dim))

    def evaluate_population(self):
        """ Compute fitness values for the algae population """
        return np.array([self.objective_function(individual) for individual in self.population])

    def movement_phase(self):
        """ Move each algae towards the best solution """
        for i in range(self.population_size):
            # Calculate direction towards best solution
            direction = self.best_solution - self.population[i]
            # Update position
            self.population[i] = self.population[i] + self.step_size * direction
            # Ensure new position is within bounds
            self.population[i] = np.clip(self.population[i], self.bounds[:, 0], self.bounds[:, 1])

    def optimize(self):
        """ Run the Artificial Algae Algorithm """
        self.initialize_population()
        for iteration in range(self.max_iter):
            # Evaluate fitness for each individual
            fitness = self.evaluate_population()
            # Find the best individual
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.population[min_idx]
                self.best_value = fitness[min_idx]

            # Update population positions
            self.movement_phase()

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

