import numpy as np

class RedDeerAlgorithm:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, 
                 step_size=0.1, p_exploration=0.1):
        """
        Initialize the Red Deer Algorithm (RDA) optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of individuals in the population.
        - max_iter: Maximum number of iterations.
        - step_size: Step size for movement.
        - p_exploration: Probability of exploration.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.step_size = step_size
        self.p_exploration = p_exploration

        self.population = None  # Population of individuals (solutions)
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_population(self):
        """ Generate initial population randomly """
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                            (self.population_size, self.dim))

    def evaluate_population(self):
        """ Compute fitness values for the population """
        return np.array([self.objective_function(individual) for individual in self.population])

    def update_position(self, index):
        """ Update individual position based on exploration or exploitation """
        if np.random.rand() < self.p_exploration:
            # Exploration: Move randomly
            new_position = self.population[index] + self.step_size * (np.random.rand(self.dim) * 2 - 1)
        else:
            # Exploitation: Move towards the best solution
            direction = self.best_solution - self.population[index]
            new_position = self.population[index] + self.step_size * direction

        # Ensure the new position is within bounds
        return np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])

    def optimize(self):
        """ Run the Red Deer Algorithm optimization """
        self.initialize_population()
        for iteration in range(self.max_iter):
            # Evaluate fitness for each individual
            fitness = self.evaluate_population()
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.population[min_idx]
                self.best_value = fitness[min_idx]

            # Update each individual's position
            for i in range(self.population_size):
                self.population[i] = self.update_position(i)

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
