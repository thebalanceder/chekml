import numpy as np

class SocialEngineeringOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100):
        """
        Initialize the Social Engineering Optimizer (SEO).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of agents (solutions).
        - max_iter: Maximum number of iterations.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter

        self.population = None  # Population of agents (solutions)
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_population(self):
        """ Generate initial population randomly within bounds """
        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]
        self.population = lb + np.random.rand(self.population_size, self.dim) * (ub - lb)

    def evaluate_population(self):
        """ Compute fitness values for the population """
        return np.array([self.objective_function(agent) for agent in self.population])

    def social_engineering_update(self, index):
        """ Update an agent's solution based on a randomly selected target agent """
        # Select a random target agent (different from current agent)
        target_index = np.random.randint(self.population_size)
        while target_index == index:
            target_index = np.random.randint(self.population_size)

        # Update solution using social engineering formula
        new_solution = (self.population[index] +
                        np.random.randn(self.dim) *
                        (self.population[target_index] - self.population[index]))

        # Clip new solution to ensure it stays within bounds
        return np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])

    def optimize(self):
        """ Run the Social Engineering Optimizer """
        self.initialize_population()

        # Evaluate initial population
        fitness = self.evaluate_population()

        for iteration in range(self.max_iter):
            # Update each agent's position
            for i in range(self.population_size):
                new_solution = self.social_engineering_update(i)
                new_fitness = self.objective_function(new_solution)

                # Update if the new solution is better
                if new_fitness < fitness[i]:
                    self.population[i] = new_solution
                    fitness[i] = new_fitness

            # Find the best solution in the current population
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.population[min_idx]
                self.best_value = fitness[min_idx]

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
