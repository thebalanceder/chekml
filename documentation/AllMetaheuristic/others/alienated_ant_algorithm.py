import numpy as np

class AlienatedAntAlgorithm:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, alpha=0.7, beta=0.3):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)
        self.population_size = population_size
        self.max_iter = max_iter
        self.alpha = alpha  # Influence of pheromones
        self.beta = beta  # Alienation factor for independent movement
        self.ants = None
        self.best_ant = None
        self.best_value = float("inf")
        self.history = []

    def initialize_population(self):
        """ Generate a random initial population of ants """
        self.ants = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.population_size, self.dim))

    def evaluate_population(self):
        """ Compute fitness values for the ants """
        return np.array([self.objective_function(ant) for ant in self.ants])

    def update_best_ant(self):
        """ Identify the best ant in the colony """
        fitness = self.evaluate_population()
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.best_value:
            self.best_value = fitness[min_idx]
            self.best_ant = self.ants[min_idx].copy()
        return fitness

    def pheromone_following(self):
        """ Move ants toward the best trail using pheromones """
        for i in range(self.population_size):
            if np.array_equal(self.ants[i], self.best_ant):
                continue  # Skip the best ant

            # Move ants towards the best solution with some randomness
            pheromone_move = self.alpha * (self.best_ant - self.ants[i])
            self.ants[i] += pheromone_move
            self.ants[i] = np.clip(self.ants[i], self.bounds[:, 0], self.bounds[:, 1])

    def alienation_effect(self):
        """ Introduce alienation, where some ants move randomly """
        for i in range(self.population_size):
            if np.random.rand() < self.beta:
                alienation_move = np.random.uniform(-0.1, 0.1, self.dim) * (self.bounds[:, 1] - self.bounds[:, 0])
                self.ants[i] += alienation_move
                self.ants[i] = np.clip(self.ants[i], self.bounds[:, 0], self.bounds[:, 1])

    def optimize(self):
        """ Run the Alienated Ant Algorithm """
        self.initialize_population()

        for generation in range(self.max_iter):
            self.update_best_ant()
            self.pheromone_following()
            self.alienation_effect()

            # Save history for visualization
            self.history.append((generation, self.best_ant.copy(), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")

        return self.best_ant, self.best_value, self.history

