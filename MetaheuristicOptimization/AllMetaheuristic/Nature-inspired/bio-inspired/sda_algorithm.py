import numpy as np

class SlimeMouldAlgorithm:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, alpha=0.8, beta=0.2):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)
        self.population_size = population_size
        self.max_iter = max_iter
        self.alpha = alpha  # Oozing effect weight
        self.beta = beta    # Random oscillatory motion weight
        self.moulds = None
        self.best_mould = None
        self.best_value = float("inf")
        self.history = []

    def initialize_population(self):
        """ Generate a random initial population of slime moulds """
        self.moulds = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.population_size, self.dim))

    def evaluate_population(self):
        """ Compute fitness values for the slime moulds """
        return np.array([self.objective_function(mould) for mould in self.moulds])

    def update_best_mould(self):
        """ Find the best slime mould in the population """
        fitness = self.evaluate_population()
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.best_value:
            self.best_value = fitness[min_idx]
            self.best_mould = self.moulds[min_idx].copy()
        return fitness

    def move_moulds(self):
        """ Move slime moulds toward the best one using oozing effect and oscillations """
        for i in range(self.population_size):
            if np.array_equal(self.moulds[i], self.best_mould):
                continue  # Skip the best slime mould

            # Oozing effect: Attraction towards the best solution
            ooze_move = self.alpha * (self.best_mould - self.moulds[i])

            # Oscillatory random movement
            oscillate_move = self.beta * np.random.uniform(-1, 1, self.dim) * (self.bounds[:, 1] - self.bounds[:, 0])

            # Update slime mould position
            self.moulds[i] += ooze_move + oscillate_move
            self.moulds[i] = np.clip(self.moulds[i], self.bounds[:, 0], self.bounds[:, 1])

    def optimize(self):
        """ Run the Slime Mould Algorithm """
        self.initialize_population()

        for generation in range(self.max_iter):
            self.update_best_mould()
            self.move_moulds()

            # Save history for visualization
            self.history.append((generation, self.best_mould.copy(), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")

        return self.best_mould, self.best_value, self.history

