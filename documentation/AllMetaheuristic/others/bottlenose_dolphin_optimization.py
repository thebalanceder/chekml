import numpy as np

class BottlenoseDolphinOptimization:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, sonar_decay=0.99, leap_prob=0.2):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)
        self.population_size = population_size
        self.max_iter = max_iter
        self.sonar_decay = sonar_decay  # Factor to reduce sonar range over time
        self.leap_prob = leap_prob  # Probability of a random leap
        self.dolphins = None
        self.best_dolphin = None
        self.best_value = float("inf")
        self.sonar_intensity = 1.0  # Initial sonar intensity
        self.history = []

    def initialize_population(self):
        """ Generate a random initial population of dolphins """
        self.dolphins = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.population_size, self.dim))

    def evaluate_population(self):
        """ Compute fitness values for the dolphins """
        return np.array([self.objective_function(dolphin) for dolphin in self.dolphins])

    def update_best_dolphin(self):
        """ Identify the best dolphin in the pod """
        fitness = self.evaluate_population()
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.best_value:
            self.best_value = fitness[min_idx]
            self.best_dolphin = self.dolphins[min_idx].copy()
        return fitness

    def cooperative_hunting(self):
        """ Dolphins adjust positions using echolocation and cooperation """
        for i in range(self.population_size):
            if np.array_equal(self.dolphins[i], self.best_dolphin):
                continue  # Skip the best dolphin

            # Move dolphins towards the best one with reduced sonar intensity
            sonar_move = self.sonar_intensity * (self.best_dolphin - self.dolphins[i])
            self.dolphins[i] += sonar_move
            self.dolphins[i] = np.clip(self.dolphins[i], self.bounds[:, 0], self.bounds[:, 1])

    def random_leaps(self):
        """ Introduce random jumps to avoid local optima """
        for i in range(self.population_size):
            if np.random.rand() < self.leap_prob:
                leap_move = np.random.uniform(-1, 1, self.dim) * (self.bounds[:, 1] - self.bounds[:, 0])
                self.dolphins[i] += leap_move
                self.dolphins[i] = np.clip(self.dolphins[i], self.bounds[:, 0], self.bounds[:, 1])

    def optimize(self):
        """ Run the Bottlenose Dolphin Optimization Algorithm """
        self.initialize_population()

        for generation in range(self.max_iter):
            self.update_best_dolphin()
            self.cooperative_hunting()
            self.random_leaps()
            self.sonar_intensity *= self.sonar_decay  # Reduce sonar intensity over time

            # Save history for visualization
            self.history.append((generation, self.best_dolphin.copy(), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")

        return self.best_dolphin, self.best_value, self.history

