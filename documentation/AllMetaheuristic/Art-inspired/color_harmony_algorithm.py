import numpy as np

class ColorHarmonyAlgorithm:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, harmony_factor=1.5, mutation_rate=0.3):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.harmony_factor = harmony_factor  # Influence of the best harmonies
        self.mutation_rate = mutation_rate  # Introduces diversity in color combinations
        self.palette = None
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_palette(self):
        """ Generate an initial random set of color harmonies """
        self.palette = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.population_size, self.dim))

    def evaluate_palette(self):
        """ Compute fitness values (objective function) for the harmonies """
        return np.array([self.objective_function(harmony) for harmony in self.palette])

    def harmony_adjustment_phase(self):
        """ Harmonies adjust towards the best color composition """
        fitness = self.evaluate_palette()
        best_index = np.argmin(fitness)
        best_harmony = self.palette[best_index]

        for i in range(self.population_size):
            if i != best_index:  # The best harmony remains unchanged
                adjustment_step = self.harmony_factor * np.random.uniform(0, 1, self.dim) * (best_harmony - self.palette[i])
                self.palette[i] += adjustment_step
                self.palette[i] = np.clip(self.palette[i], self.bounds[:, 0], self.bounds[:, 1])  # Keep within bounds

    def mutation_phase(self):
        """ Introduce small random shifts to maintain diversity in color selection """
        for i in range(self.population_size):
            mutation_shift = self.mutation_rate * np.random.uniform(-1, 1, self.dim)
            self.palette[i] += mutation_shift
            self.palette[i] = np.clip(self.palette[i], self.bounds[:, 0], self.bounds[:, 1])

    def optimize(self):
        """ Run the Color Harmony Algorithm """
        self.initialize_palette()
        for generation in range(self.max_iter):
            self.harmony_adjustment_phase()
            self.mutation_phase()

            fitness = self.evaluate_palette()
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.palette[min_idx]
                self.best_value = fitness[min_idx]

            self.history.append((generation, self.best_solution.copy(), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

