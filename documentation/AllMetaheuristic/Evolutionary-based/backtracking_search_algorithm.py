import numpy as np

class BacktrackingSearchAlgorithm:
    def __init__(self, objective_function, dim, bounds, pop_size=50, dim_rate=0.5, max_iter=100):
        """
        Initialize BSA optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of variables (dimensions).
        - bounds: Tuple of (lower_bounds, upper_bounds).
        - pop_size: Population size.
        - dim_rate: Rate of dimensions to update.
        - max_iter: Maximum number of generations.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)
        self.pop_size = pop_size
        self.dim_rate = dim_rate
        self.max_iter = max_iter

        self.population = None
        self.fitness = None
        self.historical_pop = None
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def generate_population(self):
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        return np.random.uniform(low, high, (self.pop_size, self.dim))

    def boundary_control(self, pop):
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        for i in range(self.pop_size):
            for j in range(self.dim):
                k = np.random.rand() < np.random.rand()
                if pop[i, j] < low[j]:
                    pop[i, j] = low[j] if k else np.random.uniform(low[j], high[j])
                if pop[i, j] > high[j]:
                    pop[i, j] = high[j] if k else np.random.uniform(low[j], high[j])
        return pop

    def get_scale_factor(self):
        return 3 * np.random.randn()  # Standard Brownian walk

    def optimize(self):
        self.population = self.generate_population()
        self.fitness = np.array([self.objective_function(ind) for ind in self.population])
        self.historical_pop = self.generate_population()

        for epoch in range(self.max_iter):
            if np.random.rand() < np.random.rand():
                self.historical_pop = self.population.copy()

            np.random.shuffle(self.historical_pop)
            F = self.get_scale_factor()

            # Generate map
            map_mask = np.zeros((self.pop_size, self.dim))
            if np.random.rand() < np.random.rand():
                for i in range(self.pop_size):
                    u = np.random.permutation(self.dim)
                    map_mask[i, u[:int(np.ceil(self.dim_rate * np.random.rand() * self.dim))]] = 1
            else:
                for i in range(self.pop_size):
                    map_mask[i, np.random.randint(0, self.dim)] = 1

            # Recombination
            offspring = self.population + (map_mask * F) * (self.historical_pop - self.population)
            offspring = self.boundary_control(offspring)
            fitness_offspring = np.array([self.objective_function(ind) for ind in offspring])

            # Selection
            improved = fitness_offspring < self.fitness
            self.fitness[improved] = fitness_offspring[improved]
            self.population[improved] = offspring[improved]

            # Store best solution
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_value:
                self.best_value = self.fitness[best_idx]
                self.best_solution = self.population[best_idx].copy()

            self.history.append((epoch, self.best_solution.copy(), self.best_value))
            print(f"Iteration {epoch + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
