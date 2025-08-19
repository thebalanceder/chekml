import numpy as np

class AnimalMigrationOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_fes=150000):
        """
        Initialize the Animal Migration Optimization (AMO) algorithm.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of animals (solutions).
        - max_fes: Maximum number of function evaluations.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_fes = max_fes

        self.population = None  # Population of animals (solutions)
        self.fitness = None  # Fitness values of the population
        self.global_best_solution = None
        self.global_best_fitness = float("inf")
        self.fes = 0  # Function evaluations counter
        self.history = []

    def initialize_population(self):
        """Initialize the population randomly within bounds."""
        lower_bounds = self.bounds[:, 0]
        upper_bounds = self.bounds[:, 1]
        self.population = lower_bounds + np.random.rand(self.population_size, self.dim) * (upper_bounds - lower_bounds)
        self.fitness = self.evaluate_population()
        self.fes = self.population_size
        self.update_global_best()

    def evaluate_population(self):
        """Compute fitness values for the population."""
        return np.array([self.objective_function(individual) for individual in self.population])

    def update_bounds(self):
        """Ensure population stays within bounds by reinitializing out-of-bound values."""
        lower_bounds = self.bounds[:, 0]
        upper_bounds = self.bounds[:, 1]
        for i in range(self.population_size):
            for j in range(self.dim):
                if self.population[i, j] < lower_bounds[j] or self.population[i, j] > upper_bounds[j]:
                    self.population[i, j] = lower_bounds[j] + np.random.rand() * (upper_bounds[j] - lower_bounds[j])

    def update_global_best(self):
        """Update the global best solution and fitness."""
        min_idx = np.argmin(self.fitness)
        if self.fitness[min_idx] < self.global_best_fitness:
            self.global_best_fitness = self.fitness[min_idx]
            self.global_best_solution = self.population[min_idx].copy()

    def get_indices(self):
        """Generate random indices for each individual, excluding itself."""
        r1 = np.zeros(self.population_size, dtype=int)
        r2 = np.zeros(self.population_size, dtype=int)
        r3 = np.zeros(self.population_size, dtype=int)
        r4 = np.zeros(self.population_size, dtype=int)
        r5 = np.zeros(self.population_size, dtype=int)

        for i in range(self.population_size):
            sequence = np.arange(self.population_size)
            sequence = np.delete(sequence, i)  # Remove current index

            temp = np.random.randint(0, len(sequence))
            r1[i] = sequence[temp]
            sequence = np.delete(sequence, temp)

            temp = np.random.randint(0, len(sequence))
            r2[i] = sequence[temp]
            sequence = np.delete(sequence, temp)

            temp = np.random.randint(0, len(sequence))
            r3[i] = sequence[temp]
            sequence = np.delete(sequence, temp)

            temp = np.random.randint(0, len(sequence))
            r4[i] = sequence[temp]
            sequence = np.delete(sequence, temp)

            temp = np.random.randint(0, len(sequence))
            r5[i] = sequence[temp]

        return r1, r2, r3, r4, r5

    def neighborhood_learning(self, i):
        """Select exemplar particle based on neighborhood."""
        if i == 0:
            lseq = [self.population_size - 2, self.population_size - 1, i, i + 1, i + 2]
        elif i == 1:
            lseq = [self.population_size - 1, i - 1, i, i + 1, i + 2]
        elif i == self.population_size - 2:
            lseq = [i - 2, i - 1, i, self.population_size - 1, 0]
        elif i == self.population_size - 1:
            lseq = [i - 2, i - 1, i, 0, 1]
        else:
            lseq = [i - 2, i - 1, i, i + 1, i + 2]

        j = np.random.permutation(5)
        return lseq[j[1]]  # Return the second element after permutation

    def optimize(self):
        """Run the Animal Migration Optimization algorithm."""
        self.initialize_population()
        iteration = 0

        while self.fes <= self.max_fes:
            iteration += 1
            self.update_bounds()

            # Neighborhood-based learning phase
            new_population = np.zeros_like(self.population)
            for i in range(self.population_size):
                FF = np.random.normal(0, 1)
                exemplar = np.zeros(self.dim, dtype=int)
                for d in range(self.dim):
                    exemplar[d] = self.neighborhood_learning(i)
                for d in range(self.dim):
                    new_population[i, d] = self.population[i, d] + FF * (self.population[exemplar[d], d] - self.population[i, d])

            new_population = self.update_bounds_new(new_population)
            new_fitness = np.array([self.objective_function(individual) for individual in new_population])
            self.fes += self.population_size

            for i in range(self.population_size):
                if new_fitness[i] <= self.fitness[i]:
                    self.population[i] = new_population[i]
                    self.fitness[i] = new_fitness[i]

            # Update probabilities based on fitness ranking
            sort_indices = np.argsort(self.fitness)
            probabilities = np.zeros(self.population_size)
            for i in range(self.population_size):
                probabilities[sort_indices[i]] = (self.population_size - i) / self.population_size

            # Global migration phase
            r1, r2, r3, r4, r5 = self.get_indices()
            new_population = np.zeros_like(self.population)
            for i in range(self.population_size):
                for j in range(self.dim):
                    if np.random.rand() > probabilities[i]:
                        new_population[i, j] = self.population[r1[i], j] + np.random.rand() * (self.global_best_solution[j] - self.population[i, j]) + np.random.rand() * (self.population[r3[i], j] - self.population[i, j])
                    else:
                        new_population[i, j] = self.population[i, j]

            new_population = self.update_bounds_new(new_population)
            new_fitness = np.array([self.objective_function(individual) for individual in new_population])
            self.fes += self.population_size

            for i in range(self.population_size):
                if new_fitness[i] <= self.fitness[i]:
                    self.population[i] = new_population[i]
                    self.fitness[i] = new_fitness[i]

            self.update_global_best()
            self.history.append((iteration, self.global_best_solution.copy(), self.global_best_fitness))
            print(f"Iteration {iteration}: Best Fitness = {self.global_best_fitness}")

        return self.global_best_solution, self.global_best_fitness, self.history

    def update_bounds_new(self, population):
        """Ensure new population stays within bounds."""
        lower_bounds = self.bounds[:, 0]
        upper_bounds = self.bounds[:, 1]
        for i in range(self.population_size):
            for j in range(self.dim):
                if population[i, j] < lower_bounds[j] or population[i, j] > upper_bounds[j]:
                    population[i, j] = lower_bounds[j] + np.random.rand() * (upper_bounds[j] - lower_bounds[j])
        return population
