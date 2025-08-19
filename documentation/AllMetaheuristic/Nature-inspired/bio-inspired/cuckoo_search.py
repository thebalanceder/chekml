import numpy as np
from scipy.special import gamma

class CuckooSearchOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=25, max_iter=100, pa=0.25):
        """
        Initialize the Cuckoo Search optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of nests (solutions), default is 25.
        - max_iter: Maximum number of iterations, default is 1000.
        - pa: Discovery rate of alien eggs/solutions, default is 0.25.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.pa = pa

        self.nests = None  # Population of nests (solutions)
        self.best_nest = None
        self.best_value = float("inf")
        self.history = []

    def initialize_nests(self):
        """Generate initial nests randomly."""
        self.nests = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                       (self.population_size, self.dim))

    def evaluate_nests(self):
        """Compute fitness values for the nests."""
        return np.array([self.objective_function(nest) for nest in self.nests])

    def get_cuckoos(self):
        """Generate new solutions using Levy flights."""
        n = self.population_size
        beta = 3/2
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2))) ** (1 / beta)

        new_nests = self.nests.copy()
        for j in range(n):
            s = self.nests[j, :]
            u = np.random.randn(self.dim) * sigma
            v = np.random.randn(self.dim)
            step = u / np.abs(v) ** (1 / beta)

            stepsize = 0.01 * step * (s - self.best_nest)
            s = s + stepsize * np.random.randn(self.dim)

            new_nests[j, :] = self.simplebounds(s)
        return new_nests

    def empty_nests(self):
        """Replace some nests with new solutions based on discovery probability."""
        n = self.population_size
        K = np.random.rand(n, self.dim) > self.pa

        idx = np.random.permutation(n)
        stepsize = np.random.rand() * (self.nests[idx, :] - self.nests[np.random.permutation(n), :])
        new_nests = self.nests + stepsize * K

        for j in range(n):
            new_nests[j, :] = self.simplebounds(new_nests[j, :])
        return new_nests

    def simplebounds(self, s):
        """Apply bounds to a solution."""
        s = np.maximum(s, self.bounds[:, 0])
        s = np.minimum(s, self.bounds[:, 1])
        return s

    def optimize(self):
        """Run the Cuckoo Search optimization."""
        self.initialize_nests()
        N_iter = 0

        for generation in range(self.max_iter):
            fitness = self.evaluate_nests()
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_nest = self.nests[min_idx].copy()
                self.best_value = fitness[min_idx]

            # Generate new solutions via Levy flights
            new_nests = self.get_cuckoos()
            new_fitness = np.array([self.objective_function(nest) for nest in new_nests])
            for j in range(self.population_size):
                if new_fitness[j] <= fitness[j]:
                    fitness[j] = new_fitness[j]
                    self.nests[j, :] = new_nests[j, :]
            N_iter += self.population_size

            # Discovery and randomization
            new_nests = self.empty_nests()
            new_fitness = np.array([self.objective_function(nest) for nest in new_nests])
            for j in range(self.population_size):
                if new_fitness[j] <= fitness[j]:
                    fitness[j] = new_fitness[j]
                    self.nests[j, :] = new_nests[j, :]
            N_iter += self.population_size

            # Update best if necessary
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_nest = self.nests[min_idx].copy()
                self.best_value = fitness[min_idx]

            self.history.append((generation, self.best_nest.copy(), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")

        print(f"Total number of iterations={N_iter}")
        return self.best_nest, self.best_value, self.history
