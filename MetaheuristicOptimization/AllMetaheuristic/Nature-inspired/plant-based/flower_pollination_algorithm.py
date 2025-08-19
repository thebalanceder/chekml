import numpy as np
from scipy.special import gamma

class FlowerPollinationAlgorithm:
    def __init__(self, objective_function, dim, bounds, population_size=20, max_iter=100, switch_prob=0.8):
        """
        Initialize the Flower Pollination Algorithm (FPA).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of flowers/solutions.
        - max_iter: Maximum number of iterations.
        - switch_prob: Probability of switching between global and local pollination.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.switch_prob = switch_prob

        self.flowers = None  # Population of flowers (solutions)
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_flowers(self):
        """ Generate initial population of flowers randomly within bounds """
        self.flowers = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                        (self.population_size, self.dim))
        self.fitness = np.array([self.objective_function(flower) for flower in self.flowers])
        min_idx = np.argmin(self.fitness)
        self.best_solution = self.flowers[min_idx].copy()
        self.best_value = self.fitness[min_idx]

    def levy_flight(self):
        """ Draw samples from a Levy distribution """
        beta = 3/2
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        step = u / np.abs(v) ** (1 / beta)
        return 0.01 * step

    def simple_bounds(self, solution):
        """ Apply lower and upper bounds to a solution """
        return np.clip(solution, self.bounds[:, 0], self.bounds[:, 1])

    def global_pollination(self, flower):
        """ Perform global pollination using Levy flight """
        step_size = self.levy_flight()
        delta = step_size * (flower - self.best_solution)
        new_solution = flower + delta
        return self.simple_bounds(new_solution)

    def local_pollination(self, flower, flower_indices):
        """ Perform local pollination using two random flowers """
        epsilon = np.random.rand()
        flower_j, flower_k = self.flowers[flower_indices[0]], self.flowers[flower_indices[1]]
        new_solution = flower + epsilon * (flower_j - flower_k)
        return self.simple_bounds(new_solution)

    def optimize(self):
        """ Run the Flower Pollination Algorithm """
        self.initialize_flowers()

        for iteration in range(self.max_iter):
            new_flowers = self.flowers.copy()
            for i in range(self.population_size):
                if np.random.rand() > self.switch_prob:
                    # Global pollination
                    new_flowers[i] = self.global_pollination(self.flowers[i])
                else:
                    # Local pollination
                    indices = np.random.permutation(self.population_size)[:2]
                    new_flowers[i] = self.local_pollination(self.flowers[i], indices)

                # Evaluate new solution
                new_fitness = self.objective_function(new_flowers[i])
                if new_fitness <= self.fitness[i]:
                    self.flowers[i] = new_flowers[i]
                    self.fitness[i] = new_fitness

                # Update global best
                if new_fitness <= self.best_value:
                    self.best_solution = new_flowers[i].copy()
                    self.best_value = new_fitness

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            
            # Display progress every 100 iterations
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Best Value = {self.best_value}")

        print(f"Total number of evaluations: {self.max_iter * self.population_size}")
        print(f"Best solution: {self.best_solution}")
        print(f"Best value: {self.best_value}")
        return self.best_solution, self.best_value, self.history

