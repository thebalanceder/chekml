# cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class SocialEngineeringOptimizer:
    cdef public:
        object objective_function
        int dim
        np.ndarray bounds
        int population_size
        int max_iter
        np.ndarray population
        np.ndarray best_solution
        double best_value
        list history

    def __init__(self, object objective_function, int dim, bounds, int population_size=50, int max_iter=100):
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
        self.bounds = np.array(bounds, dtype=np.double)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.best_value = float("inf")
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_population(self):
        """ Generate initial population randomly within bounds """
        cdef np.ndarray[double, ndim=1] lb = self.bounds[:, 0]
        cdef np.ndarray[double, ndim=1] ub = self.bounds[:, 1]
        self.population = lb + np.random.rand(self.population_size, self.dim) * (ub - lb)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[double, ndim=1] evaluate_population(self):
        """ Compute fitness values for the population """
        cdef np.ndarray[double, ndim=1] fitness = np.zeros(self.population_size, dtype=np.double)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.population[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[double, ndim=1] social_engineering_update(self, int index):
        """ Update an agent's solution based on a randomly selected target agent """
        cdef int target_index = np.random.randint(self.population_size)
        while target_index == index:
            target_index = np.random.randint(self.population_size)

        # Update solution using social engineering formula
        cdef np.ndarray[double, ndim=1] new_solution = (
            self.population[index] +
            np.random.randn(self.dim) * (self.population[target_index] - self.population[index])
        )

        # Clip new solution to ensure it stays within bounds
        return np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """ Run the Social Engineering Optimizer """
        self.initialize_population()
        cdef np.ndarray[double, ndim=1] fitness = self.evaluate_population()
        cdef int iteration, i, min_idx
        cdef double new_fitness
        cdef np.ndarray[double, ndim=1] new_solution  # Explicitly declare type

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
                self.best_solution = self.population[min_idx].copy()
                self.best_value = fitness[min_idx]

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
