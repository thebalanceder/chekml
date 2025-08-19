# cython: language_level=3
# distutils: language=c++

import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport fmin, fmax

# Define numpy types
np.import_array()
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)  # Disable bounds checking for performance
@cython.wraparound(False)   # Disable negative indexing for performance
cdef class RedDeerAlgorithm:
    cdef public object objective_function
    cdef public int dim
    cdef public np.ndarray bounds
    cdef public int population_size
    cdef public int max_iter
    cdef public double step_size
    cdef public double p_exploration
    cdef public np.ndarray population
    cdef public np.ndarray best_solution
    cdef public double best_value
    cdef public list history

    def __init__(self, objective_function, int dim, bounds, int population_size=50, 
                 int max_iter=100, double step_size=0.1, double p_exploration=0.1):
        """
        Initialize the Red Deer Algorithm (RDA) optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of individuals in the population.
        - max_iter: Maximum number of iterations.
        - step_size: Step size for movement.
        - p_exploration: Probability of exploration.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.step_size = step_size
        self.p_exploration = p_exploration

        self.population = None  # Population of individuals (solutions)
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_population(self):
        """ Generate initial population randomly """
        cdef double[:, :] bounds_view = self.bounds
        cdef double[:, :] pop_view
        self.population = np.random.uniform(bounds_view[:, 0], bounds_view[:, 1], 
                                            (self.population_size, self.dim))
        pop_view = self.population  # Memory view for faster access

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray evaluate_population(self):
        """ Compute fitness values for the population """
        cdef int i
        cdef double[:] fitness = np.zeros(self.population_size, dtype=DTYPE)
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.population[i])
        return np.asarray(fitness)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray update_position(self, int index):
        """ Update individual position based on exploration or exploitation """
        cdef double[:] new_position = np.zeros(self.dim, dtype=DTYPE)
        cdef double[:] current_position = self.population[index]
        cdef double[:] best_position = self.best_solution
        cdef double[:, :] bounds_view = self.bounds
        cdef int j
        cdef double r

        # Random number for exploration/exploitation decision
        r = rand() / <double>RAND_MAX
        if r < self.p_exploration:
            # Exploration: Move randomly
            for j in range(self.dim):
                new_position[j] = current_position[j] + self.step_size * ((rand() / <double>RAND_MAX) * 2 - 1)
        else:
            # Exploitation: Move towards the best solution
            for j in range(self.dim):
                new_position[j] = current_position[j] + self.step_size * (best_position[j] - current_position[j])

        # Ensure the new position is within bounds
        for j in range(self.dim):
            new_position[j] = fmax(fmin(new_position[j], bounds_view[j, 1]), bounds_view[j, 0])

        return np.asarray(new_position)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """ Run the Red Deer Algorithm optimization """
        self.initialize_population()
        cdef int iteration, i, min_idx
        cdef double[:] fitness
        cdef double best_fitness

        for iteration in range(self.max_iter):
            # Evaluate fitness for each individual
            fitness = self.evaluate_population()
            min_idx = np.argmin(fitness)
            best_fitness = fitness[min_idx]

            if best_fitness < self.best_value:
                self.best_solution = self.population[min_idx].copy()
                self.best_value = best_fitness

            # Update each individual's position
            for i in range(self.population_size):
                self.population[i] = self.update_position(i)

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
