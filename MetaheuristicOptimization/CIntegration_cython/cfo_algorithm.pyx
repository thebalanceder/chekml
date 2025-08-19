# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time

cdef class CentralForceOptimizer:
    cdef object objective_function
    cdef int dim
    cdef double[:] bounds
    cdef int population_size
    cdef int max_iter
    cdef double alpha
    cdef double[:, :] population
    cdef double[:] best_solution
    cdef double best_value
    cdef list history

    def __init__(self, object objective_function, int dim, tuple bounds, int population_size=50, int max_iter=100, double alpha=0.1):
        """
        Initialize the Central Force Optimization (CFO) algorithm.

        Parameters:
        - objective_function: Function to optimize (takes a 1D NumPy array, returns a scalar).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension (same for all dimensions).
        - population_size: Number of individuals in the population.
        - max_iter: Maximum number of iterations.
        - alpha: Learning rate for position updates.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.double)
        self.population_size = population_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.population = None
        self.best_solution = np.empty(dim, dtype=np.double)
        self.best_value = float("inf")
        self.history = []

    cdef void initialize_population(self):
        """Generate initial population randomly within bounds."""
        cdef double lower = self.bounds[0]
        cdef double upper = self.bounds[1]
        cdef int i, j
        cdef double[:, :] population = np.empty((self.population_size, self.dim), dtype=np.double)
        
        srand(<unsigned int>time(NULL))
        for i in range(self.population_size):
            for j in range(self.dim):
                population[i, j] = lower + (upper - lower) * (<double>rand() / <double>RAND_MAX)
        
        self.population = population

    cdef double[:] evaluate_population(self):
        """Compute fitness values for the population."""
        cdef double[:] fitness = np.empty(self.population_size, dtype=np.double)
        cdef int i
        
        for i in range(self.population_size):
            fitness[i] = self.objective_function(np.asarray(self.population[i]))
        
        return fitness

    cdef void update_positions(self):
        """Update population positions toward the center of mass."""
        cdef double[:] center_of_mass = np.empty(self.dim, dtype=np.double)
        cdef double[:, :] population = self.population
        cdef double alpha = self.alpha
        cdef double lower = self.bounds[0]
        cdef double upper = self.bounds[1]
        cdef int i, j
        cdef double direction
        
        # Compute center of mass
        for j in range(self.dim):
            center_of_mass[j] = 0.0
            for i in range(self.population_size):
                center_of_mass[j] += population[i, j]
            center_of_mass[j] /= self.population_size
        
        # Update positions and enforce bounds
        for i in range(self.population_size):
            for j in range(self.dim):
                direction = center_of_mass[j] - population[i, j]
                population[i, j] += alpha * direction
                if population[i, j] < lower:
                    population[i, j] = lower
                elif population[i, j] > upper:
                    population[i, j] = upper

    cpdef tuple optimize(self):
        """Run the Central Force Optimization algorithm."""
        self.initialize_population()
        cdef double[:] fitness
        cdef cnp.ndarray[cnp.intp_t, ndim=1] sorted_indices
        cdef double[:, :] sorted_population
        cdef int i, generation
        cdef double min_fitness
        
        for generation in range(self.max_iter):
            # Evaluate fitness
            fitness = self.evaluate_population()
            
            # Sort population by fitness
            sorted_indices = np.argsort(fitness)
            sorted_population = np.empty((self.population_size, self.dim), dtype=np.double)
            for i in range(self.population_size):
                sorted_population[i, :] = self.population[sorted_indices[i], :]
            self.population = sorted_population
            fitness = np.asarray(fitness)[sorted_indices]
            
            # Update best solution
            min_fitness = fitness[0]
            if min_fitness < self.best_value:
                self.best_value = min_fitness
                for i in range(self.dim):
                    self.best_solution[i] = self.population[0, i]
            
            # Update positions
            self.update_positions()
            
            # Store history
            self.history.append((generation, np.asarray(self.best_solution).copy(), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")
        
        return np.asarray(self.best_solution), self.best_value, self.history
