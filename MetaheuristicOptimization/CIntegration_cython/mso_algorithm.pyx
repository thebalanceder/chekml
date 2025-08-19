# cython: language_level=3
# distutils: language = c++

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport exp

cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class MonkeySearchOptimizer:
    cdef object objective_function
    cdef int dim
    cdef double[:, :] bounds
    cdef int population_size
    cdef int max_iter
    cdef double p_explore
    cdef double max_p_explore
    cdef double min_p_explore
    cdef double[:, :] population
    cdef double[:] best_solution
    cdef double best_fitness
    cdef list history

    def __init__(self, object objective_function, int dim, bounds, int population_size=20, int max_iter=100, 
                 double p_explore=0.2, double max_p_explore=0.8, double min_p_explore=0.1):
        """
        Initialize the Monkey Search Algorithm (MSA) optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: List of (lower, upper) bounds for each dimension.
        - population_size: Number of monkey positions (solutions).
        - max_iter: Maximum number of iterations.
        - p_explore: Initial probability of exploration.
        - max_p_explore: Maximum probability of exploration.
        - min_p_explore: Minimum probability of exploration.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.double)
        self.population_size = population_size
        self.max_iter = max_iter
        self.p_explore = p_explore
        self.max_p_explore = max_p_explore
        self.min_p_explore = min_p_explore
        self.history = []

        self.population = None
        self.best_solution = None
        self.best_fitness = float("inf")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_population(self):
        """Generate initial monkey positions randomly"""
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                           (self.population_size, self.dim))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double[:] evaluate_population(self):
        """Compute fitness values for the monkey positions"""
        cdef double[:] fitness = np.zeros(self.population_size, dtype=np.double)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.population[i, :])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_position(self, int iter):
        """Update monkey positions based on exploration or exploitation"""
        cdef double[:, :] r = np.random.uniform(0, 1, (self.population_size, self.dim))
        cdef int i, j
        cdef double rand_val, perturbation

        # Update probability of exploration adaptively
        self.p_explore = max(self.max_p_explore * exp(-0.1 * iter), self.min_p_explore)
        
        for i in range(self.population_size):
            for j in range(self.dim):
                rand_val = np.random.uniform(0, 1)
                if rand_val < self.p_explore:  # Exploration
                    if r[i, j] < 0.5:
                        self.population[i, j] = self.best_solution[j] + np.random.uniform(0, 1) * (self.bounds[j, 1] - self.best_solution[j])
                    else:
                        self.population[i, j] = self.best_solution[j] - np.random.uniform(0, 1) * (self.best_solution[j] - self.bounds[j, 0])
                else:  # Exploitation
                    perturbation = np.random.normal(0, 1) * (self.bounds[j, 1] - self.bounds[j, 0]) / 10
                    self.population[i, j] = self.best_solution[j] + perturbation
                
                # Bound the positions
                if self.population[i, j] < self.bounds[j, 0]:
                    self.population[i, j] = self.bounds[j, 0]
                elif self.population[i, j] > self.bounds[j, 1]:
                    self.population[i, j] = self.bounds[j, 1]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple optimize(self):
        """Run the Monkey Search Algorithm Optimization"""
        self.initialize_population()
        
        # Evaluate initial population
        cdef double[:] fitness = self.evaluate_population()
        cdef int min_idx = np.argmin(fitness)
        self.best_fitness = fitness[min_idx]
        self.best_solution = self.population[min_idx, :].copy()
        
        cdef int iter
        for iter in range(self.max_iter):
            # Update positions
            self.update_position(iter)
            
            # Evaluate updated population
            fitness = self.evaluate_population()
            
            # Update best solution and fitness
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_fitness:
                self.best_fitness = fitness[min_idx]
                self.best_solution = self.population[min_idx, :].copy()
            
            self.history.append((iter, np.array(self.best_solution), self.best_fitness))
            print(f"Iteration {iter + 1}: Best Fitness = {self.best_fitness}")
        
        return np.array(self.best_solution), self.best_fitness, self.history

