import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt

# Ensure NumPy C API is initialized
cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double sphere_function(double[:] x):
    """Sphere function for testing the optimizer"""
    cdef int i
    cdef double result = 0.0
    for i in range(x.shape[0]):
        result += x[i] * x[i]
    return result

cdef class PhototropicOptimizer:
    cdef object objective_function
    cdef int dim
    cdef double[:, :] bounds
    cdef int population_size
    cdef int max_iter
    cdef double step_size
    cdef double[:, :] population
    cdef double[:] best_solution
    cdef double best_value
    cdef list history

    def __init__(self, objective_function, int dim, bounds, int population_size=50, int max_iter=100, double step_size=0.1):
        """
        Initialize the Phototropic Optimization Algorithm (POA) optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of individuals in the population.
        - max_iter: Maximum number of iterations.
        - step_size: Step size for movement towards the best solution.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.float64)
        self.population_size = population_size
        self.max_iter = max_iter
        self.step_size = step_size
        self.best_value = float("inf")
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_population(self):
        """Generate initial population randomly within bounds"""
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                           (self.population_size, self.dim)).astype(np.float64)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double[:] evaluate_population(self):
        """Compute fitness values for the population"""
        cdef double[:] fitness = np.zeros(self.population_size, dtype=np.float64)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.population[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_positions(self):
        """Update each individual's position towards the best solution"""
        cdef int i, j
        cdef double[:] direction = np.zeros(self.dim, dtype=np.float64)
        cdef double norm_direction
        for i in range(self.population_size):
            # Calculate direction towards the best solution
            norm_direction = 0.0
            for j in range(self.dim):
                direction[j] = self.best_solution[j] - self.population[i, j]
                norm_direction += direction[j] * direction[j]
            norm_direction = sqrt(norm_direction)
            
            # Normalize direction (handle zero norm case)
            if norm_direction != 0:
                for j in range(self.dim):
                    direction[j] /= norm_direction
            
            # Update position
            for j in range(self.dim):
                self.population[i, j] += self.step_size * direction[j]
                
                # Ensure the new position is within bounds
                if self.population[i, j] < self.bounds[j, 0]:
                    self.population[i, j] = self.bounds[j, 0]
                elif self.population[i, j] > self.bounds[j, 1]:
                    self.population[i, j] = self.bounds[j, 1]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple optimize(self):
        """Run the Phototropic Optimization Algorithm"""
        self.initialize_population()
        cdef double[:] fitness
        cdef int min_idx, iteration, i
        cdef double[:, :] best_solution_copy
        
        for iteration in range(self.max_iter):
            # Evaluate fitness for each individual
            fitness = self.evaluate_population()
            
            # Find the best individual in the population
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.population[min_idx].copy()
                self.best_value = fitness[min_idx]
            
            # Update positions towards the best solution
            self.update_positions()
            
            # Store history
            best_solution_copy = np.array(self.best_solution, copy=True).reshape(1, -1)
            self.history.append((iteration, best_solution_copy[0], self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")
        
        return np.array(self.best_solution), self.best_value, self.history
