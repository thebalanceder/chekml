# cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython

# Type definitions for NumPy
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef np.intp_t INDEX_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class RadialMovementOptimizer:
    cdef public:
        object objective_function
        int dim
        np.ndarray bounds
        int population_size
        int max_iter
        double alpha
        np.ndarray population
        np.ndarray best_solution
        double best_value
        list history

    def __init__(self, objective_function, int dim, bounds, int population_size=50, int max_iter=100, double alpha=0.1):
        """
        Initialize the Radial Movement Optimization (RMO) algorithm.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of solutions in the population.
        - max_iter: Maximum number of iterations.
        - alpha: Learning rate for position updates.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)
        self.population_size = population_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.population = None
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void initialize_population(self):
        """ Generate initial population randomly within bounds """
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                            (self.population_size, self.dim))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[DTYPE_t, ndim=1] evaluate_population(self):
        """ Compute fitness values for the population """
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = np.zeros(self.population_size, dtype=DTYPE)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.population[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[DTYPE_t, ndim=1] update_reference_point(self):
        """ Calculate the mean of the population as the reference point """
        return np.mean(self.population, axis=0)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update_positions(self, np.ndarray[DTYPE_t, ndim=1] reference_point):
        """ Update each individual's position towards the reference point """
        cdef int i, j
        cdef np.ndarray[DTYPE_t, ndim=2] population = self.population
        cdef np.ndarray[DTYPE_t, ndim=1] direction
        cdef np.ndarray[DTYPE_t, ndim=1] lower_bounds = self.bounds[:, 0]
        cdef np.ndarray[DTYPE_t, ndim=1] upper_bounds = self.bounds[:, 1]
        
        for i in range(self.population_size):
            direction = reference_point - population[i, :]
            for j in range(self.dim):
                population[i, j] += self.alpha * direction[j]
                # Clip to bounds
                if population[i, j] < lower_bounds[j]:
                    population[i, j] = lower_bounds[j]
                elif population[i, j] > upper_bounds[j]:
                    population[i, j] = upper_bounds[j]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple optimize(self):
        """ Run the Radial Movement Optimization algorithm """
        self.initialize_population()
        
        cdef int iteration, i
        cdef np.ndarray[DTYPE_t, ndim=1] fitness
        cdef np.ndarray[INDEX_t, ndim=1] sorted_indices
        cdef np.ndarray[DTYPE_t, ndim=1] reference_point
        cdef double current_best_value
        
        for iteration in range(self.max_iter):
            # Evaluate fitness for each individual
            fitness = self.evaluate_population()
            
            # Sort population based on fitness
            sorted_indices = np.argsort(fitness).astype(np.intp)
            fitness = fitness[sorted_indices]
            self.population = self.population[sorted_indices, :]
            
            # Update best solution if a better one is found
            current_best_value = fitness[0]
            if current_best_value < self.best_value:
                self.best_solution = self.population[0, :].copy()
                self.best_value = current_best_value
            
            # Update reference point
            reference_point = self.update_reference_point()
            
            # Update population positions
            self.update_positions(reference_point)
            
            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")
        
        return self.best_solution, self.best_value, self.history

# Example objective function (Sphere function)
cpdef double sphere_function(np.ndarray[DTYPE_t, ndim=1] x):
    return np.sum(x ** 2)
