# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython

# Define types for numpy arrays
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef class CollidingBodiesOptimizer:
    cdef object objective_function
    cdef int dim
    cdef list bounds
    cdef int population_size
    cdef int max_iterations
    cdef double alpha
    cdef np.ndarray lower_limit
    cdef np.ndarray upper_limit
    cdef list history
    cdef np.ndarray global_best_solution
    cdef double global_best_fitness

    def __init__(self, object objective_function, int dim, list bounds, int population_size=50, int max_iterations=100, double alpha=0.1):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = bounds
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.lower_limit = np.array([b[0] for b in bounds], dtype=DTYPE)
        self.upper_limit = np.array([b[1] for b in bounds], dtype=DTYPE)
        self.history = []
        self.global_best_fitness = np.inf
        self.global_best_solution = np.zeros(dim, dtype=DTYPE)

    cpdef tuple optimize(self):
        # Declare typed variables
        cdef np.ndarray[DTYPE_t, ndim=2] population = self.lower_limit + (self.upper_limit - self.lower_limit) * np.random.rand(self.population_size, self.dim)
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = np.zeros(self.population_size, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] center_of_mass = np.zeros(self.dim, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] direction = np.zeros(self.dim, dtype=DTYPE)
        cdef np.ndarray[long, ndim=1] sorted_indices
        cdef int iter, i, j
        cdef double fitness_i

        # Main loop
        for iter in range(self.max_iterations):
            # Evaluate fitness
            for i in range(self.population_size):
                fitness_i = self.objective_function(population[i, :])
                fitness[i] = fitness_i
                
                # Update global best
                if fitness_i < self.global_best_fitness:
                    self.global_best_fitness = fitness_i
                    for j in range(self.dim):
                        self.global_best_solution[j] = population[i, j]
            
            # Sort population based on fitness
            sorted_indices = np.argsort(fitness)
            fitness = fitness[sorted_indices]
            population = population[sorted_indices, :]
            
            # Update positions
            for i in range(self.population_size):
                # Calculate center of mass
                for j in range(self.dim):
                    center_of_mass[j] = 0
                    for k in range(self.population_size):
                        center_of_mass[j] += population[k, j]
                    center_of_mass[j] /= self.population_size
                
                # Move towards center of mass
                for j in range(self.dim):
                    direction[j] = center_of_mass[j] - population[i, j]
                    population[i, j] += self.alpha * direction[j]
                    
                    # Ensure bounds
                    if population[i, j] > self.upper_limit[j]:
                        population[i, j] = self.upper_limit[j]
                    elif population[i, j] < self.lower_limit[j]:
                        population[i, j] = self.lower_limit[j]
                
                # Store history for visualization
                self.history.append((iter, population[i, :].copy(), fitness[i]))
        
        return self.global_best_solution, self.global_best_fitness, self.history
