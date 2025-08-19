import numpy as np
cimport numpy as cnp
cimport cython

# Ensure NumPy C API is initialized
cnp.import_array()

# Define types for performance
ctypedef cnp.double_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class LocustSwarmOptimizer:
    cdef object objective_function
    cdef int dim
    cdef cnp.ndarray bounds
    cdef int population_size
    cdef int max_iter
    cdef double step_size
    cdef cnp.ndarray population
    cdef cnp.ndarray best_solution
    cdef double best_value
    cdef list history

    def __init__(self, objective_function, int dim, bounds, int population_size=50, int max_iter=100, double step_size=0.1):
        """
        Initialize the Locust Swarm Optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of locusts (solutions).
        - max_iter: Maximum number of iterations.
        - step_size: Step size for movement towards the best solution.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.double)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.step_size = step_size
        self.best_value = float("inf")
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def initialize_population(self):
        """ Generate initial locust population randomly """
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                           (self.population_size, self.dim))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def evaluate_population(self):
        """ Compute fitness values for the locust population """
        cdef cnp.ndarray[DTYPE_t, ndim=1] fitness = np.zeros(self.population_size, dtype=np.double)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.population[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def update_positions(self):
        """ Update each locust's position towards the best solution """
        cdef int i, j
        cdef cnp.ndarray[DTYPE_t, ndim=1] direction
        cdef cnp.ndarray[DTYPE_t, ndim=1] rand_vec = np.random.rand(self.dim)
        for i in range(self.population_size):
            # Move each locust towards the best solution
            direction = self.best_solution - self.population[i]
            for j in range(self.dim):
                self.population[i, j] += self.step_size * direction[j] * rand_vec[j]
            
            # Ensure the new position is within bounds
            for j in range(self.dim):
                if self.population[i, j] < self.bounds[j, 0]:
                    self.population[i, j] = self.bounds[j, 0]
                elif self.population[i, j] > self.bounds[j, 1]:
                    self.population[i, j] = self.bounds[j, 1]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """ Run the Locust Swarm Algorithm Optimization """
        self.initialize_population()
        cdef cnp.ndarray[DTYPE_t, ndim=1] fitness
        cdef int iteration, min_idx
        for iteration in range(self.max_iter):
            # Evaluate fitness for each locust
            fitness = self.evaluate_population()
            
            # Find the best locust in the population
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.population[min_idx].copy()
                self.best_value = fitness[min_idx]
            
            # Update positions of all locusts
            self.update_positions()
            
            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")
        
        return self.best_solution, self.best_value, self.history

# Example objective function (Sphere function)
def sphere_function(x):
    return np.sum(x**2)
