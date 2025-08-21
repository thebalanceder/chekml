import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fmin, fmax

# Define numpy types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class WildGeeseMigrationOptimizer:
    cdef public:
        object objective_function
        int dim
        np.ndarray bounds
        int num_geese
        int max_iter
        double alpha
        double beta
        double gamma
        np.ndarray geese
        np.ndarray best_solution
        double best_fitness
        list history

    def __init__(self, objective_function, int dim, bounds, int num_geese=20, int max_iter=100, 
                 double alpha=0.9, double beta=0.1, double gamma=0.1):
        """
        Initialize the Wild Geese Migration Optimization algorithm.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - num_geese: Number of geese (solutions).
        - max_iter: Maximum number of iterations.
        - alpha: Scaling factor.
        - beta: Learning rate.
        - gamma: Randomization parameter.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)
        self.num_geese = num_geese
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.best_fitness = float("inf")
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def initialize_geese(self):
        """ Randomly initialize goose positions """
        self.geese = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                       (self.num_geese, self.dim)).astype(DTYPE)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def evaluate_geese(self):
        """ Compute fitness values for all geese """
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = np.empty(self.num_geese, dtype=DTYPE)
        cdef int i
        for i in range(self.num_geese):
            fitness[i] = self.objective_function(self.geese[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def update_positions(self):
        """ Update positions of all geese based on the best goose """
        cdef np.ndarray[DTYPE_t, ndim=1] best_goose = self.geese[0]
        cdef np.ndarray[DTYPE_t, ndim=1] goose
        cdef np.ndarray[DTYPE_t, ndim=1] rand_beta = np.random.rand(self.dim).astype(DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] rand_gamma = np.random.rand(self.dim).astype(DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] bounds_diff = self.bounds[:, 1] - self.bounds[:, 0]
        cdef np.ndarray[DTYPE_t, ndim=1] lower_bounds = self.bounds[:, 0]
        cdef np.ndarray[DTYPE_t, ndim=1] upper_bounds = self.bounds[:, 1]
        cdef int i, j
        cdef double val

        for i in range(self.num_geese):
            goose = self.geese[i]
            for j in range(self.dim):
                goose[j] = (self.alpha * goose[j] + 
                           self.beta * rand_beta[j] * (best_goose[j] - goose[j]) + 
                           self.gamma * rand_gamma[j] * bounds_diff[j])
                # Clip to bounds
                goose[j] = fmin(fmax(goose[j], lower_bounds[j]), upper_bounds[j])
            self.geese[i] = goose

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """ Run the Wild Geese Migration Optimization algorithm """
        self.initialize_geese()
        cdef np.ndarray[DTYPE_t, ndim=1] fitness
        cdef np.ndarray[np.intp_t, ndim=1] sorted_indices
        cdef int iteration, min_idx
        cdef double min_fitness

        for iteration in range(self.max_iter):
            # Evaluate fitness
            fitness = self.evaluate_geese()
            
            # Sort geese based on fitness
            sorted_indices = np.argsort(fitness)
            fitness = fitness[sorted_indices]
            self.geese = self.geese[sorted_indices]
            
            # Update best solution
            min_fitness = fitness[0]
            if min_fitness < self.best_fitness:
                self.best_solution = self.geese[0].copy()
                self.best_fitness = min_fitness
            
            # Update goose positions
            self.update_positions()
            
            # Store history
            self.history.append((iteration, self.best_solution.copy(), self.best_fitness))
            print(f"Iteration {iteration + 1}: Best Fitness = {self.best_fitness}")
        
        print("\nOptimization finished.")
        print(f"Best solution found: {self.best_solution}")
        print(f"Best fitness: {self.best_fitness}")
        
        return self.best_solution, self.best_fitness, self.history
