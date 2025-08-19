# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time
from libc.math cimport sqrt, log, cos

cdef class CoralReefsOptimizer:
    cdef object objective_function
    cdef int dim
    cdef double[:] bounds
    cdef int population_size
    cdef int max_iter
    cdef int num_reefs
    cdef double alpha
    cdef list reefs
    cdef double[:] best_solution
    cdef double best_value
    cdef list history

    def __init__(self, object objective_function, int dim, tuple bounds, int population_size=50, 
                 int max_iter=100, int num_reefs=10, double alpha=0.1):
        """
        Initialize the Coral Reefs Optimization (CRO) algorithm.

        Parameters:
        - objective_function: Function to optimize (takes a 1D NumPy array, returns a scalar).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension (same for all dimensions).
        - population_size: Number of solutions per reef.
        - max_iter: Maximum number of iterations.
        - num_reefs: Number of reefs (subpopulations).
        - alpha: Scaling factor for local search perturbation.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.double)
        self.population_size = population_size
        self.max_iter = max_iter
        self.num_reefs = num_reefs
        self.alpha = alpha
        self.reefs = None
        self.best_solution = np.empty(dim, dtype=np.double)
        self.best_value = float("inf")
        self.history = []

    cdef void initialize_reefs(self):
        """Generate initial reef populations randomly within bounds."""
        cdef double lower = self.bounds[0]
        cdef double upper = self.bounds[1]
        cdef int i, j, k
        cdef double[:, :] reef
        self.reefs = []

        srand(<unsigned int>time(NULL))
        for i in range(self.num_reefs):
            reef = np.empty((self.population_size, self.dim), dtype=np.double)
            for j in range(self.population_size):
                for k in range(self.dim):
                    reef[j, k] = lower + (upper - lower) * (<double>rand() / <double>RAND_MAX)
            self.reefs.append(np.asarray(reef))

    cdef double[:, :] evaluate_reefs(self):
        """Compute fitness values for all solutions in each reef."""
        cdef double[:, :] fitness = np.empty((self.num_reefs, self.population_size), dtype=np.double)
        cdef int i, j
        cdef double[:, :] reef

        for i in range(self.num_reefs):
            reef = self.reefs[i]
            for j in range(self.population_size):
                fitness[i, j] = self.objective_function(np.asarray(reef[j]))
        return fitness

    cdef void migration_phase(self):
        """Exchange solutions between reefs to simulate migration."""
        cdef int i, j, idx, idx_replace
        cdef double[:, :] reef_i, reef_j
        cdef double[:] solution_to_migrate

        srand(<unsigned int>time(NULL))
        for i in range(self.num_reefs):
            reef_i = self.reefs[i]
            for j in range(self.num_reefs):
                if i != j:
                    reef_j = self.reefs[j]
                    idx = rand() % self.population_size
                    solution_to_migrate = np.asarray(reef_i[idx]).copy()
                    idx_replace = rand() % self.population_size
                    for k in range(self.dim):
                        reef_j[idx_replace, k] = solution_to_migrate[k]

    cdef void local_search_phase(self):
        """Perform local search by perturbing solutions in each reef."""
        cdef double lower = self.bounds[0]
        cdef double upper = self.bounds[1]
        cdef int i, j, k
        cdef double[:, :] reef
        cdef double perturbation

        srand(<unsigned int>time(NULL))
        for i in range(self.num_reefs):
            reef = self.reefs[i]
            for j in range(self.population_size):
                for k in range(self.dim):
                    # Approximate Gaussian noise using Box-Muller transform
                    perturbation = self.alpha * sqrt(-2.0 * log((<double>rand() / <double>RAND_MAX))) * \
                                  cos(2.0 * 3.141592653589793 * (<double>rand() / <double>RAND_MAX))
                    reef[j, k] += perturbation
                    if reef[j, k] < lower:
                        reef[j, k] = lower
                    elif reef[j, k] > upper:
                        reef[j, k] = upper

    cpdef tuple optimize(self):
        """Run the Coral Reefs Optimization algorithm."""
        self.initialize_reefs()
        cdef double[:, :] fitness
        cdef int generation, i, j
        cdef double min_fitness
        cdef int best_reef_idx, best_solution_idx
        cdef double[:, :] reef

        for generation in range(self.max_iter):
            # Evaluate fitness
            fitness = self.evaluate_reefs()

            # Find best solution across all reefs
            min_fitness = float("inf")
            best_reef_idx = 0
            best_solution_idx = 0
            for i in range(self.num_reefs):
                for j in range(self.population_size):
                    if fitness[i, j] < min_fitness:
                        min_fitness = fitness[i, j]
                        best_reef_idx = i
                        best_solution_idx = j
            if min_fitness < self.best_value:
                self.best_value = min_fitness
                reef = self.reefs[best_reef_idx]
                for k in range(self.dim):
                    self.best_solution[k] = reef[best_solution_idx, k]

            # Migration phase
            self.migration_phase()

            # Local search phase
            self.local_search_phase()

            self.history.append((generation, np.asarray(self.best_solution).copy(), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")

        return np.asarray(self.best_solution), self.best_value, self.history
