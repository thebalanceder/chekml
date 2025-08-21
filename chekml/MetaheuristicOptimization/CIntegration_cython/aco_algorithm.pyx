# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport pow, fmax
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time

# Define type for objective function (Python callable)
ctypedef double (*obj_func_t)(double[:])

cdef class AntColonyOptimizer:
    cdef readonly int dim, max_iter, n_ant, n_bins
    cdef readonly double Q, tau0, alpha, beta, rho
    cdef readonly object objective_function
    cdef double[:, :] bounds
    cdef double[:, :] bins  # n_bins x dim
    cdef double[:, :] tau   # n_bins x dim
    cdef double[:, :] eta   # n_bins x dim
    cdef double[:] ant_costs
    cdef int[:, :] ant_tours
    cdef double[:, :] ant_xs
    cdef double[:] best_solution
    cdef double best_cost
    cdef list history

    def __init__(self, object objective_function, int dim, bounds, 
                 int max_iter=300, int n_ant=40, double Q=1.0, double tau0=0.1, 
                 double alpha=1.0, double beta=0.02, double rho=0.1, int n_bins=10):
        """
        Initialize the Ant Colony Optimizer for continuous optimization problems.

        Parameters:
        - objective_function: Function to optimize (minimization).
        - dim: Number of dimensions (variables).
        - bounds: List of tuples [(lower, upper), ...] for each dimension.
        - max_iter: Maximum number of iterations.
        - n_ant: Number of ants (population size).
        - Q: Pheromone deposit factor.
        - tau0: Initial pheromone value.
        - alpha: Pheromone exponential weight.
        - beta: Heuristic exponential weight.
        - rho: Pheromone evaporation rate.
        - n_bins: Number of discrete bins per dimension for pheromone grid.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.max_iter = max_iter
        self.n_ant = n_ant
        self.Q = Q
        self.tau0 = tau0
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.n_bins = n_bins

        # Initialize arrays
        self.bounds = np.array(bounds, dtype=np.double)
        self.bins = np.zeros((self.n_bins, self.dim), dtype=np.double)
        self.tau = np.full((self.n_bins, self.dim), self.tau0, dtype=np.double)
        self.eta = np.ones((self.n_bins, self.dim), dtype=np.double)
        self.ant_costs = np.zeros(self.n_ant, dtype=np.double)
        self.ant_tours = np.zeros((self.n_ant, self.dim), dtype=np.int32)
        self.ant_xs = np.zeros((self.n_ant, self.dim), dtype=np.double)
        self.best_solution = np.zeros(self.dim, dtype=np.double)
        self.best_cost = float('inf')
        self.history = []

        # Seed the random number generator
        srand(<unsigned int>time(NULL))

        # Initialize bins
        cdef int d, i
        for d in range(self.dim):
            for i in range(self.n_bins):
                self.bins[i, d] = self.bounds[d, 0] + (self.bounds[d, 1] - self.bounds[d, 0]) * i / (self.n_bins - 1)

    cdef int _roulette_wheel_selection(self, double[:] P) nogil:
        """Select an index based on roulette wheel selection."""
        cdef double r = <double>rand() / <double>RAND_MAX
        cdef double cumsum = 0.0
        cdef int i
        for i in range(P.shape[0]):
            cumsum += P[i]
            if r <= cumsum:
                return i
        return P.shape[0] - 1  # Fallback

    @cython.cdivision(True)
    cdef void _construct_solutions(self):
        """Construct solutions for all ants."""
        cdef int k, d, bin_idx
        cdef double[:] P = np.zeros(self.n_bins, dtype=np.double)
        cdef double sum_P

        for k in range(self.n_ant):
            for d in range(self.dim):
                # Compute probabilities
                sum_P = 0.0
                for i in range(self.n_bins):
                    P[i] = pow(self.tau[i, d], self.alpha) * pow(self.eta[i, d], self.beta)
                    sum_P += P[i]
                
                # Normalize probabilities
                if sum_P > 0:
                    for i in range(self.n_bins):
                        P[i] /= sum_P
                
                # Select bin
                bin_idx = self._roulette_wheel_selection(P)
                self.ant_tours[k, d] = bin_idx
                self.ant_xs[k, d] = self.bins[bin_idx, d]
            
            # Evaluate cost
            self.ant_costs[k] = self.objective_function(self.ant_xs[k])

    @cython.cdivision(True)
    cdef void _update_pheromones(self):
        """Update pheromone trails."""
        cdef int k, d, bin_idx
        for k in range(self.n_ant):
            for d in range(self.dim):
                bin_idx = self.ant_tours[k, d]
                self.tau[bin_idx, d] += self.Q / (1.0 + fmax(0.0, self.ant_costs[k] - self.best_cost))

    cdef void _evaporate_pheromones(self):
        """Apply pheromone evaporation."""
        cdef int i, d
        for i in range(self.n_bins):
            for d in range(self.dim):
                self.tau[i, d] *= (1.0 - self.rho)

    def optimize(self):
        """Run the Ant Colony Optimization algorithm for continuous problems."""
        cdef int it, k
        cdef double current_best_cost
        cdef int current_best_idx

        for it in range(self.max_iter):
            # Construct solutions
            self._construct_solutions()

            # Find best ant
            current_best_cost = float('inf')
            current_best_idx = 0
            for k in range(self.n_ant):
                if self.ant_costs[k] < current_best_cost:
                    current_best_cost = self.ant_costs[k]
                    current_best_idx = k

            # Update global best
            if current_best_cost < self.best_cost:
                self.best_cost = current_best_cost
                for d in range(self.dim):
                    self.best_solution[d] = self.ant_xs[current_best_idx, d]

            # Update pheromones
            self._update_pheromones()
            self._evaporate_pheromones()

            # Store history
            self.history.append((it, np.array(self.best_solution), self.best_cost))
            
            # Print progress
            print(f"Iteration {it + 1}: Best Cost = {self.best_cost}")

        return np.array(self.best_solution), self.best_cost, self.history
