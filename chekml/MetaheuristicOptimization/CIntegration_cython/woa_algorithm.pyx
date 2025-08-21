# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport cos, pi, abs as c_abs
from libc.stdlib cimport rand, RAND_MAX

# Define NumPy array types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef class WhaleOptimizationAlgorithm:
    cdef public:
        object objective_function
        int dim
        np.ndarray bounds
        int population_size
        int max_iter
        np.ndarray positions
        np.ndarray leader_pos
        double leader_score
        list search_history

    def __init__(self, objective_function, int dim, bounds, int population_size=30, int max_iter=100):
        """
        Initialize the Whale Optimization Algorithm (WOA).

        Parameters:
        - objective_function: Function to optimize (minimize).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of search agents (whales).
        - max_iter: Maximum number of iterations.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)
        self.population_size = population_size
        self.max_iter = max_iter
        self.positions = None
        self.leader_pos = None
        self.leader_score = float("inf")
        self.search_history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def initialize_positions(self):
        """Initialize the first population of search agents."""
        cdef np.ndarray[DTYPE_t, ndim=1] lb = self.bounds[:, 0]
        cdef np.ndarray[DTYPE_t, ndim=1] ub = self.bounds[:, 1]
        cdef int i
        if lb.size == 1:  # Single boundary for all dimensions
            self.positions = np.random.rand(self.population_size, self.dim) * (ub[0] - lb[0]) + lb[0]
        else:  # Different boundaries for each dimension
            self.positions = np.zeros((self.population_size, self.dim), dtype=DTYPE)
            for i in range(self.dim):
                self.positions[:, i] = np.random.rand(self.population_size) * (ub[i] - lb[i]) + lb[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def enforce_bounds(self):
        """Return search agents that go beyond the boundaries to valid space."""
        cdef np.ndarray[DTYPE_t, ndim=1] lb = self.bounds[:, 0]
        cdef np.ndarray[DTYPE_t, ndim=1] ub = self.bounds[:, 1]
        cdef np.ndarray[DTYPE_t, ndim=2] positions = self.positions
        cdef int i, j
        cdef bint flag4ub, flag4lb
        for i in range(self.population_size):
            for j in range(self.dim):
                flag4ub = positions[i, j] > ub[j if lb.size > 1 else 0]
                flag4lb = positions[i, j] < lb[j if lb.size > 1 else 0]
                if flag4ub:
                    positions[i, j] = ub[j if lb.size > 1 else 0]
                elif flag4lb:
                    positions[i, j] = lb[j if lb.size > 1 else 0]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """Run the Whale Optimization Algorithm."""
        self.initialize_positions()
        self.leader_pos = np.zeros(self.dim, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2] positions = self.positions
        cdef np.ndarray[DTYPE_t, ndim=1] leader_pos = self.leader_pos
        cdef double leader_score = self.leader_score
        cdef int t, i, j, rand_leader_index
        cdef double a, a2, r1, r2, A, C, b, l, p, fitness, D_X_rand, D_Leader, distance2Leader
        cdef np.ndarray[DTYPE_t, ndim=1] X_rand

        for t in range(self.max_iter):
            # Store current positions in history
            self.search_history.append(positions.copy())

            # Evaluate and update leader
            for i in range(self.population_size):
                self.enforce_bounds()
                fitness = self.objective_function(positions[i, :])
                if fitness < leader_score:
                    leader_score = fitness
                    leader_pos[:] = positions[i, :]
                    self.leader_score = leader_score
                    self.leader_pos = leader_pos

            # Update parameters
            a = 2.0 - t * (2.0 / self.max_iter)  # Linearly decreases from 2 to 0
            a2 = -1.0 + t * (-1.0 / self.max_iter)  # Linearly decreases from -1 to -2

            # Update positions of search agents
            for i in range(self.population_size):
                r1 = <double>rand() / RAND_MAX
                r2 = <double>rand() / RAND_MAX
                A = 2.0 * a * r1 - a  # Eq. (2.3)
                C = 2.0 * r2  # Eq. (2.4)
                b = 1.0  # Parameter in Eq. (2.5)
                l = (a2 - 1.0) * (<double>rand() / RAND_MAX) + 1.0  # Parameter in Eq. (2.5)
                p = <double>rand() / RAND_MAX  # Random number for strategy selection

                for j in range(self.dim):
                    if p < 0.5:
                        if c_abs(A) >= 1.0:  # Search for prey (exploration)
                            rand_leader_index = <int>(self.population_size * (<double>rand() / RAND_MAX))
                            X_rand = positions[rand_leader_index, :]
                            D_X_rand = c_abs(C * X_rand[j] - positions[i, j])  # Eq. (2.7)
                            positions[i, j] = X_rand[j] - A * D_X_rand  # Eq. (2.8)
                        else:  # Encircling prey (exploitation)
                            D_Leader = c_abs(C * leader_pos[j] - positions[i, j])  # Eq. (2.1)
                            positions[i, j] = leader_pos[j] - A * D_Leader  # Eq. (2.2)
                    else:  # Spiral bubble-net attack
                        distance2Leader = c_abs(leader_pos[j] - positions[i, j])
                        positions[i, j] = distance2Leader * np.exp(b * l) * cos(l * 2.0 * pi) + leader_pos[j]  # Eq. (2.5)

            print(f"Iteration {t + 1}: Best Score = {self.leader_score}")

        return self.leader_pos, self.leader_score, self.search_history
