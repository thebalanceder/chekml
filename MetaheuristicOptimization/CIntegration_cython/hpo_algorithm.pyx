# cython: language_level=3
import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport cos, pi

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class HunterPreyOptimizer:
    cdef cnp.ndarray hp_positions
    cdef cnp.ndarray target_position
    cdef cnp.ndarray bounds
    cdef cnp.ndarray convergence_curve
    cdef list history
    cdef double target_score
    cdef int dim
    cdef int population_size
    cdef int max_iter
    cdef double constriction_coeff
    cdef object objective_function

    def __init__(self, objective_function, int dim, bounds, int population_size=30, int max_iter=100, double constriction_coeff=0.1):
        """
        Initialize the Hunter-Prey Optimizer (HPO).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of hunters/preys (solutions).
        - max_iter: Maximum number of iterations.
        - constriction_coeff: Constriction coefficient (B parameter).
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.double)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.constriction_coeff = constriction_coeff

        self.hp_positions = None
        self.target_position = None
        self.target_score = float("inf")
        self.convergence_curve = np.zeros(max_iter, dtype=np.double)
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def initialize_positions(self):
        """ Generate initial hunter/prey positions randomly """
        cdef cnp.ndarray[double, ndim=1] lb = self.bounds[:, 0]
        cdef cnp.ndarray[double, ndim=1] ub = self.bounds[:, 1]
        self.hp_positions = np.random.uniform(lb, ub, (self.population_size, self.dim))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def evaluate_positions(self):
        """ Compute fitness values for the hunter/prey positions """
        cdef cnp.ndarray[double, ndim=1] fitness = np.empty(self.population_size, dtype=np.double)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.hp_positions[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def update_position(self, int i, double c, int kbest):
        """ Update position of a hunter/prey based on HPO rules """
        cdef cnp.ndarray[char, ndim=1] r1 = np.random.rand(self.dim) < c
        cdef double r2 = np.random.rand()
        cdef cnp.ndarray[double, ndim=1] r3 = np.random.rand(self.dim)
        cdef cnp.ndarray[char, ndim=1] idx = r1 == 0
        cdef cnp.ndarray[double, ndim=1] z = r2 * idx + r3 * (~idx)
        cdef cnp.ndarray[double, ndim=1] xi, SI, new_pos
        cdef cnp.ndarray[double, ndim=1] dist
        cdef cnp.ndarray[cnp.intp_t, ndim=1] idxsortdist
        cdef int j
        cdef double rr

        if np.random.rand() < self.constriction_coeff:
            # Safe mode: Move towards mean and selected individual
            xi = np.mean(self.hp_positions, axis=0)
            dist = np.sqrt(np.sum((xi - self.hp_positions) ** 2, axis=1))
            idxsortdist = np.argsort(dist)
            SI = self.hp_positions[idxsortdist[kbest - 1]]  # kbest-th closest
            self.hp_positions[i] = (self.hp_positions[i] + 0.5 * (
                (2 * c * z * SI - self.hp_positions[i]) +
                (2 * (1 - c) * z * xi - self.hp_positions[i])))
        else:
            # Attack mode: Move towards target with cosine perturbation
            new_pos = np.zeros(self.dim, dtype=np.double)
            for j in range(self.dim):
                rr = -1 + 2 * z[j]
                new_pos[j] = 2 * z[j] * cos(2 * pi * rr) * (self.target_position[j] - self.hp_positions[i][j]) + self.target_position[j]
            self.hp_positions[i] = new_pos

        # Ensure bounds
        self.hp_positions[i] = np.clip(self.hp_positions[i], self.bounds[:, 0], self.bounds[:, 1])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """ Run the Hunter-Prey Optimization algorithm """
        # Initialization
        self.initialize_positions()
        cdef cnp.ndarray[double, ndim=1] hp_fitness = self.evaluate_positions()
        cdef int min_idx = np.argmin(hp_fitness)
        self.target_position = self.hp_positions[min_idx].copy()
        self.target_score = hp_fitness[min_idx]
        self.convergence_curve[0] = self.target_score
        self.history.append((0, self.target_position.copy(), self.target_score))

        # Main loop
        cdef int it, i
        cdef double c
        cdef int kbest
        cdef double fitness
        for it in range(1, self.max_iter):
            c = 1 - it * (0.98 / self.max_iter)  # Update C parameter
            kbest = round(self.population_size * c)  # Update kbest

            for i in range(self.population_size):
                self.update_position(i, c, kbest)
                fitness = self.objective_function(self.hp_positions[i])
                # Update target if better solution found
                if fitness < self.target_score:
                    self.target_position = self.hp_positions[i].copy()
                    self.target_score = fitness

            self.convergence_curve[it] = self.target_score
            self.history.append((it, self.target_position.copy(), self.target_score))
            print(f"Iteration: {it + 1}, Best Cost = {self.target_score}")

        return self.target_position, self.target_score, self.convergence_curve, self.history
