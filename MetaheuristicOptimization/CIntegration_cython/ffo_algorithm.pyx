# cython: language_level=3
import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt

cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class FruitFlyOptimizationAlgorithm:
    cdef object objective_function
    cdef int dim
    cdef cnp.ndarray bounds
    cdef int population_size
    cdef int max_iter
    cdef double search_range
    cdef str mode
    cdef double delta
    cdef double chaos_factor
    cdef cnp.ndarray swarm
    cdef cnp.ndarray swarm_axis
    cdef cnp.ndarray best_solution
    cdef double best_value
    cdef list history

    def __init__(self, object objective_function, int dim, cnp.ndarray bounds, 
                 int population_size=20, int max_iter=100, double search_range=1.0,
                 str mode='FOA', double delta=0.5, double chaos_factor=0.5):
        """
        Initialize the Fruit Fly Optimization Algorithm (FOA) optimizer.

        Parameters:
        - objective_function: Function to optimize (expects numpy array input, returns scalar).
        - dim: Number of dimensions (variables).
        - bounds: Array of (lower, upper) bounds for each dimension [(low, high), ...].
        - population_size: Number of fruit flies in the swarm.
        - max_iter: Maximum number of iterations.
        - search_range: Range for random direction and distance in smell-based search.
        - mode: Optimization mode ('FOA', 'MFOA', or 'CFOA').
        - delta: Parameter for MFOA to adjust smell concentration (0 ≤ delta ≤ 1).
        - chaos_factor: Scaling factor for CFOA chaotic search.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = bounds
        self.population_size = population_size
        self.max_iter = max_iter
        self.search_range = search_range
        self.mode = mode.upper()
        self.delta = delta
        self.chaos_factor = chaos_factor
        self.best_value = float("inf")
        self.history = []

        if self.mode not in ['FOA', 'MFOA', 'CFOA']:
            raise ValueError("Mode must be 'FOA', 'MFOA', or 'CFOA'")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def initialize_swarm(self):
        """Initialize the fruit fly swarm with random positions."""
        cdef int axis_dim = 3 if self.mode == 'MFOA' else 2
        cdef cnp.ndarray[cnp.double_t, ndim=1] swarm_axis = \
            np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], axis_dim)
        self.swarm_axis = swarm_axis

        cdef cnp.ndarray[cnp.double_t, ndim=2] swarm = \
            np.zeros((self.population_size, self.dim), dtype=np.double)
        cdef int i
        for i in range(self.population_size):
            swarm[i] = self.smell_based_search(i)
        self.swarm = swarm

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.double_t, ndim=1] smell_based_search(self, int index):
        """Select smell-based search based on mode."""
        if self.mode == 'CFOA':
            return self.chaos_based_search(index)
        elif self.mode == 'MFOA':
            return self.modified_smell_search(index)
        else:
            return self.basic_smell_search(index)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.double_t, ndim=1] basic_smell_search(self, int index):
        """Basic FOA smell-based search (2D)."""
        cdef cnp.ndarray[cnp.double_t, ndim=1] random_value = \
            self.search_range * (2 * np.random.rand(self.dim) - 1)
        cdef cnp.ndarray[cnp.double_t, ndim=1] new_position = \
            self.swarm_axis[:self.dim] + random_value
        return np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.double_t, ndim=1] modified_smell_search(self, int index):
        """Modified FOA smell-based search (3D)."""
        cdef cnp.ndarray[cnp.double_t, ndim=1] random_value = \
            self.search_range * (2 * np.random.rand(3) - 1)
        cdef cnp.ndarray[cnp.double_t, ndim=1] new_position = \
            self.swarm_axis + random_value
        new_position = new_position[:self.dim]
        return np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.double_t, ndim=1] chaos_based_search(self, int index):
        """CFOA chaos-based search."""
        cdef cnp.ndarray[cnp.double_t, ndim=1] chaotic_value = \
            self.chaos_factor * (2 * np.random.rand(self.dim) - 1)
        cdef cnp.ndarray[cnp.double_t, ndim=1] max_diff = \
            np.maximum(self.bounds[:, 1] - self.swarm_axis[:self.dim],
                       self.swarm_axis[:self.dim] - self.bounds[:, 0])
        cdef cnp.ndarray[cnp.double_t, ndim=1] new_position = \
            self.swarm_axis[:self.dim] + chaotic_value * max_diff
        return np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.double_t, ndim=1] evaluate_swarm(self):
        """Compute smell concentration (fitness) for the swarm."""
        cdef cnp.ndarray[cnp.double_t, ndim=1] smell = \
            np.zeros(self.population_size, dtype=np.double)
        cdef int i, j
        cdef double dist, s, delta_adjust
        cdef cnp.ndarray[cnp.double_t, ndim=1] position
        for i in range(self.population_size):
            position = self.swarm[i]
            dist = 0.0
            for j in range(self.dim):
                dist += position[j] * position[j]
            dist = sqrt(dist)
            if self.mode == 'MFOA':
                delta_adjust = dist * (0.5 - self.delta)
                s = 1.0 / dist + delta_adjust if dist != 0 else 1e-10
            else:
                s = 1.0 / dist if dist != 0 else 1e-10
            smell[i] = self.objective_function(s * np.ones(self.dim))
        return smell

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void local_search(self):
        """CFOA local search using golden ratio."""
        cdef int i, j
        cdef double xl, yl
        cdef cnp.ndarray[cnp.double_t, ndim=1] local_position
        cdef double dist, s, local_smell
        for i in range(self.population_size):
            xl = 0.618 * self.swarm_axis[0] + 0.382 * self.swarm[i, 0]
            yl = 0.618 * self.swarm_axis[1] + 0.382 * self.swarm[i, 1]
            local_position = np.array([xl, yl])[:self.dim]
            local_position = np.clip(local_position, self.bounds[:, 0], self.bounds[:, 1])
            dist = 0.0
            for j in range(self.dim):
                dist += local_position[j] * local_position[j]
            dist = sqrt(dist)
            s = 1.0 / dist if dist != 0 else 1e-10
            local_smell = self.objective_function(s * np.ones(self.dim))
            if local_smell < self.objective_function(self.swarm[i]):
                for j in range(self.dim):
                    self.swarm[i, j] = local_position[j]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """Run the Fruit Fly Optimization Algorithm."""
        self.initialize_swarm()
        cdef double smell_best = float("inf")
        cdef cnp.ndarray[cnp.double_t, ndim=1] smell
        cdef int generation, i, min_idx
        cdef double current_best_smell
        cdef cnp.ndarray[cnp.double_t, ndim=1] new_axis

        for generation in range(self.max_iter):
            for i in range(self.population_size):
                self.swarm[i] = self.smell_based_search(i)

            smell = self.evaluate_swarm()
            min_idx = np.argmin(smell)
            current_best_smell = smell[min_idx]

            if current_best_smell < smell_best:
                smell_best = current_best_smell
                self.best_solution = self.swarm[min_idx].copy()
                self.best_value = self.objective_function(self.best_solution)
                if self.mode == 'MFOA':
                    new_axis = np.array([self.best_solution[0], self.best_solution[1], self.best_solution[0]]) \
                        if self.dim >= 2 else np.array([self.best_solution[0], self.best_solution[0], self.best_solution[0]])
                else:
                    new_axis = np.array([self.best_solution[0], self.best_solution[1]]) \
                        if self.dim >= 2 else np.array([self.best_solution[0], self.best_solution[0]])
                self.swarm_axis = new_axis

            if self.mode == 'CFOA' and generation >= self.max_iter // 2:
                self.local_search()

            self.history.append((generation, self.best_solution.copy(), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
