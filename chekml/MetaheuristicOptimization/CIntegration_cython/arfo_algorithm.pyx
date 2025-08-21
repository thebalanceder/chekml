# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt
from cython.parallel import prange

# Define NumPy types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

class ArtificialRootForagingOptimizer:
    def __init__(self, objective_function, int dim, bounds, int population_size=50, int max_iter=100, 
                 double branching_threshold=0.6, int max_branching=5, int min_branching=1, 
                 double initial_std=1.0, double final_std=0.01, double max_elongation=0.1):
        """
        Initialize the Hybrid Artificial Root Foraging Optimizer (HARFO).
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.branching_threshold = branching_threshold
        self.max_branching = max_branching
        self.min_branching = min_branching
        self.initial_std = initial_std
        self.final_std = final_std
        self.max_elongation = max_elongation

        self.roots = None  # Population of root apices (solutions)
        self.best_solution = None
        self.best_value = np.inf
        self.history = []

    def initialize_roots(self):
        """ Generate initial root apices randomly within bounds """
        self.roots = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                       (self.population_size, self.dim)).astype(DTYPE)

    def evaluate_fitness(self):
        """ Compute fitness values for the root population """
        return np.array([self.objective_function(root) for root in self.roots], dtype=DTYPE)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def calculate_auxin_concentration(self, np.ndarray[DTYPE_t, ndim=1] fitness):
        """ Calculate auxin concentration for each root based on fitness """
        cdef double f_min = np.min(fitness)
        cdef double f_max = np.max(fitness)
        cdef np.ndarray[DTYPE_t, ndim=1] f_normalized
        cdef np.ndarray[DTYPE_t, ndim=1] auxin
        cdef int i, n = fitness.shape[0]
        
        if f_max == f_min:
            f_normalized = np.ones(n, dtype=DTYPE) / n
        else:
            f_normalized = np.empty(n, dtype=DTYPE)
            for i in prange(n, nogil=True):
                f_normalized[i] = (fitness[i] - f_min) / (f_max - f_min)
        
        cdef double sum_f = np.sum(f_normalized)
        auxin = f_normalized / sum_f
        return auxin

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def construct_von_neumann_topology(self, int current_pop_size):
        """ Construct Von Neumann topology for root-to-root communication """
        cdef int rows = int(sqrt(current_pop_size))
        cdef int cols = (current_pop_size + rows - 1) // rows
        cdef np.ndarray[np.int32_t, ndim=2] topology = np.full((current_pop_size, 4), -1, dtype=np.int32)
        cdef int i, row, col

        for i in range(current_pop_size):
            row, col = i // cols, i % cols
            if col > 0:
                topology[i, 0] = i - 1  # Left
            if col < cols - 1 and i + 1 < current_pop_size:
                topology[i, 1] = i + 1  # Right
            if row > 0:
                topology[i, 2] = i - cols  # Up
            if row < rows - 1 and i + cols < current_pop_size:
                topology[i, 3] = i + cols  # Down
        return topology

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def main_root_regrowth(self, int root_idx, np.ndarray[DTYPE_t, ndim=1] local_best):
        """ Apply regrowing operator for main roots """
        cdef double local_inertia = 0.5
        cdef np.ndarray[DTYPE_t, ndim=1] rand_coeff = np.random.rand(self.dim).astype(DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] new_position = np.empty(self.dim, dtype=DTYPE)
        cdef int i

        for i in range(self.dim):
            new_position[i] = self.roots[root_idx, i] + local_inertia * rand_coeff[i] * (local_best[i] - self.roots[root_idx, i])
            new_position[i] = min(max(new_position[i], self.bounds[i, 0]), self.bounds[i, 1])
        return new_position

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def main_root_branching(self, int root_idx, np.ndarray[DTYPE_t, ndim=1] auxin, int current_iter):
        """ Apply branching operator for main roots if condition is met """
        cdef double R1
        cdef int num_new_roots
        cdef double std
        cdef np.ndarray[DTYPE_t, ndim=2] new_roots
        cdef int i, j

        if auxin[root_idx] > self.branching_threshold:
            R1 = np.random.rand()
            num_new_roots = int(R1 * auxin[root_idx] * (self.max_branching - self.min_branching) + self.min_branching)
            std = ((self.max_iter - current_iter) / self.max_iter) * (self.initial_std - self.final_std) + self.final_std
            new_roots = np.random.normal(self.roots[root_idx], std, 
                                         (num_new_roots, self.dim)).astype(DTYPE)
            for i in range(num_new_roots):
                for j in range(self.dim):
                    new_roots[i, j] = min(max(new_roots[i, j], self.bounds[j, 0]), self.bounds[j, 1])
            return new_roots
        return None

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def lateral_root_growth(self, int root_idx):
        """ Apply random walk for lateral roots """
        cdef double rand_length = np.random.rand() * self.max_elongation
        cdef np.ndarray[DTYPE_t, ndim=1] random_vector = np.random.randn(self.dim).astype(DTYPE)
        cdef double norm = sqrt(np.sum(random_vector**2))
        cdef np.ndarray[DTYPE_t, ndim=1] growth_angle = random_vector / norm
        cdef np.ndarray[DTYPE_t, ndim=1] new_position = np.empty(self.dim, dtype=DTYPE)
        cdef int i

        for i in range(self.dim):
            new_position[i] = self.roots[root_idx, i] + rand_length * growth_angle[i]
            new_position[i] = min(max(new_position[i], self.bounds[i, 0]), self.bounds[i, 1])
        return new_position

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def dead_root_elimination(self, np.ndarray[DTYPE_t, ndim=1] auxin):
        """ Remove roots with low auxin concentration """
        cdef double elimination_threshold = np.percentile(auxin, 10)
        cdef np.ndarray[np.uint8_t, ndim=1, cast=True] keep_mask = auxin > elimination_threshold
        self.roots = self.roots[keep_mask]
        return keep_mask

    def optimize(self):
        """ Run the Hybrid Artificial Root Foraging Optimization (HARFO) """
        self.initialize_roots()

        for iteration in range(self.max_iter):
            current_pop_size = self.roots.shape[0]
            topology = self.construct_von_neumann_topology(current_pop_size)

            fitness = self.evaluate_fitness()
            auxin = self.calculate_auxin_concentration(fitness)
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.roots[min_idx].copy()
                self.best_value = fitness[min_idx]

            main_root_mask = auxin > np.median(auxin)
            lateral_root_mask = ~main_root_mask

            new_roots = []
            for i in range(current_pop_size):
                if main_root_mask[i]:
                    neighbor_indices = topology[i]
                    valid_neighbors = neighbor_indices[neighbor_indices >= 0]
                    if len(valid_neighbors) > 0:
                        neighbor_fitness = fitness[valid_neighbors]
                        local_best_idx = valid_neighbors[np.argmin(neighbor_fitness)]
                        local_best = self.roots[local_best_idx]
                    else:
                        local_best = self.roots[i]

                    self.roots[i] = self.main_root_regrowth(i, local_best)

                    new_branch = self.main_root_branching(i, auxin, iteration)
                    if new_branch is not None:
                        new_roots.append(new_branch)

            for i in range(current_pop_size):
                if lateral_root_mask[i]:
                    self.roots[i] = self.lateral_root_growth(i)

            if new_roots:
                new_roots = np.vstack(new_roots)
                self.roots = np.vstack([self.roots, new_roots])
                if self.roots.shape[0] > self.population_size:
                    fitness = self.evaluate_fitness()
                    keep_indices = np.argsort(fitness)[:self.population_size]
                    self.roots = self.roots[keep_indices]

            fitness = self.evaluate_fitness()
            auxin = self.calculate_auxin_concentration(fitness)
            keep_mask = self.dead_root_elimination(auxin)
            fitness = fitness[keep_mask]
            auxin = auxin[keep_mask]

            while self.roots.shape[0] < self.population_size:
                new_root = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim).astype(DTYPE)
                self.roots = np.vstack([self.roots, new_root])

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
