# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, fabs
from libc.stdlib cimport rand, RAND_MAX

# Define NumPy types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef class ChickenSwarmOptimizer:
    """
    Cythonized Chicken Swarm Optimization (CSO) algorithm for unconstrained optimization problems.
    Based on the MATLAB implementation by Xian-Bing Meng.
    """
    cdef:
        object objective_function
        int dim
        np.ndarray bounds
        int population_size
        int max_iter
        int update_freq
        double rooster_ratio
        double hen_ratio
        double mother_ratio
        int rooster_num
        int hen_num
        int chick_num
        int mother_num
        np.ndarray positions
        np.ndarray fitness
        np.ndarray best_positions
        np.ndarray best_fitness
        np.ndarray global_best_position
        double global_best_fitness
        list history

    def __init__(self, object objective_function, int dim, bounds, int population_size=100, 
                 int max_iter=100, int update_freq=10, double rooster_ratio=0.15, 
                 double hen_ratio=0.7, double mother_ratio=0.5):
        """
        Initialize the CSO optimizer.

        Parameters:
        - objective_function: Function to optimize (minimize).
        - dim: Number of dimensions (variables).
        - bounds: List of (lower, upper) bounds for each dimension.
        - population_size: Total number of chickens (solutions).
        - max_iter: Maximum number of iterations.
        - update_freq: How often the swarm hierarchy is updated (G).
        - rooster_ratio: Proportion of roosters in the population.
        - hen_ratio: Proportion of hens in the population.
        - mother_ratio: Proportion of hens that are mothers.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)
        self.population_size = population_size
        self.max_iter = max_iter
        self.update_freq = update_freq
        self.rooster_ratio = rooster_ratio
        self.hen_ratio = hen_ratio
        self.mother_ratio = mother_ratio

        # Calculate group sizes
        self.rooster_num = int(np.round(population_size * rooster_ratio))
        self.hen_num = int(np.round(population_size * hen_ratio))
        self.chick_num = population_size - self.rooster_num - self.hen_num
        self.mother_num = int(np.round(self.hen_num * mother_ratio))

        # Initialize arrays
        self.positions = None
        self.fitness = None
        self.best_positions = None
        self.best_fitness = None
        self.global_best_position = None
        self.global_best_fitness = float("inf")
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_population(self):
        """Generate initial chicken positions randomly within bounds."""
        cdef:
            np.ndarray[DTYPE_t, ndim=2] positions
            np.ndarray[DTYPE_t, ndim=1] fitness
            np.ndarray[DTYPE_t, ndim=2] best_positions
            np.ndarray[DTYPE_t, ndim=1] best_fitness
            np.ndarray[DTYPE_t, ndim=1] lb = self.bounds[:, 0]
            np.ndarray[DTYPE_t, ndim=1] ub = self.bounds[:, 1]
            int i, j
            double min_fitness
            int min_idx

        positions = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        fitness = np.zeros(self.population_size, dtype=DTYPE)
        for i in range(self.population_size):
            fitness[i] = self.objective_function(positions[i])
        
        self.positions = positions
        self.fitness = fitness
        self.best_positions = positions.copy()
        self.best_fitness = fitness.copy()
        
        min_idx = np.argmin(fitness)
        self.global_best_position = positions[min_idx].copy()
        self.global_best_fitness = fitness[min_idx]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[DTYPE_t, ndim=1] apply_bounds(self, np.ndarray[DTYPE_t, ndim=1] position):
        """Apply lower and upper bounds to a position."""
        return np.clip(position, self.bounds[:, 0], self.bounds[:, 1])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int randi_tabu(self, int min_val, int max_val, int tabu):
        """Generate random integer excluding a tabu value."""
        cdef int temp
        while True:
            temp = min_val + int((max_val - min_val + 1) * (rand() / <double>RAND_MAX))
            if temp != tabu:
                return temp

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray randperm_f(self, int range_size, int dim):
        """Generate a permutation, extending randperm for larger dimensions."""
        cdef np.ndarray temp = np.random.permutation(range_size)
        if dim > range_size:
            extra = np.random.randint(1, range_size + 1, dim - range_size)
            return np.concatenate([temp, extra])
        return temp[:dim]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[DTYPE_t, ndim=1] update_rooster(self, int idx, np.ndarray sort_indices):
        """Update rooster position based on CSO rules."""
        cdef:
            int another_rooster = self.randi_tabu(1, self.rooster_num, idx + 1)
            int another_idx = sort_indices[another_rooster - 1]
            int curr_idx = sort_indices[idx]
            double sigma
            np.ndarray[DTYPE_t, ndim=1] new_position
            int j

        if self.best_fitness[curr_idx] <= self.best_fitness[another_idx]:
            sigma = 1.0
        else:
            sigma = exp((self.best_fitness[another_idx] - self.best_fitness[curr_idx]) /
                        (fabs(self.best_fitness[curr_idx]) + 2.2e-16))

        new_position = np.zeros(self.dim, dtype=DTYPE)
        for j in range(self.dim):
            new_position[j] = self.best_positions[curr_idx, j] * (1.0 + sigma * np.random.randn())
        
        return self.apply_bounds(new_position)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[DTYPE_t, ndim=1] update_hen(self, int idx, np.ndarray sort_indices, np.ndarray mate):
        """Update hen position based on CSO rules."""
        cdef:
            int mate_idx = sort_indices[mate[idx - self.rooster_num] - 1]
            int other = self.randi_tabu(1, idx, mate[idx - self.rooster_num])
            int other_idx = sort_indices[other - 1]
            int curr_idx = sort_indices[idx - 1]
            double c1, c2
            np.ndarray[DTYPE_t, ndim=1] new_position
            int j

        c1 = exp((self.best_fitness[curr_idx] - self.best_fitness[mate_idx]) /
                 (fabs(self.best_fitness[curr_idx]) + 2.2e-16))
        c2 = exp(-self.best_fitness[curr_idx] + self.best_fitness[other_idx])

        new_position = np.zeros(self.dim, dtype=DTYPE)
        for j in range(self.dim):
            new_position[j] = (self.best_positions[curr_idx, j] +
                              (self.best_positions[mate_idx, j] - self.best_positions[curr_idx, j]) * c1 * np.random.rand() +
                              (self.best_positions[other_idx, j] - self.best_positions[curr_idx, j]) * c2 * np.random.rand())
        
        return self.apply_bounds(new_position)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[DTYPE_t, ndim=1] update_chick(self, int idx, np.ndarray sort_indices, np.ndarray mother_indices):
        """Update chick position based on CSO rules."""
        cdef:
            int curr_idx = sort_indices[idx - 1]
            int mother_idx = sort_indices[mother_indices[idx - self.rooster_num - self.hen_num] - 1]
            double fl = 0.5 + 0.4 * np.random.rand()  # FL in [0.5, 0.9]
            np.ndarray[DTYPE_t, ndim=1] new_position
            int j

        new_position = np.zeros(self.dim, dtype=DTYPE)
        for j in range(self.dim):
            new_position[j] = (self.best_positions[curr_idx, j] +
                              (self.best_positions[mother_idx, j] - self.best_positions[curr_idx, j]) * fl)
        
        return self.apply_bounds(new_position)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """Run the Chicken Swarm Optimization algorithm."""
        cdef:
            int t, i
            np.ndarray sort_indices
            np.ndarray mother_lib
            np.ndarray mate
            np.ndarray mother_indices
            np.ndarray[DTYPE_t, ndim=1] new_position

        self.initialize_population()

        for t in range(self.max_iter):
            # Update swarm hierarchy every update_freq iterations or at start
            if t % self.update_freq == 0 or t == 0:
                sort_indices = np.argsort(self.best_fitness)
                mother_lib = np.random.permutation(self.hen_num)[:self.mother_num] + self.rooster_num
                mate = self.randperm_f(self.rooster_num, self.hen_num)
                mother_indices = mother_lib[np.random.randint(0, self.mother_num, self.chick_num)]

            # Update roosters
            for i in range(self.rooster_num):
                self.positions[sort_indices[i]] = self.update_rooster(i, sort_indices)
                self.fitness[sort_indices[i]] = self.objective_function(self.positions[sort_indices[i]])

            # Update hens
            for i in range(self.rooster_num, self.rooster_num + self.hen_num):
                self.positions[sort_indices[i]] = self.update_hen(i, sort_indices, mate)
                self.fitness[sort_indices[i]] = self.objective_function(self.positions[sort_indices[i]])

            # Update chicks
            for i in range(self.rooster_num + self.hen_num, self.population_size):
                self.positions[sort_indices[i]] = self.update_chick(i, sort_indices, mother_indices)
                self.fitness[sort_indices[i]] = self.objective_function(self.positions[sort_indices[i]])

            # Update individual and global bests
            for i in range(self.population_size):
                if self.fitness[i] < self.best_fitness[i]:
                    self.best_fitness[i] = self.fitness[i]
                    self.best_positions[i] = self.positions[i].copy()
                if self.best_fitness[i] < self.global_best_fitness:
                    self.global_best_fitness = self.best_fitness[i]
                    self.global_best_position = self.best_positions[i].copy()

            self.history.append((t, self.global_best_position.copy(), self.global_best_fitness))
            print(f"Iteration {t + 1}: Best Value = {self.global_best_fitness}")

        return self.global_best_position, self.global_best_fitness, self.history
