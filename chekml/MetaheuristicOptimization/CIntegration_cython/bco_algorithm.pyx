#cython: language_level=3
import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport sqrt, exp, cos, sin

# Define types for numpy arrays
ctypedef cnp.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class BacterialColonyOptimizer:
    cdef public:
        object objective_function
        int dim
        cnp.ndarray bounds
        int population_size
        int max_iter
        double chemotaxis_step_max
        double chemotaxis_step_min
        double elimination_ratio
        double reproduction_threshold
        double migration_probability
        double communication_prob
        cnp.ndarray bacteria
        cnp.ndarray energy_levels
        cnp.ndarray best_solution
        double best_value
        cnp.ndarray global_best
        list history

    def __init__(self, objective_function, int dim, bounds, int population_size=50, int max_iter=1000,
                 double chemotaxis_step_max=0.2, double chemotaxis_step_min=0.01, double elimination_ratio=0.2,
                 double reproduction_threshold=0.5, double migration_probability=0.1, double communication_prob=0.5):
        """
        Initialize the Bacterial Colony Optimization (BCO) algorithm.

        Parameters:
        - objective_function: Function to optimize (minimization).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of bacteria (solutions).
        - max_iter: Maximum number of iterations.
        - chemotaxis_step_max: Maximum chemotaxis step size.
        - chemotaxis_step_min: Minimum chemotaxis step size.
        - elimination_ratio: Percentage of worst bacteria eliminated per iteration.
        - reproduction_threshold: Energy threshold for reproduction eligibility.
        - migration_probability: Probability of migration for bacteria.
        - communication_prob: Probability of information exchange between bacteria.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.float64)
        self.population_size = population_size
        self.max_iter = max_iter
        self.chemotaxis_step_max = chemotaxis_step_max
        self.chemotaxis_step_min = chemotaxis_step_min
        self.elimination_ratio = elimination_ratio
        self.reproduction_threshold = reproduction_threshold
        self.migration_probability = migration_probability
        self.communication_prob = communication_prob
        self.bacteria = None
        self.energy_levels = None
        self.best_solution = None
        self.best_value = float("inf")
        self.global_best = None
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void initialize_bacteria(self):
        """ Generate initial bacteria population randomly """
        self.bacteria = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                         (self.population_size, self.dim)).astype(np.float64)
        self.energy_levels = np.zeros(self.population_size, dtype=np.float64)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef cnp.ndarray[DTYPE_t, ndim=1] evaluate_bacteria(self):
        """ Compute fitness values for the bacteria population """
        cdef cnp.ndarray[DTYPE_t, ndim=1] fitness = np.zeros(self.population_size, dtype=np.float64)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.bacteria[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double compute_chemotaxis_step(self, int iteration):
        """ Compute adaptive chemotaxis step size (linearly decreasing) """
        return self.chemotaxis_step_min + (self.chemotaxis_step_max - self.chemotaxis_step_min) * \
               ((self.max_iter - iteration) / self.max_iter)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void chemotaxis_and_communication(self, int iteration):
        """ Perform chemotaxis and communication phase """
        cdef cnp.ndarray[DTYPE_t, ndim=2] new_bacteria = self.bacteria.copy()
        cdef cnp.ndarray[DTYPE_t, ndim=1] fitness = self.evaluate_bacteria()
        cdef double chemotaxis_step = self.compute_chemotaxis_step(iteration)
        cdef cnp.ndarray[DTYPE_t, ndim=1] direction = np.zeros(self.dim, dtype=np.float64)
        cdef cnp.ndarray[DTYPE_t, ndim=1] turbulent = np.zeros(self.dim, dtype=np.float64)
        cdef int i, j, neighbor_idx
        cdef double r, neighbor_fitness

        for i in range(self.population_size):
            r = rand() / RAND_MAX
            if r < 0.5:
                # Turbulent direction (Gaussian noise)
                for j in range(self.dim):
                    turbulent[j] = np.random.randn()
                for j in range(self.dim):
                    direction[j] = chemotaxis_step * (
                        0.5 * (self.global_best[j] - self.bacteria[i, j]) +
                        0.5 * (self.bacteria[i, j] - self.bacteria[i, j]) +
                        turbulent[j]
                    )
            else:
                for j in range(self.dim):
                    direction[j] = chemotaxis_step * (
                        0.5 * (self.global_best[j] - self.bacteria[i, j]) +
                        0.5 * (self.bacteria[i, j] - self.bacteria[i, j])
                    )

            for j in range(self.dim):
                new_bacteria[i, j] += direction[j]
                if new_bacteria[i, j] < self.bounds[j, 0]:
                    new_bacteria[i, j] = self.bounds[j, 0]
                elif new_bacteria[i, j] > self.bounds[j, 1]:
                    new_bacteria[i, j] = self.bounds[j, 1]

            # Communication
            if rand() / RAND_MAX < self.communication_prob:
                if rand() / RAND_MAX < 0.5:
                    # Dynamic neighbor oriented
                    neighbor_idx = (i + (-1 if rand() % 2 == 0 else 1)) % self.population_size
                else:
                    # Random oriented
                    neighbor_idx = rand() % self.population_size

                neighbor_fitness = self.objective_function(self.bacteria[neighbor_idx])
                if neighbor_fitness < fitness[i]:
                    for j in range(self.dim):
                        new_bacteria[i, j] = self.bacteria[neighbor_idx, j]
                elif fitness[i] > self.best_value:
                    for j in range(self.dim):
                        new_bacteria[i, j] = self.global_best[j]

        self.bacteria = new_bacteria

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void elimination_and_reproduction(self):
        """ Perform elimination and reproduction phase """
        cdef cnp.ndarray[DTYPE_t, ndim=1] fitness = self.evaluate_bacteria()
        cdef cnp.ndarray[cnp.intp_t, ndim=1] sorted_indices = np.argsort(fitness)
        cdef int num_eliminate = int(self.elimination_ratio * self.population_size)
        cdef int num_reproduce = num_eliminate // 2
        cdef int i, j, idx

        # Update energy levels
        for i in range(self.population_size):
            self.energy_levels[i] = 1 / (1 + fitness[i])

        # Elimination
        for i in range(num_eliminate):
            idx = sorted_indices[self.population_size - 1 - i]
            if self.energy_levels[idx] < self.reproduction_threshold:
                for j in range(self.dim):
                    self.bacteria[idx, j] = self.bounds[j, 0] + (self.bounds[j, 1] - self.bounds[j, 0]) * \
                                            (rand() / RAND_MAX)

        # Reproduction
        for i in range(num_reproduce):
            idx = sorted_indices[i]
            if self.energy_levels[idx] >= self.reproduction_threshold:
                for j in range(self.dim):
                    self.bacteria[sorted_indices[self.population_size - 1 - i], j] = self.bacteria[idx, j]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void migration(self):
        """ Perform migration phase """
        cdef int i, j
        cdef double norm
        for i in range(self.population_size):
            if rand() / RAND_MAX < self.migration_probability:
                norm = 0.0
                for j in range(self.dim):
                    norm += (self.bacteria[i, j] - self.global_best[j]) ** 2
                norm = sqrt(norm)
                if self.energy_levels[i] < self.reproduction_threshold or norm < 1e-3:
                    for j in range(self.dim):
                        self.bacteria[i, j] = self.bounds[j, 0] + (self.bounds[j, 1] - self.bounds[j, 0]) * \
                                             (rand() / RAND_MAX)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple optimize(self):
        """ Run the Bacterial Colony Optimization algorithm """
        self.initialize_bacteria()
        self.global_best = self.bacteria[0].copy()
        cdef cnp.ndarray[DTYPE_t, ndim=1] fitness
        cdef int min_idx, iteration
        cdef double fitness_value

        for iteration in range(self.max_iter):
            fitness = self.evaluate_bacteria()
            min_idx = np.argmin(fitness)
            fitness_value = fitness[min_idx]
            if fitness_value < self.best_value:
                self.best_solution = self.bacteria[min_idx].copy()
                self.best_value = fitness_value
                self.global_best = self.best_solution.copy()

            self.chemotaxis_and_communication(iteration)
            self.elimination_and_reproduction()
            self.migration()

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
