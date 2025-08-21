# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
from scipy.special import gamma
from libc.math cimport sin, cos, exp, fabs, sqrt, M_PI

cdef class SpiderWaspOptimizer:
    cdef object objective_function
    cdef int dim
    cdef cnp.ndarray bounds
    cdef int population_size
    cdef int max_iter
    cdef double trade_off
    cdef double crossover_prob
    cdef int min_population
    cdef cnp.ndarray positions
    cdef cnp.ndarray best_solution
    cdef double best_score
    cdef cnp.ndarray convergence_curve
    cdef cnp.ndarray fitness

    def __init__(self, object objective_function, int dim, list bounds, 
                 int population_size=100, int max_iter=50000,
                 double trade_off=0.3, double crossover_prob=0.2, 
                 int min_population=20):
        """
        Initialize the Spider Wasp Optimizer (SWO).

        Parameters:
        - objective_function: Function to optimize (minimization problem).
        - dim: Number of dimensions (variables).
        - bounds: List of (lower, upper) bounds for each dimension.
        - population_size: Number of spider wasps (solutions).
        - max_iter: Maximum number of iterations/function evaluations.
        - trade_off: Trade-off probability between hunting and mating behaviors (TR).
        - crossover_prob: Crossover probability for mating behavior (Cr).
        - min_population: Minimum population size for reduction (N_min).
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.double)
        self.population_size = population_size
        self.max_iter = max_iter
        self.trade_off = trade_off
        self.crossover_prob = crossover_prob
        self.min_population = min_population
        self.positions = None
        self.best_solution = None
        self.best_score = float("inf")
        self.convergence_curve = np.zeros(max_iter, dtype=np.double)
        self.fitness = None

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_positions(self):
        """Initialize the positions of spider wasps randomly within bounds."""
        cdef cnp.ndarray[cnp.double_t, ndim=1] lb = self.bounds[:, 0]
        cdef cnp.ndarray[cnp.double_t, ndim=1] ub = self.bounds[:, 1]
        cdef int i
        if lb.shape[0] == 1:  # Single bound for all dimensions
            self.positions = np.random.rand(self.population_size, self.dim) * (ub[0] - lb[0]) + lb[0]
        else:  # Different bounds for each dimension
            self.positions = np.zeros((self.population_size, self.dim), dtype=np.double)
            for i in range(self.dim):
                self.positions[:, i] = np.random.rand(self.population_size) * (ub[i] - lb[i]) + lb[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.double_t, ndim=1] evaluate_positions(self):
        """Compute fitness values for the spider wasp positions."""
        cdef cnp.ndarray[cnp.double_t, ndim=1] fitness = np.zeros(self.population_size, dtype=np.double)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.positions[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double levy_flight_scalar(self):
        """Generate a single Levy flight sample."""
        cdef double beta = 3.0 / 2.0
        cdef double sigma = (gamma(1 + beta) * sin(M_PI * beta / 2) /
                             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        cdef double u = np.random.randn() * sigma
        cdef double v = np.random.randn()
        cdef double step = u / fabs(v) ** (1 / beta)
        return 0.05 * step

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int hunting_behavior(self, int i, int t, cnp.ndarray[cnp.int32_t, ndim=1] JK):
        """Simulate hunting and nesting behavior."""
        cdef double r1 = np.random.rand()
        cdef double r2 = np.random.rand()
        cdef double r3 = np.random.rand()
        cdef double p = np.random.rand()
        cdef double a = 2.0 - 2.0 * (t / self.max_iter)
        cdef double a2 = -1.0 - (t / self.max_iter)
        cdef double k = 1.0 - (t / self.max_iter)
        cdef double C = a * (2.0 * r1 - 1.0)
        cdef double l = (a2 - 1.0) * np.random.rand() + 1.0
        cdef double L = self.levy_flight_scalar()
        cdef cnp.ndarray[cnp.double_t, ndim=1] vc = np.random.uniform(-k, k, self.dim)
        cdef double rn1 = np.random.randn()
        cdef cnp.ndarray[cnp.double_t, ndim=1] original_pos = self.positions[i].copy()
        cdef cnp.ndarray[cnp.double_t, ndim=1] new_pos = self.positions[i].copy()
        cdef int j
        cdef double m1, m2, B

        for j in range(self.dim):
            if i < k * self.population_size:
                if p < (1.0 - t / self.max_iter):  # Searching stage (Exploration)
                    if r1 < r2:
                        m1 = fabs(rn1) * r1
                        new_pos[j] = new_pos[j] + m1 * (self.positions[JK[0], j] - self.positions[JK[1], j])
                    else:
                        B = 1.0 / (1.0 + exp(l))
                        m2 = B * cos(l * 2.0 * M_PI)
                        new_pos[j] = self.positions[JK[i], j] + m2 * (self.bounds[j, 0] + np.random.rand() * (self.bounds[j, 1] - self.bounds[j, 0]))
                else:  # Following and escaping stage
                    if r1 < r2:
                        new_pos[j] = new_pos[j] + C * fabs(2.0 * np.random.rand() * self.positions[JK[2], j] - new_pos[j])
                    else:
                        new_pos[j] = new_pos[j] * vc[j]
            else:
                if r1 < r2:
                    new_pos[j] = self.best_solution[j] + cos(2.0 * l * M_PI) * (self.best_solution[j] - new_pos[j])
                else:
                    new_pos[j] = self.positions[JK[0], j] + r3 * fabs(L) * (self.positions[JK[0], j] - new_pos[j]) + \
                                 (1.0 - r3) * (np.random.rand() > np.random.rand()) * (self.positions[JK[2], j] - self.positions[JK[1], j])

        # Bound checking
        for j in range(self.dim):
            if new_pos[j] < self.bounds[j, 0]:
                new_pos[j] = self.bounds[j, 0]
            elif new_pos[j] > self.bounds[j, 1]:
                new_pos[j] = self.bounds[j, 1]

        cdef double new_fitness = self.objective_function(new_pos)

        # Update position and best solution
        if new_fitness < self.fitness[i]:
            self.fitness[i] = new_fitness
            self.positions[i] = new_pos
            if new_fitness < self.best_score:
                self.best_score = new_fitness
                self.best_solution = new_pos.copy()
        else:
            self.positions[i] = original_pos

        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int mating_behavior(self, int i, int t, cnp.ndarray[cnp.int32_t, ndim=1] JK):
        """Simulate mating behavior."""
        cdef double a2 = -1.0 - (t / self.max_iter)
        cdef double l = (a2 - 1.0) * np.random.rand() + 1.0
        cdef cnp.ndarray[cnp.double_t, ndim=1] original_pos = self.positions[i].copy()
        cdef cnp.ndarray[cnp.double_t, ndim=1] new_pos = self.positions[i].copy()
        cdef cnp.ndarray[cnp.double_t, ndim=1] v1, v2
        cdef cnp.ndarray[cnp.double_t, ndim=1] male_pos = np.zeros(self.dim, dtype=np.double)
        cdef double rn1 = np.random.randn()
        cdef double rn2 = np.random.randn()
        cdef int j

        # Step sizes for male spider wasp
        if self.fitness[JK[0]] < self.fitness[i]:
            v1 = self.positions[JK[0]] - self.positions[i]
        else:
            v1 = self.positions[i] - self.positions[JK[0]]

        if self.fitness[JK[1]] < self.fitness[JK[2]]:
            v2 = self.positions[JK[1]] - self.positions[JK[2]]
        else:
            v2 = self.positions[JK[2]] - self.positions[JK[1]]

        for j in range(self.dim):
            male_pos[j] = new_pos[j] + (exp(l)) * fabs(rn1) * v1[j] + (1.0 - exp(l)) * fabs(rn2) * v2[j]
            if np.random.rand() < self.crossover_prob:
                new_pos[j] = male_pos[j]

        # Bound checking
        for j in range(self.dim):
            if new_pos[j] < self.bounds[j, 0]:
                new_pos[j] = self.bounds[j, 0]
            elif new_pos[j] > self.bounds[j, 1]:
                new_pos[j] = self.bounds[j, 1]

        cdef double new_fitness = self.objective_function(new_pos)

        # Update position and best solution
        if new_fitness < self.fitness[i]:
            self.fitness[i] = new_fitness
            self.positions[i] = new_pos
            if new_fitness < self.best_score:
                self.best_score = new_fitness
                self.best_solution = new_pos.copy()
        else:
            self.positions[i] = original_pos

        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """Run the Spider Wasp Optimization algorithm."""
        self.initialize_positions()
        self.fitness = self.evaluate_positions()
        cdef int min_idx = np.argmin(self.fitness)
        self.best_score = self.fitness[min_idx]
        self.best_solution = self.positions[min_idx].copy()

        cdef int t = 0
        cdef int current_population = self.population_size
        cdef cnp.ndarray[cnp.int32_t, ndim=1] JK
        cdef int i

        while t < self.max_iter:
            JK = np.random.permutation(current_population).astype(np.int32)

            if np.random.rand() < self.trade_off:  # Hunting and nesting behavior
                for i in range(current_population):
                    t += self.hunting_behavior(i, t, JK)
                    if t >= self.max_iter:
                        break
                    self.convergence_curve[t] = self.best_score
            else:  # Mating behavior
                for i in range(current_population):
                    t += self.mating_behavior(i, t, JK)
                    if t >= self.max_iter:
                        break
                    self.convergence_curve[t] = self.best_score

            # Population reduction
            current_population = int(self.min_population + (self.population_size - self.min_population) * ((self.max_iter - t) / self.max_iter))
            if current_population < self.min_population:
                current_population = self.min_population

            # Sort and trim population
            if current_population < self.positions.shape[0]:
                sorted_indices = np.argsort(self.fitness)[:current_population]
                self.positions = self.positions[sorted_indices]
                self.fitness = self.fitness[sorted_indices]

        self.convergence_curve[t - 1] = self.best_score
        return self.best_solution, self.best_score, self.convergence_curve
