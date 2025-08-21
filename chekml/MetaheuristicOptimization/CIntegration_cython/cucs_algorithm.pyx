import numpy as np
cimport numpy as cnp
cimport cython
from scipy.special import gamma
from libc.math cimport sin, pi, fabs, pow

# Type definitions for Cython
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int32_t ITYPE_t

cdef class CuckooSearchOptimizer:
    cdef public:
        object objective_function
        ITYPE_t dim
        cnp.ndarray bounds
        ITYPE_t population_size
        ITYPE_t max_iter
        DTYPE_t pa
        cnp.ndarray nests
        cnp.ndarray best_nest
        DTYPE_t best_value
        list history

    def __init__(self, object objective_function, ITYPE_t dim, object bounds, 
                 ITYPE_t population_size=25, ITYPE_t max_iter=100, DTYPE_t pa=0.25):
        """
        Initialize the Cuckoo Search optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Array of (lower, upper) bounds for each dimension.
        - population_size: Number of nests (solutions), default is 25.
        - max_iter: Maximum number of iterations, default is 1000.
        - pa: Discovery rate of alien eggs/solutions, default is 0.25.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.float64)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.pa = pa
        self.nests = None
        self.best_nest = None
        self.best_value = np.inf
        self.history = []

    cpdef void initialize_nests(self):
        """Generate initial nests randomly."""
        self.nests = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                       (self.population_size, self.dim))

    cpdef cnp.ndarray evaluate_nests(self):
        """Compute fitness values for the nests."""
        cdef cnp.ndarray[DTYPE_t, ndim=1] fitness = np.empty(self.population_size, dtype=np.float64)
        cdef ITYPE_t j
        for j in range(self.population_size):
            fitness[j] = self.objective_function(self.nests[j])
        return fitness

    cpdef cnp.ndarray get_cuckoos(self):
        """Generate new solutions using Levy flights."""
        cdef ITYPE_t n = self.population_size
        cdef DTYPE_t beta = 3.0 / 2.0
        cdef DTYPE_t sigma = (gamma(1 + beta) * sin(pi * beta / 2) /
                              (gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2))) ** (1 / beta)
        cdef cnp.ndarray[DTYPE_t, ndim=2] new_nests = self.nests.copy()
        cdef cnp.ndarray[DTYPE_t, ndim=1] s, u, v, step, stepsize
        cdef ITYPE_t j, k
        cdef DTYPE_t v_abs, step_val

        for j in range(n):
            s = new_nests[j, :]
            u = np.random.randn(self.dim) * sigma
            v = np.random.randn(self.dim)
            step = np.empty(self.dim, dtype=np.float64)
            for k in range(self.dim):
                v_abs = fabs(v[k])
                if v_abs > 0:
                    step_val = u[k] / pow(v_abs, 1.0 / beta)
                else:
                    step_val = u[k]
                step[k] = step_val

            stepsize = 0.01 * step * (s - self.best_nest)
            for k in range(self.dim):
                s[k] += stepsize[k] * np.random.randn()
            
            new_nests[j, :] = self.simplebounds(s)
        return new_nests

    cpdef cnp.ndarray empty_nests(self):
        """Replace some nests with new solutions based on discovery probability."""
        cdef ITYPE_t n = self.population_size
        cdef cnp.ndarray[DTYPE_t, ndim=2] K = (np.random.rand(n, self.dim) > self.pa).astype(np.float64)
        cdef cnp.ndarray[ITYPE_t, ndim=1] idx = np.random.permutation(n).astype(np.int32)
        cdef cnp.ndarray[ITYPE_t, ndim=1] idx2 = np.random.permutation(n).astype(np.int32)
        cdef cnp.ndarray[DTYPE_t, ndim=2] stepsize = np.random.rand() * (self.nests[idx, :] - self.nests[idx2, :])
        cdef cnp.ndarray[DTYPE_t, ndim=2] new_nests = self.nests + stepsize * K
        cdef ITYPE_t j

        for j in range(n):
            new_nests[j, :] = self.simplebounds(new_nests[j, :])
        return new_nests

    cpdef cnp.ndarray simplebounds(self, cnp.ndarray[DTYPE_t, ndim=1] s):
        """Apply bounds to a solution."""
        cdef cnp.ndarray[DTYPE_t, ndim=1] result = s.copy()
        cdef ITYPE_t k
        for k in range(self.dim):
            if result[k] < self.bounds[k, 0]:
                result[k] = self.bounds[k, 0]
            elif result[k] > self.bounds[k, 1]:
                result[k] = self.bounds[k, 1]
        return result

    cpdef tuple optimize(self):
        """Run the Cuckoo Search optimization."""
        self.initialize_nests()
        cdef ITYPE_t N_iter = 0
        cdef ITYPE_t generation, j, min_idx
        cdef cnp.ndarray[DTYPE_t, ndim=1] fitness, new_fitness
        cdef cnp.ndarray[DTYPE_t, ndim=2] new_nests
        cdef DTYPE_t fmin

        for generation in range(self.max_iter):
            fitness = self.evaluate_nests()
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_nest = self.nests[min_idx].copy()
                self.best_value = fitness[min_idx]

            # Generate new solutions via Levy flights
            new_nests = self.get_cuckoos()
            new_fitness = np.empty(self.population_size, dtype=np.float64)
            for j in range(self.population_size):
                new_fitness[j] = self.objective_function(new_nests[j])
                if new_fitness[j] <= fitness[j]:
                    fitness[j] = new_fitness[j]
                    self.nests[j, :] = new_nests[j, :]
            N_iter += self.population_size

            # Discovery and randomization
            new_nests = self.empty_nests()
            for j in range(self.population_size):
                new_fitness[j] = self.objective_function(new_nests[j])
                if new_fitness[j] <= fitness[j]:
                    fitness[j] = new_fitness[j]
                    self.nests[j, :] = new_nests[j, :]
            N_iter += self.population_size

            # Update best if necessary
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_nest = self.nests[min_idx].copy()
                self.best_value = fitness[min_idx]

            self.history.append((generation, self.best_nest.copy(), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")

        print(f"Total number of iterations={N_iter}")
        return self.best_nest, self.best_value, self.history

