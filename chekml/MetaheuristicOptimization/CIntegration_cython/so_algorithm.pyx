import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport cos, sin, M_PI

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class SpiralOptimizationAlgorithm:
    cdef public:
        object obj_func
        int dim
        cnp.ndarray bounds
        int pop_size
        int max_iter
        double c
        double alpha
        list history

    def __init__(self, object objective_function, int dim, bounds, int population_size=50, int max_iter=100, double c=0.1, double alpha=1):
        self.obj_func = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.double)
        self.pop_size = population_size
        self.max_iter = max_iter
        self.c = c
        self.alpha = alpha
        self.history = []

    def optimize(self):
        cdef cnp.ndarray[cnp.double_t, ndim=2] population
        cdef cnp.ndarray[cnp.double_t, ndim=1] fitness
        cdef cnp.ndarray[cnp.double_t, ndim=1] lb
        cdef cnp.ndarray[cnp.double_t, ndim=1] ub
        cdef cnp.ndarray[cnp.double_t, ndim=1] best_solution
        cdef cnp.ndarray[cnp.double_t, ndim=2] r
        cdef cnp.ndarray[cnp.double_t, ndim=2] theta
        cdef cnp.ndarray[cnp.double_t, ndim=2] direction
        cdef cnp.ndarray[cnp.double_t, ndim=2] step
        cdef cnp.ndarray[cnp.double_t, ndim=2] new_position
        cdef cnp.ndarray[cnp.double_t, ndim=1] new_fitness
        cdef cnp.ndarray[cnp.intp_t, ndim=1] sorted_indices
        cdef double best_fitness
        cdef int best_idx, min_idx, i, j, k

        # Initialize population
        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]
        population = lb + np.random.rand(self.pop_size, self.dim) * (ub - lb)
        fitness = np.apply_along_axis(self.obj_func, 1, population)

        # Identify best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for i in range(self.max_iter):
            # Spiral movement
            r = np.random.rand(self.pop_size, 1)
            theta = 2 * M_PI * np.random.rand(self.pop_size, 1)
            direction = np.zeros((self.pop_size, self.dim), dtype=np.double)
            for j in range(self.pop_size):
                direction[j, 0] = cos(theta[j, 0])
                direction[j, 1] = sin(theta[j, 0])

            step = self.c * r * direction
            new_position = population + step

            # Keep within bounds
            for j in range(self.pop_size):
                for k in range(self.dim):
                    if new_position[j, k] < lb[k]:
                        new_position[j, k] = lb[k]
                    elif new_position[j, k] > ub[k]:
                        new_position[j, k] = ub[k]

            # Evaluate new fitness
            new_fitness = np.apply_along_axis(self.obj_func, 1, new_position)

            # Update best solution
            min_idx = np.argmin(new_fitness)
            if new_fitness[min_idx] < best_fitness:
                best_fitness = new_fitness[min_idx]
                best_solution = new_position[min_idx].copy()

            # Sort and select top individuals
            sorted_indices = np.argsort(new_fitness)
            population = new_position[sorted_indices]

            # Save history
            self.history.append((best_fitness, best_solution.copy()))

        return best_solution, best_fitness, self.history
