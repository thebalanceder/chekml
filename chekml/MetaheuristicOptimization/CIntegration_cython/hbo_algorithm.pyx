# hbo_algorithm.pyx

import numpy as np
cimport numpy as np
from libc.math cimport fabs

cdef class HeapBasedOptimizer:
    cdef:
        object obj_func
        int dim, pop_size, max_iter
        object lower_bounds  # Changed from buffer type
        object upper_bounds  # Changed from buffer type

    def __init__(self, objective_function, int dim, bounds, int population_size=30, int max_iter=100):
        self.obj_func = objective_function
        self.dim = dim
        self.lower_bounds = np.array([b[0] for b in bounds], dtype=np.float64)
        self.upper_bounds = np.array([b[1] for b in bounds], dtype=np.float64)
        self.pop_size = population_size
        self.max_iter = max_iter

    cdef np.ndarray[np.double_t, ndim=2] initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dim))

    cdef np.ndarray[np.double_t, ndim=1] ensure_bounds(self, np.ndarray[np.double_t, ndim=1] position):
        return np.clip(position, self.lower_bounds, self.upper_bounds)

    def optimize(self):
        cdef:
            np.ndarray[np.double_t, ndim=2] population
            np.ndarray[np.double_t, ndim=1] fitness
            int best_idx, i, col_start, col_end, col_idx
            np.ndarray[np.double_t, ndim=1] best_solution
            double best_value
            list history
            np.ndarray[np.double_t, ndim=1] individual, a, b, c, heap_candidate
            double new_fitness
            object heap_indices  # Changed from typed buffer
            double r1, r2

        population = self.initialize_population()
        fitness = np.array([self.obj_func(ind) for ind in population], dtype=np.float64)

        best_idx = int(np.argmin(fitness))
        best_solution = population[best_idx].copy()
        best_value = fitness[best_idx]
        history = [(best_value, best_solution.copy())]

        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                individual = population[i].copy()
                heap_indices = np.random.choice(self.pop_size, size=3, replace=False)
                a, b, c = population[heap_indices[0]], population[heap_indices[1]], population[heap_indices[2]]

                r1 = np.random.rand()
                r2 = np.random.rand()
                heap_candidate = a + r1 * (b - c)

                col_start = 0
                col_end = self.dim - 1
                if col_end >= col_start:
                    col_idx = np.random.randint(col_start, col_end + 1)
                    individual[col_idx] = heap_candidate[col_idx]

                individual = self.ensure_bounds(individual)
                new_fitness = self.obj_func(individual)

                if new_fitness < fitness[i]:
                    population[i] = individual
                    fitness[i] = new_fitness
                    if new_fitness < best_value:
                        best_solution = individual.copy()
                        best_value = new_fitness

            history.append((best_value, best_solution.copy()))

        return best_solution, best_value, history

