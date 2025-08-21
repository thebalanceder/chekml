# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport exp, ceil

cdef class GreatWallConstructionOptimizer:
    # Declare attributes with compatible types
    cdef public object objective_function
    cdef public int dim, population_size, max_iter, runs
    cdef public object bounds  # as object, not np.ndarray with buffer
    cdef public object best_solution
    cdef public double best_value
    cdef public list history

    def __init__(self, objective_function, int dim, bounds, int population_size=30, int max_iter=500, int runs=1):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.float64)
        self.population_size = population_size
        self.max_iter = max_iter
        self.runs = runs
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    cpdef np.ndarray initialize_population(self):
        cdef np.ndarray[np.float64_t, ndim=2] bounds_arr = self.bounds
        cdef np.ndarray[np.float64_t, ndim=1] lb = bounds_arr[:, 0]
        cdef np.ndarray[np.float64_t, ndim=1] ub = bounds_arr[:, 1]
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    cpdef np.ndarray evaluate_population(self, np.ndarray[np.float64_t, ndim=2] population):
        return np.array([self.objective_function(ind) for ind in population])

    cpdef tuple optimize(self):
        cdef double g = 9.8
        cdef double m = 3
        cdef double e = 0.1
        cdef double P = 9
        cdef double Q = 6
        cdef double Cmax = exp(3)
        cdef double Cmin = exp(2)
        cdef int run, i, t, LNP
        cdef double r1, r2, C, F, new_fit

        cdef np.ndarray[np.float64_t, ndim=2] bounds_arr = self.bounds
        cdef np.ndarray[np.float64_t, ndim=1] lb = bounds_arr[:, 0]
        cdef np.ndarray[np.float64_t, ndim=1] ub = bounds_arr[:, 1]
        cdef double best_overall = float("inf")
        cdef np.ndarray[np.float64_t, ndim=1] best_solution_overall = np.empty(self.dim, dtype=np.float64)

        for run in range(self.runs):
            population = self.initialize_population()
            fitness = self.evaluate_population(population)

            sorted_indices = np.argsort(fitness)
            Worker1 = population[sorted_indices[0]].copy()
            Worker2 = population[sorted_indices[1]].copy()
            Worker3 = population[sorted_indices[2]].copy()
            Worker1_fit = fitness[sorted_indices[0]]
            Worker2_fit = fitness[sorted_indices[1]]
            Worker3_fit = fitness[sorted_indices[2]]

            LNP = <int>ceil(self.population_size * e)

            for t in range(1, self.max_iter + 1):
                C = Cmax - ((Cmax - Cmin) * t / self.max_iter)

                for i in range(self.population_size):
                    r1 = np.random.rand()
                    r2 = np.random.rand()

                    if i < LNP:
                        F = (m * g * r1) / (P * Q * (1 + t))
                        new_pos = population[i] + F * np.sign(np.random.randn(self.dim)) * C
                    else:
                        influence = (Worker1 + Worker2 + Worker3) / 3 - population[i]
                        new_pos = population[i] + r2 * influence * C

                    new_pos = np.clip(new_pos, lb, ub)
                    new_fit = self.objective_function(new_pos)

                    if new_fit < fitness[i]:
                        population[i] = new_pos
                        fitness[i] = new_fit

                        if new_fit < Worker1_fit:
                            Worker3, Worker3_fit = Worker2.copy(), Worker2_fit
                            Worker2, Worker2_fit = Worker1.copy(), Worker1_fit
                            Worker1, Worker1_fit = new_pos.copy(), new_fit

                if Worker1_fit < best_overall:
                    best_overall = Worker1_fit
                    best_solution_overall = Worker1.copy()

                self.history.append((t, Worker1_fit))
                print(f"Iteration {t}: Best Value = {Worker1_fit}")

        self.best_solution = best_solution_overall
        self.best_value = best_overall
        return self.best_solution, self.best_value, self.history

