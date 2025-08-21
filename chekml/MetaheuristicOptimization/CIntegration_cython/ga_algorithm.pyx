# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, pi, sqrt, fabs
from libc.stdlib cimport rand, srand
from libc.time cimport time
cimport cython

cdef class GeneticAlgorithm:
    cdef public int dim, num_pop, num_iter
    cdef public str mode
    cdef np.ndarray bounds, lower, upper
    cdef object obj_func, best_solution, history
    cdef double best_fitness

    def __init__(self, objective_function, int dim, bounds, int num_pop=5, int num_iter=100, mode='min'):
        self.obj_func = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.float64)
        self.num_pop = num_pop
        self.num_iter = num_iter
        self.mode = mode.lower()

        self.lower = self.bounds[:, 0]
        self.upper = self.bounds[:, 1]

        self.best_solution = None
        self.best_fitness = np.inf if self.mode == 'min' else -np.inf
        self.history = []

    def optimize(self):
        cdef int pop_index, iteration, i, d, best_idx, parent_idx
        cdef np.ndarray[np.float64_t, ndim=2] population
        cdef np.ndarray[np.float64_t, ndim=1] fitness
        cdef np.ndarray[np.float64_t, ndim=1] current_best_solution
        cdef double current_best_fitness
        cdef np.ndarray[np.intp_t, ndim=1] indices

        # Main population used across all subpopulations
        population = np.random.uniform(self.lower, self.upper, (self.num_pop, self.dim))
        fitness = np.empty(self.num_pop, dtype=np.float64)

        for pop_index in range(1):  # only one GA run, outer loop removed
            print(f"- IndexPop: {pop_index + 1} out of {1}")

            for iteration in range(self.num_iter):
                print(f"-- IndexIter: {iteration + 1} out of {self.num_iter}")

                for i in range(self.num_pop):
                    print(f"--- IndexChr: {i + 1} out of {self.num_pop}")
                    fitness[i] = self.obj_func(population[i])

                indices = np.argsort(fitness) if self.mode == 'min' else np.argsort(-fitness)
                best_idx = indices[0]
                current_best_fitness = fitness[best_idx]
                current_best_solution = population[best_idx].copy()

                print(f"-- ObjFuncbest: {current_best_fitness:.10e}, " + ", ".join([f"Xcbest({i}) = {x:.4f}" for i, x in enumerate(current_best_solution)]))

                for i in range(self.num_pop):
                    if i != indices[0] and i != indices[1]:
                        for d in range(self.dim):
                            if np.random.rand() < 0.5:
                                parent_idx = np.random.choice([indices[0], indices[1]])
                                population[i][d] = population[parent_idx][d]

                worst_idx = indices[-1] if indices[-1] < self.num_pop else self.num_pop - 1
                for d in range(self.dim):
                    if np.random.rand() < 0.5:
                        population[worst_idx][d] = np.random.uniform(self.lower[d], self.upper[d])

                if ((self.mode == 'min' and current_best_fitness < self.best_fitness) or
                    (self.mode == 'max' and current_best_fitness > self.best_fitness)):
                    self.best_fitness = current_best_fitness
                    self.best_solution = current_best_solution.copy()

                self.history.append((iteration, self.best_solution.copy(), self.best_fitness))

            print(f"-  ObjFunpbest: {self.best_fitness:.10e}, " + ", ".join([f"Xpbest({i}) = {x:.4f}" for i, x in enumerate(self.best_solution)]))

        return self.best_solution, self.best_fitness, self.history

