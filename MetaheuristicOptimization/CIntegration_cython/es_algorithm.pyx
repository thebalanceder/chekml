# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.math cimport sin, pi, pow
from scipy.special import gamma  # Use this instead

cdef class EagleStrategyOptimizer:
    cdef:
        object obj_func
        int dim, population_size, max_iter
        double c1, c2, w_max, w_min
        object bounds, population, velocity
        object local_best, local_best_cost
        object global_best
        double global_best_cost
        object history

    def __init__(self, objective_function, int dim, bounds, int population_size=300, int max_iter=1000,
                 double c1=2.0, double c2=2.0, double w_max=0.9, double w_min=0.4):
        self.obj_func = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.float64)
        self.population_size = population_size
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min

        self.population = None
        self.velocity = None
        self.local_best = None
        self.local_best_cost = None
        self.global_best = None
        self.global_best_cost = float("inf")

        self.history = []

    cpdef void initialize(self):
        cdef np.ndarray[np.float64_t, ndim=1] lb = self.bounds[:, 0]
        cdef np.ndarray[np.float64_t, ndim=1] ub = self.bounds[:, 1]

        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocity = np.random.rand(self.population_size, self.dim)
        self.local_best = self.population.copy()
        self.local_best_cost = np.array([self.obj_func(ind) for ind in self.population], dtype=np.float64)

        cdef int best_idx = int(np.argmin(self.local_best_cost))
        self.global_best = self.local_best[best_idx].copy()
        self.global_best_cost = self.local_best_cost[best_idx]

    cpdef void update_velocity_position(self, int iter_):
        cdef np.ndarray[np.float64_t, ndim=2] r1 = np.random.rand(self.population_size, self.dim)
        cdef np.ndarray[np.float64_t, ndim=2] r2 = np.random.rand(self.population_size, self.dim)

        cdef double w = self.w_max - ((self.w_max - self.w_min) * iter_ / self.max_iter)

        cognitive = self.c1 * r1 * (self.local_best - self.population)
        social = self.c2 * r2 * (self.global_best - self.population)
        self.velocity = w * self.velocity + cognitive + social
        self.population += self.velocity

        cdef np.ndarray[np.float64_t, ndim=1] lb = self.bounds[:, 0]
        cdef np.ndarray[np.float64_t, ndim=1] ub = self.bounds[:, 1]
        self.population = np.clip(self.population, lb, ub)

    cpdef np.ndarray[np.float64_t, ndim=2] levy_flight(self):
        cdef double beta = 1.5
        cdef double sigma = pow((gamma(1 + beta) * sin(pi * beta / 2) /
                                (gamma((1 + beta) / 2) * beta * pow(2, (beta - 1) / 2))), 1 / beta)

        cdef np.ndarray[np.float64_t, ndim=2] steps = np.zeros_like(self.population, dtype=np.float64)
        cdef int i

        for i in range(self.population_size):
            u = np.random.randn(self.dim) * sigma
            v = np.random.rand(self.dim)
            step = u / np.power(np.abs(v), 1 / beta)
            steps[i] = 0.1 * step * (self.local_best[i] - self.global_best)

        s = self.local_best + steps

        cdef np.ndarray[np.float64_t, ndim=1] lb = self.bounds[:, 0]
        cdef np.ndarray[np.float64_t, ndim=1] ub = self.bounds[:, 1]
        s = np.clip(s, lb, ub)

        return s

    cpdef tuple optimize(self):
        cdef int iter_, i
        cdef np.ndarray[np.float64_t, ndim=1] fitness, s_cost
        cdef int best_idx, new_best_idx
        cdef np.ndarray[np.float64_t, ndim=2] s

        self.initialize()

        for iter_ in range(self.max_iter):
            if np.random.rand() < 0.2:
                s = self.levy_flight()
                s_cost = np.array([self.obj_func(ind) for ind in s], dtype=np.float64)
                new_best_idx = int(np.argmin(s_cost))
                if s_cost[new_best_idx] < self.global_best_cost:
                    self.global_best = s[new_best_idx].copy()
                    self.global_best_cost = s_cost[new_best_idx]
                continue

            self.update_velocity_position(iter_)
            fitness = np.array([self.obj_func(ind) for ind in self.population], dtype=np.float64)

            for i in range(self.population_size):
                if fitness[i] < self.local_best_cost[i]:
                    self.local_best[i] = self.population[i].copy()
                    self.local_best_cost[i] = fitness[i]

            best_idx = int(np.argmin(self.local_best_cost))
            if self.local_best_cost[best_idx] < self.global_best_cost:
                self.global_best = self.local_best[best_idx].copy()
                self.global_best_cost = self.local_best_cost[best_idx]

            self.history.append((iter_, self.global_best.copy(), self.global_best_cost))
            print(f"Iteration {iter_ + 1}: Best Cost = {self.global_best_cost:.6f}")

        return self.global_best, self.global_best_cost, self.history

