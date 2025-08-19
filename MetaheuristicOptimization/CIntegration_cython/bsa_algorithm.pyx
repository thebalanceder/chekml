# distutils: language = c++

import numpy as np
cimport numpy as np
from libc.math cimport ceil

cdef class BacktrackingSearchAlgorithm:
    cdef:
        object objective_function
        int dim, pop_size, max_iter
        double dim_rate
        object bounds, population, historical_pop, fitness, best_solution, history
        double best_value

    def __init__(self, object objective_function, int dim, bounds,
                 int pop_size=50, double dim_rate=0.5, int max_iter=100):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.float64)
        self.pop_size = pop_size
        self.dim_rate = dim_rate
        self.max_iter = max_iter
        self.best_value = float("inf")
        self.history = []

    cdef np.ndarray[np.float64_t, ndim=2] generate_population(self):
        cdef np.ndarray[np.float64_t, ndim=2] result
        cdef np.ndarray[np.float64_t, ndim=1] low = self.bounds[:, 0]
        cdef np.ndarray[np.float64_t, ndim=1] high = self.bounds[:, 1]
        result = np.random.uniform(low, high, (self.pop_size, self.dim)).astype(np.float64)
        return result

    cdef np.ndarray[np.float64_t, ndim=2] boundary_control(self, np.ndarray[np.float64_t, ndim=2] pop):
        cdef int i, j
        cdef np.ndarray[np.float64_t, ndim=1] low = self.bounds[:, 0]
        cdef np.ndarray[np.float64_t, ndim=1] high = self.bounds[:, 1]
        cdef bint k
        for i in range(self.pop_size):
            for j in range(self.dim):
                k = np.random.rand() < np.random.rand()
                if pop[i, j] < low[j]:
                    pop[i, j] = low[j] if k else np.random.uniform(low[j], high[j])
                elif pop[i, j] > high[j]:
                    pop[i, j] = high[j] if k else np.random.uniform(low[j], high[j])
        return pop

    cdef double get_scale_factor(self):
        return 3 * np.random.randn()

    def optimize(self):
        cdef int epoch, i
        cdef double F
        cdef int best_idx
        cdef np.ndarray[np.float64_t, ndim=2] offspring, map_mask
        cdef np.ndarray[np.float64_t, ndim=1] fitness_offspring

        self.population = self.generate_population()
        self.fitness = np.array([self.objective_function(ind) for ind in self.population], dtype=np.float64)
        self.historical_pop = self.generate_population()
        self.best_solution = np.copy(self.population[0])

        for epoch in range(self.max_iter):
            if np.random.rand() < np.random.rand():
                self.historical_pop = self.population.copy()

            np.random.shuffle(self.historical_pop)
            F = self.get_scale_factor()

            map_mask = np.zeros((self.pop_size, self.dim), dtype=np.float64)
            if np.random.rand() < np.random.rand():
                for i in range(self.pop_size):
                    u = np.random.permutation(self.dim)
                    cnt = int(ceil(self.dim_rate * np.random.rand() * self.dim))
                    map_mask[i, u[:cnt]] = 1
            else:
                for i in range(self.pop_size):
                    map_mask[i, np.random.randint(0, self.dim)] = 1

            offspring = self.population + (map_mask * F) * (self.historical_pop - self.population)
            offspring = self.boundary_control(offspring)
            fitness_offspring = np.array([self.objective_function(ind) for ind in offspring], dtype=np.float64)

            improved = fitness_offspring < self.fitness
            self.fitness[improved] = fitness_offspring[improved]
            self.population[improved] = offspring[improved]

            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_value:
                self.best_value = self.fitness[best_idx]
                self.best_solution = self.population[best_idx].copy()

            self.history.append((epoch, self.best_solution.copy(), self.best_value))
            print(f"Iteration {epoch + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

