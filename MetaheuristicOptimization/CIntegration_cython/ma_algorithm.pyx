# cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt
from cython cimport boundscheck, wraparound

cdef class MemeticAlgorithm:
    cdef object obj_func
    cdef int dim, pop_size, max_iterations
    cdef double mutation_rate, crossover_rate
    cdef object bounds
    
    # Use `object` for class-level numpy arrays
    cdef object population, fitness, lower_bounds, upper_bounds, best_solution
    cdef double best_fitness
    cdef object history

    def __init__(self, objective_function, int dim=2, bounds=None,
                 int pop_size=20, int max_iterations=100,
                 double mutation_rate=0.1, double crossover_rate=0.7):
        self.obj_func = objective_function
        self.dim = dim
        self.bounds = bounds if bounds is not None else [(-5.0, 5.0)] * dim
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.lower_bounds = np.array([b[0] for b in self.bounds], dtype=np.float64)
        self.upper_bounds = np.array([b[1] for b in self.bounds], dtype=np.float64)
        self.population = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dim))
        self.fitness = np.apply_along_axis(self.obj_func, 1, self.population)

        self.best_solution = None
        self.best_fitness = float("inf")
        self.history = []

    def optimize(self):
        cdef int iteration, i, idx, local_search_count, num_parents, replace_count, current_best_idx
        cdef np.ndarray[np.float64_t, ndim=2] children_array
        cdef np.ndarray[np.int64_t, ndim=1] sorted_indices, parent_indices, worst_indices
        cdef np.ndarray[np.float64_t, ndim=1] new_ind
        cdef double new_fit

        for iteration in range(self.max_iterations):
            # Hill climbing local search
            local_search_count = max(1, int(0.1 * self.pop_size))
            for _ in range(local_search_count):
                idx = np.random.randint(self.pop_size)
                new_ind = self.hill_climbing(np.array(self.population[idx], dtype=np.float64))
                new_fit = self.obj_func(new_ind)
                if new_fit < self.fitness[idx]:
                    self.population[idx] = new_ind
                    self.fitness[idx] = new_fit

            sorted_indices = np.argsort(self.fitness)
            num_parents = max(2, int(self.crossover_rate * self.pop_size))
            parent_indices = sorted_indices[:num_parents]
            children = []

            for i in range(0, num_parents - 1, 2):
                p1 = self.population[parent_indices[i]]
                p2 = self.population[parent_indices[i + 1]]
                offspring = self.crossover(p1, p2)
                children.extend(offspring)

            children_array = np.array(children, dtype=np.float64)

            # Mutation
            for i in range(children_array.shape[0]):
                if np.random.rand() < self.mutation_rate:
                    children_array[i] = self.mutation(children_array[i])

            replace_count = min(children_array.shape[0], self.pop_size)
            worst_indices = np.argsort(-self.fitness)[:replace_count]
            self.population[worst_indices] = children_array[:replace_count]
            for i in range(replace_count):
                self.fitness[worst_indices[i]] = self.obj_func(self.population[worst_indices[i]])

            # Update best solution
            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[current_best_idx]
                self.best_solution = self.population[current_best_idx].copy()

            self.history.append((iteration, self.best_solution.copy()))
            print(f"Iteration {iteration + 1}: Best Fitness = {self.best_fitness}")

        return self.best_solution, self.best_fitness, self.history

    cpdef np.ndarray[np.float64_t, ndim=1] hill_climbing(self, np.ndarray[np.float64_t, ndim=1] initial_solution):
        cdef np.ndarray[np.float64_t, ndim=1] current_solution = initial_solution.copy()
        cdef np.ndarray[np.float64_t, ndim=1] new_solution
        cdef double current_fitness = self.obj_func(current_solution)
        cdef double new_fitness
        cdef int i

        for i in range(100):
            new_solution = np.clip(current_solution + 0.1 * np.random.randn(self.dim),
                                   self.lower_bounds, self.upper_bounds)
            new_fitness = self.obj_func(new_solution)
            if new_fitness < current_fitness:
                current_solution = new_solution
                current_fitness = new_fitness

        return current_solution

    cpdef list crossover(self, np.ndarray[np.float64_t, ndim=1] parent1,
                         np.ndarray[np.float64_t, ndim=1] parent2):
        cdef int point = np.random.randint(1, self.dim)
        offspring1 = np.concatenate([parent1[:point], parent2[point:]])
        offspring2 = np.concatenate([parent2[:point], parent1[point:]])
        return [offspring1, offspring2]

    cpdef np.ndarray[np.float64_t, ndim=1] mutation(self, np.ndarray[np.float64_t, ndim=1] solution):
        mutated = solution + 0.1 * np.random.randn(self.dim)
        return np.clip(mutated, self.lower_bounds, self.upper_bounds)

