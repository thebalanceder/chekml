# cython: boundscheck=False
# cython: wraparound=False
# distutils: language = c++

import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from cython.view cimport array as cvarray

cdef inline double get_cost(dict x):
    return x['cost']

cdef class KeshtelOptimizer:
    cdef public int dim, population_size, max_iter, s_max
    cdef public float p1, p2
    cdef public np.ndarray bounds
    cdef list population, history
    cdef np.ndarray best_solution
    cdef double best_value
    cdef object objective_function

    def __init__(self, objective_function, int dim, bounds, int population_size=30, int max_iter=100,
                 int s_max=4, float p1=0.2, float p2=0.5):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.float64)
        self.population_size = population_size
        self.max_iter = max_iter
        self.s_max = s_max
        self.p1 = p1
        self.p2 = p2

        self.population = []
        self.best_solution = np.zeros(self.dim, dtype=np.float64)
        self.best_value = float("inf")
        self.history = []

    cpdef np.ndarray random_solution(self):
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)

    cpdef void initialize_population(self):
        cdef int i
        cdef dict ind

        self.population = []
        for i in range(self.population_size):
            pos = self.random_solution()
            cost = self.objective_function(pos)
            ind = {'position': pos, 'cost': cost, 'nn': None}
            self.population.append(ind)

        self.population.sort(key=get_cost)

    cpdef dict nearest_neighbor(self, np.ndarray target_position, list population):
        cdef int i, n = len(population)
        cdef double min_dist = 1e20
        cdef double dist
        cdef int min_idx = -1

        for i in range(n):
            dist = np.linalg.norm(target_position - population[i]['position'])
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        return population[min_idx]

    cpdef np.ndarray swirl(self, dict individual, dict neighbor, int s):
        cdef np.ndarray pos = individual['position']
        cdef np.ndarray nn_pos = neighbor['position']
        cdef double swirl_strength = (self.s_max - s + 1) / self.s_max
        cdef np.ndarray new_pos = pos + swirl_strength * (nn_pos - pos) * np.random.uniform(-1, 1, self.dim)
        return np.clip(new_pos, self.bounds[:, 0], self.bounds[:, 1])

    cpdef np.ndarray crossover_middle(self, list p):
        cdef np.ndarray weights = np.random.dirichlet(np.ones(3))
        cdef np.ndarray new_pos = weights[0] * p[0] + weights[1] * p[1] + weights[2] * p[2]
        return np.clip(new_pos, self.bounds[:, 0], self.bounds[:, 1])

    cpdef tuple optimize(self):
        self.initialize_population()

        cdef int m1 = round(self.p1 * self.population_size)
        cdef int m2 = 2 * round((self.p2 * self.population_size) / 2)
        cdef int m3 = self.population_size - (m1 + m2)

        cdef int iteration, k, j, temp_idx, S
        cdef dict target, neighbor
        cdef list rest, lucky_keshtels, pop_m2, pop_m3
        cdef np.ndarray candidate_pos, pos
        cdef double candidate_cost, cost
        cdef list p
        cdef list i_choices
        cdef int i0, i1

        for iteration in range(self.max_iter):
            lucky_keshtels = self.population[:m1]

            for k in range(m1):
                target = lucky_keshtels[k]
                rest = self.population[k+1:]
                target['nn'] = self.nearest_neighbor(target['position'], rest)

                S = 1
                while S <= 2 * self.s_max - 1:
                    candidate_pos = self.swirl(target, target['nn'], S)
                    candidate_cost = self.objective_function(candidate_pos)
                    if candidate_cost < target['cost']:
                        target['position'] = candidate_pos
                        target['cost'] = candidate_cost
                        target['nn'] = self.nearest_neighbor(target['position'], self.population)
                        S = 1
                    else:
                        S += 1

                lucky_keshtels[k] = target

            pop_m2 = []
            for j in range(m2):
                temp_idx = j + m1
                i_choices = [x for x in range(self.population_size) if x != temp_idx]
                i0, i1 = np.random.choice(i_choices, 2, replace=False)

                p = [self.population[temp_idx]['position'],
                     self.population[i0]['position'],
                     self.population[i1]['position']]

                pos = self.crossover_middle(p)
                cost = self.objective_function(pos)
                pop_m2.append({'position': pos, 'cost': cost, 'nn': None})

            pop_m3 = []
            for j in range(m3):
                rnd_pos = self.random_solution()
                pop_m3.append({
                    'position': rnd_pos,
                    'cost': self.objective_function(rnd_pos),
                    'nn': None
                })

            self.population = lucky_keshtels + pop_m2 + pop_m3
            self.population.sort(key=get_cost)

            if self.population[0]['cost'] < self.best_value:
                self.best_solution = self.population[0]['position']
                self.best_value = self.population[0]['cost']

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

