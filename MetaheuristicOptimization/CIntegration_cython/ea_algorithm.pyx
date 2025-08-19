# cython: language_level=3
import numpy as np
cimport numpy as np
from libc.math cimport pow, fabs

cdef class NSGA2:
    cdef public int population_size  # <-- Add this
    cdef public int generations
    cdef public int V
    cdef public int M
    cdef public int mu
    cdef public int mum
    cdef public object bounds
    cdef public object lower_bound     # Add this
    cdef public object upper_bound     # Add this
    cdef public object objective_function  # For Python function
    cdef public int is_scalar
    cdef public list history
    cdef public object best_solution
    cdef public object best_value
    cdef public object population
    def __init__(self, objective_function, int population_size=100, int generations=100,
                 int dim=2, bounds=None, int mu=15, int mum=20):
        self.objective_function = objective_function
        self.population_size = population_size
        self.generations = generations
        self.V = dim
        self.mu = mu
        self.mum = mum
        self.bounds = bounds

        cdef int i
        if bounds is None:
            self.lower_bound = np.zeros(self.V)
            self.upper_bound = np.ones(self.V)
        else:
            self.lower_bound = np.array([b[0] for b in bounds], dtype=np.float64)
            self.upper_bound = np.array([b[1] for b in bounds], dtype=np.float64)

        sample_out = objective_function(np.random.uniform(self.lower_bound, self.upper_bound))
        if isinstance(sample_out, (float, int)):
            self.M = 1
            self.is_scalar = True
        else:
            self.M = len(sample_out)
            self.is_scalar = False

        self.history = []
        self.best_solution = None
        self.best_value = np.inf if self.is_scalar else np.full(self.M, np.inf)
        self.population = self.initialize_population()

    def initialize_population(self):
        pop = np.random.uniform(
            low=self.lower_bound,
            high=self.upper_bound,
            size=(self.population_size, self.V)
        )
        objectives = np.array([self.objective_function(ind) for ind in pop])
        if objectives.ndim == 1:
            objectives = objectives.reshape(-1, 1)
        return np.hstack((pop, objectives))

    def evaluate_population(self, np.ndarray[np.float64_t, ndim=2] pop):
        X = pop[:, :self.V]
        objectives = np.array([self.objective_function(ind) for ind in X])
        if objectives.ndim == 1:
            objectives = objectives.reshape(-1, 1)
        return np.hstack((X, objectives))

    def dominates(self, np.ndarray[np.float64_t, ndim=1] ind1,
                        np.ndarray[np.float64_t, ndim=1] ind2):
        cdef np.ndarray[np.float64_t, ndim=1] obj1 = ind1[self.V:]
        cdef np.ndarray[np.float64_t, ndim=1] obj2 = ind2[self.V:]
        if self.is_scalar:
            return obj1[0] < obj2[0]
        else:
            return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def fast_non_dominated_sort(self, population):
        cdef int p, q, i
        S = [[] for _ in range(len(population))]
        front = [[]]
        n = [0] * len(population)
        rank = [0] * len(population)

        for p in range(len(population)):
            S[p] = []
            n[p] = 0
            for q in range(len(population)):
                if self.dominates(population[p], population[q]):
                    S[p].append(q)
                elif self.dominates(population[q], population[p]):
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 0
                front[0].append(p)

        i = 0
        while front[i]:
            Q = []
            for p in front[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        Q.append(q)
            i += 1
            front.append(Q)

        front.pop()
        return front

    def crowding_distance(self, np.ndarray[np.float64_t, ndim=2] front):
        cdef int num_individuals = front.shape[0]
        cdef np.ndarray[np.float64_t, ndim=1] distance = np.zeros((num_individuals,))
        cdef int m, i
        if num_individuals == 0:
            return distance

        cdef np.ndarray[np.float64_t, ndim=2] front_obj = front[:, self.V:]
        for m in range(self.M):
            obj_vals = front_obj[:, m]
            sorted_idx = np.argsort(obj_vals)
            distance[sorted_idx[0]] = distance[sorted_idx[-1]] = np.inf
            obj_min = obj_vals[sorted_idx[0]]
            obj_max = obj_vals[sorted_idx[-1]]
            if obj_max - obj_min == 0:
                continue
            for i in range(1, num_individuals - 1):
                distance[sorted_idx[i]] += (
                    obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]
                ) / (obj_max - obj_min)
        return distance

    def tournament_selection(self, population, ranks, distances):
        idx1, idx2 = np.random.randint(len(population), size=2)
        if ranks[idx1] < ranks[idx2]:
            return population[idx1]
        elif ranks[idx1] > ranks[idx2]:
            return population[idx2]
        elif distances[idx1] > distances[idx2]:
            return population[idx1]
        else:
            return population[idx2]

    def sbx_crossover(self, p1, p2):
        cdef np.ndarray[np.float64_t, ndim=1] child1 = np.empty_like(p1)
        cdef np.ndarray[np.float64_t, ndim=1] child2 = np.empty_like(p2)
        cdef float u, beta
        cdef int i

        for i in range(self.V):
            u = np.random.rand()
            if u <= 0.5:
                beta = pow(2 * u, 1.0 / (self.mu + 1))
            else:
                beta = pow(1 / (2 * (1 - u)), 1.0 / (self.mu + 1))
            child1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
            child2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
        return (
            np.clip(child1, self.lower_bound, self.upper_bound),
            np.clip(child2, self.lower_bound, self.upper_bound)
        )

    def polynomial_mutation(self, child):
        cdef int i
        cdef float r, delta
        for i in range(self.V):
            r = np.random.rand()
            if r < 0.5:
                delta = pow(2 * r, 1.0 / (self.mum + 1)) - 1
            else:
                delta = 1 - pow(2 * (1 - r), 1.0 / (self.mum + 1))
            child[i] += delta
        return np.clip(child, self.lower_bound, self.upper_bound)

    def genetic_operator(self, population, ranks, distances):
        children = []
        while len(children) < self.population_size:
            if np.random.rand() < 0.9:
                p1 = self.tournament_selection(population, ranks, distances)
                p2 = self.tournament_selection(population, ranks, distances)
                c1, c2 = self.sbx_crossover(p1[:self.V], p2[:self.V])
                children.append(c1)
                if len(children) < self.population_size:
                    children.append(c2)
            else:
                p = self.tournament_selection(population, ranks, distances)
                c = self.polynomial_mutation(p[:self.V])
                children.append(c)
        return self.evaluate_population(np.array(children))

    def optimize(self):
        cdef int g, i, j
        pop = self.population
        for g in range(self.generations):
            fronts = self.fast_non_dominated_sort(pop)
            ranks = np.zeros(len(pop))
            distances = np.zeros(len(pop))

            for i, front in enumerate(fronts):
                front_array = pop[front]
                dist = self.crowding_distance(front_array)
                for j, idx in enumerate(front):
                    ranks[idx] = i
                    distances[idx] = dist[j]

            offspring = self.genetic_operator(pop, ranks, distances)
            combined = np.vstack((pop, offspring))
            fronts = self.fast_non_dominated_sort(combined)

            next_gen = []
            for front in fronts:
                front_array = combined[front]
                if len(next_gen) + len(front) > self.population_size:
                    dist = self.crowding_distance(front_array)
                    sorted_idx = np.argsort(-dist)
                    for idx in sorted_idx:
                        if len(next_gen) < self.population_size:
                            next_gen.append(front_array[idx])
                else:
                    next_gen.extend(front_array)
                if len(next_gen) >= self.population_size:
                    break

            pop = np.array(next_gen)

            if self.is_scalar:
                best_idx = np.argmin(pop[:, self.V])
                best_val = pop[best_idx, self.V]
                best_sol = pop[best_idx, :self.V]
                self.history.append([best_sol, best_val])
                if best_val < self.best_value:
                    self.best_value = best_val
                    self.best_solution = best_sol
            else:
                self.history.append([best_sol, best_val])

            print(f"Generation {g + 1} completed")

        return self.best_solution, self.best_value, self.history

