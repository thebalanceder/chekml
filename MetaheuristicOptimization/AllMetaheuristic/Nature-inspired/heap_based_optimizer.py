import numpy as np

class HeapBasedOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=30, max_iter=100):
        self.obj_func = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)
        self.pop_size = population_size
        self.max_iter = max_iter

        self.lower_bounds = self.bounds[:, 0]
        self.upper_bounds = self.bounds[:, 1]

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dim))

    def ensure_bounds(self, position):
        return np.clip(position, self.lower_bounds, self.upper_bounds)

    def optimize(self):
        population = self.initialize_population()
        fitness = np.apply_along_axis(self.obj_func, 1, population)

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_value = fitness[best_idx]
        history = [(best_value, best_solution.copy())]

        for t in range(self.max_iter):
            new_population = []

            for i in range(self.pop_size):
                individual = population[i].copy()
                heap_indices = np.random.choice(self.pop_size, size=3, replace=False)
                a, b, c = population[heap_indices]

                r1, r2 = np.random.rand(), np.random.rand()
                heap_candidate = a + r1 * (b - c)

                # Optional heap-based dimension mixing
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

