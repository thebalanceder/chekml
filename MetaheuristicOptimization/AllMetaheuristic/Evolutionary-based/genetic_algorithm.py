import numpy as np

class GeneticAlgorithm:
    def __init__(self, objective_function, dim, bounds, num_pop=5, num_iter=100, mode='min'):
        self.obj_func = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)
        self.num_pop = num_pop
        self.num_iter = num_iter
        self.mode = mode.lower()

        self.lower = self.bounds[:, 0]
        self.upper = self.bounds[:, 1]

        self.population = np.random.uniform(self.lower, self.upper, (self.num_pop, self.dim))
        self.fitness = np.full(self.num_pop, np.inf if self.mode == 'min' else -np.inf)

        self.best_solution = None
        self.best_fitness = np.inf if self.mode == 'min' else -np.inf
        self.history = []

    def optimize(self):
        for pop_index in range(self.num_pop):
            print(f"- IndexPop: {pop_index + 1} out of {self.num_pop}")
            # Random initialization for current population
            population = np.random.uniform(self.lower, self.upper, (self.num_pop, self.dim))

            for iteration in range(self.num_iter):
                print(f"-- IndexIter: {iteration + 1} out of {self.num_iter}")

                # Evaluate objective function
                for i in range(self.num_pop):
                    print(f"--- IndexChr: {i + 1} out of {self.num_pop}")
                    self.fitness[i] = self.obj_func(population[i])

                # Selection (sort by fitness)
                indices = np.argsort(self.fitness) if self.mode == 'min' else np.argsort(-self.fitness)
                best_idx = indices[0]
                current_best_fitness = self.fitness[best_idx]
                current_best_solution = population[best_idx].copy()

                # Print best
                print(f"-- ObjFuncbest: {current_best_fitness:.10e}, " + ", ".join([f"Xcbest({i}) = {x:.4f}" for i, x in enumerate(current_best_solution)]))

                # Crossover (uniform)
                for i in range(self.num_pop):
                    for d in range(self.dim):
                        if i != indices[0] and i != indices[1] and np.random.rand() < 0.5:
                            parent_idx = np.random.choice([indices[0], indices[1]])
                            population[i][d] = population[parent_idx][d]

                # Mutation (last chromosome)
                for d in range(self.dim):
                    if np.random.rand() < 0.5:
                        population[indices[-1]][d] = np.random.uniform(self.lower[d], self.upper[d])

                # Track best across all generations
                if ((self.mode == 'min' and current_best_fitness < self.best_fitness) or
                    (self.mode == 'max' and current_best_fitness > self.best_fitness)):
                    self.best_fitness = current_best_fitness
                    self.best_solution = current_best_solution

                self.history.append((iteration, self.best_solution.copy(), self.best_fitness))

            print(f"-  ObjFunpbest: {self.best_fitness:.10e}, " + ", ".join([f"Xpbest({i}) = {x:.4f}" for i, x in enumerate(self.best_solution)]))

        return self.best_solution, self.best_fitness, self.history
