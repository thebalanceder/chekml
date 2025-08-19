import numpy as np

class MemeticAlgorithm:
    def __init__(self, objective_function, dim=2, bounds=None,
                 pop_size=20, max_iterations=100,
                 mutation_rate=0.1, crossover_rate=0.7):
        """
        Initialize the Memetic Algorithm optimizer.

        Parameters:
        - objective_function: Function to be minimized.
        - dim: Number of decision variables.
        - bounds: List of (lower, upper) tuples for each dimension.
        - pop_size: Size of the population.
        - max_iterations: Maximum number of iterations.
        - mutation_rate: Probability of mutation.
        - crossover_rate: Proportion of population used for crossover.
        """
        self.obj_func = objective_function
        self.dim = dim
        self.bounds = bounds if bounds else [(-5, 5)] * dim
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.lower_bounds = np.array([b[0] for b in self.bounds])
        self.upper_bounds = np.array([b[1] for b in self.bounds])

        self.population = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dim))
        self.fitness = np.apply_along_axis(self.obj_func, 1, self.population)

        self.best_solution = None
        self.best_fitness = float("inf")
        self.history = []

    def optimize(self):
        for iteration in range(self.max_iterations):
            # Local search (Hill Climbing)
            local_search_count = max(1, int(0.1 * self.pop_size))
            for _ in range(local_search_count):
                idx = np.random.randint(self.pop_size)
                new_ind = self.hill_climbing(self.population[idx])
                new_fit = self.obj_func(new_ind)
                if new_fit < self.fitness[idx]:
                    self.population[idx] = new_ind
                    self.fitness[idx] = new_fit

            # Crossover
            sorted_indices = np.argsort(self.fitness)
            num_parents = max(2, int(self.crossover_rate * self.pop_size))
            parent_indices = sorted_indices[:num_parents]
            children = []

            for i in range(0, num_parents - 1, 2):
                p1 = self.population[parent_indices[i]]
                p2 = self.population[parent_indices[i + 1]]
                offspring = self.crossover(p1, p2)
                children.extend(offspring)

            children = np.array(children)

            # Mutation
            for i in range(len(children)):
                if np.random.rand() < self.mutation_rate:
                    children[i] = self.mutation(children[i])

            # Replace worst individuals
            replace_count = min(len(children), self.pop_size)
            worst_indices = np.argsort(-self.fitness)[:replace_count]
            self.population[worst_indices] = children[:replace_count]

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

    def hill_climbing(self, initial_solution):
        current_solution = initial_solution.copy()
        current_fitness = self.obj_func(current_solution)

        for _ in range(100):
            perturbation = 0.1 * np.random.randn(self.dim)
            new_solution = np.clip(current_solution + perturbation, self.lower_bounds, self.upper_bounds)
            new_fitness = self.obj_func(new_solution)

            if new_fitness < current_fitness:
                current_solution = new_solution
                current_fitness = new_fitness

        return current_solution

    def crossover(self, parent1, parent2):
        point = np.random.randint(1, self.dim)
        offspring1 = np.concatenate([parent1[:point], parent2[point:]])
        offspring2 = np.concatenate([parent2[:point], parent1[point:]])
        return [offspring1, offspring2]

    def mutation(self, solution):
        mutated = solution + 0.1 * np.random.randn(self.dim)
        return np.clip(mutated, self.lower_bounds, self.upper_bounds)

