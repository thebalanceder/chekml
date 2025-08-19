import numpy as np

class FutureSearchOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=30, max_iter=100, num_runs=30):
        """
        Initialize the Future Search Algorithm optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of solutions in the population.
        - max_iter: Maximum number of iterations.
        - num_runs: Number of independent runs.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.num_runs = num_runs

        self.population = None  # Population of solutions
        self.local_best_positions = None  # Local best positions for each solution
        self.local_best_values = None  # Local best fitness values
        self.global_best_position = None  # Global best position
        self.global_best_value = float("inf")  # Global best fitness value
        self.global_best_history = []  # History of global best values per run
        self.best_positions = []  # Best positions per run

    def initialize_population(self):
        """ Generate initial population randomly within bounds """
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                           (self.population_size, self.dim))
        self.local_best_positions = self.population.copy()
        self.local_best_values = np.array([self.objective_function(ind) for ind in self.population])
        min_idx = np.argmin(self.local_best_values)
        self.global_best_position = self.population[min_idx].copy()
        self.global_best_value = self.local_best_values[min_idx]

    def clip_to_bounds(self, solution):
        """ Ensure solution stays within bounds """
        return np.clip(solution, self.bounds[:, 0], self.bounds[:, 1])

    def update_population(self):
        """ Update population using global and local bests """
        for i in range(self.population_size):
            # Update rule: Move towards global and local bests
            self.population[i] = (self.population[i] + 
                                 (-self.population[i] + self.global_best_position) * np.random.rand() +
                                 (-self.population[i] + self.local_best_positions[i]) * np.random.rand())
            self.population[i] = self.clip_to_bounds(self.population[i])

            # Evaluate new solution
            new_fitness = self.objective_function(self.population[i])

            # Update local best
            if new_fitness <= self.local_best_values[i]:
                self.local_best_positions[i] = self.population[i].copy()
                self.local_best_values[i] = new_fitness

            # Update global best
            if new_fitness <= self.global_best_value:
                self.global_best_position = self.population[i].copy()
                self.global_best_value = new_fitness

    def update_with_initial_strategy(self):
        """ Update solutions using initial strategy based on global and local bests """
        temp_population = np.zeros_like(self.population)
        for i in range(self.population_size):
            temp_population[i] = (self.global_best_position + 
                                 (self.global_best_position - self.local_best_positions[i]) * np.random.rand())
            temp_population[i] = self.clip_to_bounds(temp_population[i])

            # Evaluate new solution
            new_fitness = self.objective_function(temp_population[i])

            # Update if better than previous fitness
            if new_fitness <= self.local_best_values[i]:
                self.population[i] = temp_population[i].copy()
                self.local_best_positions[i] = temp_population[i].copy()
                self.local_best_values[i] = new_fitness

        # Update global best based on temporary population
        temp_fitness = np.array([self.objective_function(ind) for ind in temp_population])
        min_idx = np.argmin(temp_fitness)
        if temp_fitness[min_idx] <= self.global_best_value:
            self.global_best_position = temp_population[min_idx].copy()
            self.global_best_value = temp_fitness[min_idx]

    def optimize(self):
        """ Run the Future Search Algorithm for multiple runs """
        for run in range(self.num_runs):
            self.initialize_population()
            best_values_iter = []

            for iteration in range(self.max_iter):
                self.update_population()
                self.update_with_initial_strategy()
                best_values_iter.append(self.global_best_value)

                # Display progress
                print(f"Run {run + 1}, Iteration {iteration + 1}: Best Value = {self.global_best_value}")

            # Store results for this run
            self.global_best_history.append(best_values_iter)
            self.best_positions.append(self.global_best_position.copy())

        # Find the best result across all runs
        best_run_idx = np.argmin([min(history) for history in self.global_best_history])
        best_score = min(self.global_best_history[best_run_idx])
        best_position = self.best_positions[best_run_idx]

        print(f"Best Score across all runs: {best_score}")
        print(f"Best Position: {best_position}")

        return best_position, best_score, self.global_best_history

