import numpy as np

class AcrossNeighborhoodSearch:
    def __init__(self, objective_function, dim=2, bounds=None,
                 num_neighborhoods=5, neighborhood_radius=0.1,
                 max_iterations=100, mutation_rate=0.1):
        """
        Initialize the Across Neighborhood Search (ANS) optimizer.

        Parameters:
        - objective_function: Function to be minimized.
        - dim: Number of decision variables.
        - bounds: List of (lower, upper) tuples for each dimension.
        - num_neighborhoods: Number of neighborhoods (agents).
        - neighborhood_radius: Not used directly, kept for future expansion.
        - max_iterations: Maximum number of iterations.
        - mutation_rate: Step size for updates.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = bounds if bounds else [(-5, 5)] * dim
        self.num_neighborhoods = num_neighborhoods
        self.max_iterations = max_iterations
        self.mutation_rate = mutation_rate

        self.lower_bounds = np.array([b[0] for b in self.bounds])
        self.upper_bounds = np.array([b[1] for b in self.bounds])

        self.populations = np.random.uniform(self.lower_bounds, self.upper_bounds,
                                             (self.num_neighborhoods, self.dim))
        self.fitness = np.full(self.num_neighborhoods, np.inf)
        self.best_solution = None
        self.best_fitness = float("inf")
        self.history = []

    def optimize(self):
        for iteration in range(self.max_iterations):
            # Evaluate fitness
            for i in range(self.num_neighborhoods):
                self.fitness[i] = self.objective_function(self.populations[i])
            
            # Update positions
            for i in range(self.num_neighborhoods):
                # Select a random neighbor
                neighbor_index = np.random.randint(0, self.num_neighborhoods - 1)
                if neighbor_index >= i:
                    neighbor_index += 1

                # Compute direction and update
                direction = self.populations[neighbor_index] - self.populations[i]
                self.populations[i] += self.mutation_rate * direction

                # Enforce boundary constraints
                self.populations[i] = np.clip(self.populations[i], self.lower_bounds, self.upper_bounds)

            # Update best solution found
            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[current_best_idx]
                self.best_solution = self.populations[current_best_idx].copy()

            self.history.append((iteration, self.best_solution.copy()))
            print(f"Iteration {iteration + 1}: Best Fitness = {self.best_fitness}")

        return self.best_solution, self.best_fitness, self.history
