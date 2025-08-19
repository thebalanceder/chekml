import numpy as np

class OwlSearchOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, 
                 step_size=0.1, p_explore=0.1):
        """
        Initialize the Owl Search Algorithm (OSA) optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of owls (solutions).
        - max_iter: Maximum number of iterations.
        - step_size: Step size for movement.
        - p_explore: Probability of exploration.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.step_size = step_size
        self.p_explore = p_explore

        self.population = None  # Population of owl positions (solutions)
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_population(self):
        """Generate initial owl population randomly within bounds"""
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                           (self.population_size, self.dim))

    def evaluate_population(self):
        """Compute fitness values for the owl population"""
        return np.array([self.objective_function(owl) for owl in self.population])

    def exploration_phase(self, index):
        """Simulate owl exploration with random movement"""
        random_move = self.step_size * (np.random.rand(self.dim) * 2 - 1)
        new_solution = self.population[index] + random_move
        return np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])

    def exploitation_phase(self, index):
        """Simulate owl exploitation by moving towards the best solution"""
        direction = self.best_solution - self.population[index]
        new_solution = self.population[index] + self.step_size * direction
        return np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])

    def optimize(self):
        """Run the Owl Search Algorithm optimization"""
        self.initialize_population()
        
        for iteration in range(self.max_iter):
            # Evaluate fitness for each owl
            fitness = self.evaluate_population()
            min_idx = np.argmin(fitness)
            
            # Update best solution if a better one is found
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.population[min_idx]
                self.best_value = fitness[min_idx]

            # Update each owl's position
            for i in range(self.population_size):
                if np.random.rand() < self.p_explore:
                    # Exploration: Random movement
                    self.population[i] = self.exploration_phase(i)
                else:
                    # Exploitation: Move towards best solution
                    self.population[i] = self.exploitation_phase(i)

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

