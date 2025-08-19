import numpy as np

class HuntingSearch:
    def __init__(self, num_hunters, dim, bounds, alpha=0.8, beta=0.2, iterations=100):
        self.num_hunters = num_hunters  # Population size
        self.dim = dim  # Dimension of the search space
        self.bounds = np.array(bounds)  # Search space bounds [(low, high), ...]
        self.alpha = alpha  # Tracking weight (toward best solution)
        self.beta = beta  # Random exploration factor
        self.iterations = iterations  # Max iterations

        # Initialize hunters randomly within bounds
        self.hunters = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.num_hunters, self.dim))
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def evaluate_hunters(self, objective_function):
        """Compute fitness values for all hunters."""
        return np.array([objective_function(hunter) for hunter in self.hunters])

    def move_hunters(self, fitness):
        """Move hunters towards the best found prey (solution) with some randomness."""
        best_idx = np.argmin(fitness)  # Best hunter index
        best_hunter = self.hunters[best_idx]  # Current best solution

        for i in range(self.num_hunters):
            if i != best_idx:  # Skip the best hunter
                direction = best_hunter - self.hunters[i]
                random_exploration = np.random.uniform(-1, 1, self.dim)
                self.hunters[i] += self.alpha * direction + self.beta * random_exploration  # Update position

        # Ensure solutions stay within bounds
        self.hunters = np.clip(self.hunters, self.bounds[:, 0], self.bounds[:, 1])

    def optimize(self, objective_function):
        """Run the Hunting Search Algorithm."""
        for iteration in range(self.iterations):
            fitness = self.evaluate_hunters(objective_function)
            self.move_hunters(fitness)

            # Update best solution
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.hunters[min_idx]
                self.best_value = fitness[min_idx]

            self.history.append((iteration, self.best_solution.copy(), self.best_value))

            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

