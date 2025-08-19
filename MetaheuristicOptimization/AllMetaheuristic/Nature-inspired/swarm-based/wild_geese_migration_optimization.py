import numpy as np

class WildGeeseMigrationOptimizer:
    def __init__(self, objective_function, dim, bounds, num_geese=20, max_iter=100, 
                 alpha=0.9, beta=0.1, gamma=0.1):
        """
        Initialize the Wild Geese Migration Optimization algorithm.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - num_geese: Number of geese (solutions).
        - max_iter: Maximum number of iterations.
        - alpha: Scaling factor.
        - beta: Learning rate.
        - gamma: Randomization parameter.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.num_geese = num_geese
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.geese = None  # Population of geese (solutions)
        self.best_solution = None
        self.best_fitness = float("inf")
        self.history = []

    def initialize_geese(self):
        """ Randomly initialize goose positions """
        self.geese = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                       (self.num_geese, self.dim))

    def evaluate_geese(self):
        """ Compute fitness values for all geese """
        return np.array([self.objective_function(goose) for goose in self.geese])

    def update_positions(self):
        """ Update positions of all geese based on the best goose """
        best_goose = self.geese[0]  # Best goose after sorting
        for i in range(self.num_geese):
            # Update position
            self.geese[i] = (self.alpha * self.geese[i] + 
                            self.beta * np.random.rand(self.dim) * (best_goose - self.geese[i]) + 
                            self.gamma * np.random.rand(self.dim) * (self.bounds[:, 1] - self.bounds[:, 0]))
            # Ensure positions stay within bounds
            self.geese[i] = np.clip(self.geese[i], self.bounds[:, 0], self.bounds[:, 1])

    def optimize(self):
        """ Run the Wild Geese Migration Optimization algorithm """
        self.initialize_geese()
        for iteration in range(self.max_iter):
            # Evaluate fitness for each goose
            fitness = self.evaluate_geese()
            
            # Sort geese based on fitness
            sorted_indices = np.argsort(fitness)
            fitness = fitness[sorted_indices]
            self.geese = self.geese[sorted_indices]
            
            # Update best solution
            if fitness[0] < self.best_fitness:
                self.best_solution = self.geese[0].copy()
                self.best_fitness = fitness[0]
            
            # Update goose positions
            self.update_positions()
            
            # Store history
            self.history.append((iteration, self.best_solution.copy(), self.best_fitness))
            print(f"Iteration {iteration + 1}: Best Fitness = {self.best_fitness}")
        
        print("\nOptimization finished.")
        print(f"Best solution found: {self.best_solution}")
        print(f"Best fitness: {self.best_fitness}")
        
        return self.best_solution, self.best_fitness, self.history

# Example objective function (Sphere function)
def sphere_function(x):
    return np.sum(x ** 2)

if __name__ == "__main__":
    # Example usage
    dim = 5
    bounds = [(-10, 10)] * dim
    optimizer = WildGeeseMigrationOptimizer(sphere_function, dim, bounds)
    best_solution, best_fitness, history = optimizer.optimize()
