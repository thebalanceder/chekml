import numpy as np

class JaguarAlgorithmOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100,
                 p_cruise=0.8, cruising_distance=0.1, alpha=0.1):
        """
        Initialize the Jaguar Algorithm Optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of jaguars (solutions).
        - max_iter: Maximum number of iterations.
        - p_cruise: Probability of cruising (controls exploration).
        - cruising_distance: Maximum cruising distance.
        - alpha: Learning rate for position updates.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.p_cruise = p_cruise
        self.cruising_distance = cruising_distance
        self.alpha = alpha

        self.population = None  # Population of jaguars (solutions)
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_population(self):
        """ Generate initial jaguar population randomly """
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                           (self.population_size, self.dim))

    def evaluate_population(self):
        """ Compute fitness values for the jaguar population """
        return np.array([self.objective_function(individual) for individual in self.population])

    def cruising_phase(self, index, iteration):
        """ Simulate cruising behavior for top-performing jaguars """
        # Generate random direction
        direction = np.random.uniform(-1, 1, self.dim)
        
        # Normalize direction
        direction = direction / np.linalg.norm(direction)
        
        # Calculate adaptive cruising distance
        current_cruising_distance = self.cruising_distance * (1 - iteration / self.max_iter)
        
        # Update position
        new_solution = self.population[index] + self.alpha * current_cruising_distance * direction
        
        # Ensure new position is within bounds
        return np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])

    def random_walk_phase(self, index):
        """ Simulate random walk for non-cruising jaguars """
        # Generate random direction
        direction = np.random.uniform(-1, 1, self.dim)
        
        # Normalize direction
        direction = direction / np.linalg.norm(direction)
        
        # Update position
        new_solution = self.population[index] + self.alpha * direction
        
        # Ensure new position is within bounds
        return np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])

    def optimize(self):
        """ Run the Jaguar Algorithm Optimization """
        self.initialize_population()
        
        for iteration in range(self.max_iter):
            # Evaluate fitness
            fitness = self.evaluate_population()
            
            # Sort population based on fitness
            sorted_indices = np.argsort(fitness)
            fitness = fitness[sorted_indices]
            self.population = self.population[sorted_indices]
            
            # Update best solution
            if fitness[0] < self.best_value:
                self.best_solution = self.population[0].copy()
                self.best_value = fitness[0]
            
            # Determine number of cruising jaguars
            num_cruising = round(self.p_cruise * self.population_size)
            
            # Update cruising jaguars
            for i in range(num_cruising):
                self.population[i] = self.cruising_phase(i, iteration)
            
            # Update remaining jaguars with random walk
            for i in range(num_cruising, self.population_size):
                self.population[i] = self.random_walk_phase(i)
            
            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")
        
        return self.best_solution, self.best_value, self.history

