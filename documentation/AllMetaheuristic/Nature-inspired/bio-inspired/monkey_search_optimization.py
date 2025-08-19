import numpy as np

class MonkeySearchOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=20, max_iter=100, 
                 p_explore=0.2, max_p_explore=0.8, min_p_explore=0.1):
        """
        Initialize the Monkey Search Algorithm (MSA) optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of monkey positions (solutions).
        - max_iter: Maximum number of iterations.
        - p_explore: Initial probability of exploration.
        - max_p_explore: Maximum probability of exploration.
        - min_p_explore: Minimum probability of exploration.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.p_explore = p_explore
        self.max_p_explore = max_p_explore
        self.min_p_explore = min_p_explore

        self.population = None  # Population of monkey positions (solutions)
        self.best_solution = None
        self.best_fitness = float("inf")
        self.history = []

    def initialize_population(self):
        """ Generate initial monkey positions randomly """
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                           (self.population_size, self.dim))

    def evaluate_population(self):
        """ Compute fitness values for the monkey positions """
        return np.array([self.objective_function(flow) for flow in self.population])

    def update_position(self, iter):
        """ Update monkey positions based on exploration or exploitation """
        r = np.random.rand(self.population_size, self.dim)
        
        # Update probability of exploration adaptively
        self.p_explore = max(self.max_p_explore * np.exp(-0.1 * iter), self.min_p_explore)
        
        for i in range(self.population_size):
            for j in range(self.dim):
                if np.random.rand() < self.p_explore:  # Exploration
                    if r[i, j] < 0.5:
                        self.population[i, j] = self.best_solution[j] + np.random.rand() * (self.bounds[j, 1] - self.best_solution[j])
                    else:
                        self.population[i, j] = self.best_solution[j] - np.random.rand() * (self.best_solution[j] - self.bounds[j, 0])
                else:  # Exploitation
                    self.population[i, j] = self.best_solution[j] + np.random.randn() * (self.bounds[j, 1] - self.bounds[j, 0]) / 10  # Local perturbation
                
                # Bound the positions
                self.population[i, j] = np.clip(self.population[i, j], self.bounds[j, 0], self.bounds[j, 1])

    def optimize(self):
        """ Run the Monkey Search Algorithm Optimization """
        self.initialize_population()
        
        # Evaluate initial population
        fitness = self.evaluate_population()
        min_idx = np.argmin(fitness)
        self.best_fitness = fitness[min_idx]
        self.best_solution = self.population[min_idx].copy()
        
        for iter in range(self.max_iter):
            # Update positions
            self.update_position(iter)
            
            # Evaluate updated population
            fitness = self.evaluate_population()
            
            # Update best solution and fitness
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_fitness:
                self.best_fitness = fitness[min_idx]
                self.best_solution = self.population[min_idx].copy()
            
            self.history.append((iter, self.best_solution.copy(), self.best_fitness))
            print(f"Iteration {iter + 1}: Best Fitness = {self.best_fitness}")
        
        return self.best_solution, self.best_fitness, self.history

