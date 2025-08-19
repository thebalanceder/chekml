import numpy as np

class CollidingBodiesOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iterations=100, alpha=0.1):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = bounds
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.lower_limit = np.array([b[0] for b in bounds])
        self.upper_limit = np.array([b[1] for b in bounds])
        self.history = []
        self.global_best_solution = None
        self.global_best_fitness = np.inf

    def optimize(self):
        # Initialize population
        population = self.lower_limit + (self.upper_limit - self.lower_limit) * np.random.rand(self.population_size, self.dim)
        
        # Main loop
        for iter in range(self.max_iterations):
            # Evaluate fitness
            fitness = np.zeros(self.population_size)
            for i in range(self.population_size):
                fitness[i] = self.objective_function(population[i, :])
                
                # Update global best if current fitness is better
                if fitness[i] < self.global_best_fitness:
                    self.global_best_fitness = fitness[i]
                    self.global_best_solution = population[i, :].copy()
            
            # Sort population based on fitness
            sorted_indices = np.argsort(fitness)
            fitness = fitness[sorted_indices]
            population = population[sorted_indices, :]
            
            # Update positions
            for i in range(self.population_size):
                # Calculate center of mass
                center_of_mass = np.mean(population, axis=0)
                
                # Move towards center of mass
                direction = center_of_mass - population[i, :]
                population[i, :] = population[i, :] + self.alpha * direction
                
                # Ensure bounds
                population[i, :] = np.maximum(np.minimum(population[i, :], self.upper_limit), self.lower_limit)
                
                # Store history for visualization
                self.history.append((iter, population[i, :].copy(), fitness[i]))
        
        # Return the global best solution
        return self.global_best_solution, self.global_best_fitness, self.history
