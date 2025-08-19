import numpy as np

class BiogeographyBasedOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, 
                 keep_rate=0.2, alpha=0.9, mutation_prob=0.1, mutation_scale=0.02):
        """
        Initialize the Biogeography-Based Optimization (BBO) algorithm.

        Parameters:
        - objective_function: Function to optimize (minimize).
        - dim: Number of dimensions (variables).
        - bounds: List of (lower, upper) bounds for each dimension.
        - population_size: Number of habitats (solutions).
        - max_iter: Maximum number of iterations.
        - keep_rate: Proportion of best habitats to keep each iteration.
        - alpha: Migration step size control parameter.
        - mutation_prob: Probability of mutation for each variable.
        - mutation_scale: Scale of mutation (relative to bounds range).
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.keep_rate = keep_rate
        self.n_keep = round(keep_rate * population_size)
        self.n_new = population_size - self.n_keep
        self.alpha = alpha
        self.mutation_prob = mutation_prob
        self.mutation_sigma = mutation_scale * (bounds[0][1] - bounds[0][0])

        # Migration rates
        self.mu = np.linspace(1, 0, population_size)  # Emigration rates
        self.lambda_ = 1 - self.mu  # Immigration rates

        self.habitats = None  # Population of habitats (solutions)
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_habitats(self):
        """Generate initial habitats randomly within bounds."""
        self.habitats = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                         (self.population_size, self.dim))
        # Ensure initial habitats are within bounds
        self.habitats = np.clip(self.habitats, self.bounds[:, 0], self.bounds[:, 1])

    def evaluate_habitats(self):
        """Compute fitness values for all habitats."""
        return np.array([self.objective_function(habitat) for habitat in self.habitats])

    def roulette_wheel_selection(self, probabilities):
        """Select an index using roulette wheel selection."""
        r = np.random.rand()
        cumsum = np.cumsum(probabilities)
        return np.where(r <= cumsum)[0][0]

    def migration(self, new_habitats, habitat_idx, var_idx):
        """Perform migration for a single variable of a habitat."""
        if np.random.rand() <= self.lambda_[habitat_idx]:
            # Emigration probabilities
            ep = self.mu.copy()
            ep[habitat_idx] = 0
            ep = ep / np.sum(ep)
            
            # Select source habitat
            source_idx = self.roulette_wheel_selection(ep)
            
            # Migration
            new_habitats[habitat_idx][var_idx] += (
                self.alpha * (self.habitats[source_idx][var_idx] - self.habitats[habitat_idx][var_idx])
            )

    def mutation(self, new_habitats, habitat_idx, var_idx):
        """Apply mutation to a single variable of a habitat."""
        if np.random.rand() <= self.mutation_prob:
            new_habitats[habitat_idx][var_idx] += self.mutation_sigma * np.random.randn()

    def enforce_bounds(self, habitats):
        """Ensure all habitats are within the specified bounds."""
        return np.clip(habitats, self.bounds[:, 0], self.bounds[:, 1])

    def optimize(self):
        """Run the Biogeography-Based Optimization algorithm."""
        self.initialize_habitats()
        
        for iteration in range(self.max_iter):
            # Evaluate current population
            fitness = self.evaluate_habitats()
            sorted_indices = np.argsort(fitness)
            self.habitats = self.habitats[sorted_indices]
            fitness = fitness[sorted_indices]
            
            # Update best solution, ensuring it's within bounds
            if fitness[0] < self.best_value:
                self.best_solution = np.clip(self.habitats[0].copy(), self.bounds[:, 0], self.bounds[:, 1])
                self.best_value = self.objective_function(self.best_solution)
            
            # Create new population
            new_habitats = self.habitats.copy()
            
            # Migration and mutation
            for i in range(self.population_size):
                for k in range(self.dim):
                    self.migration(new_habitats, i, k)
                    self.mutation(new_habitats, i, k)
                
                # Apply bounds after migration and mutation
                new_habitats[i] = self.enforce_bounds(new_habitats[i])
                
                # Evaluate new habitat
                fitness[i] = self.objective_function(new_habitats[i])
            
            # Sort new population
            sorted_indices = np.argsort(fitness)
            new_habitats = new_habitats[sorted_indices]
            fitness = fitness[sorted_indices]
            
            # Select next iteration population
            self.habitats = np.vstack((self.habitats[:self.n_keep], new_habitats[:self.n_new]))
            
            # Ensure all habitats are within bounds
            self.habitats = self.enforce_bounds(self.habitats)
            
            # Re-evaluate to ensure consistency
            fitness = self.evaluate_habitats()
            sorted_indices = np.argsort(fitness)
            self.habitats = self.habitats[sorted_indices]
            
            # Store history with best solution
            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}, Best Solution = {self.best_solution}")
        
        return self.best_solution, self.best_value, self.history
