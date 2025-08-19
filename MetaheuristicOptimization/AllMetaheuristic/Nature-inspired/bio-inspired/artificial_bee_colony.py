import numpy as np

def roulette_wheel_selection(probabilities):
    """
    Perform roulette wheel selection based on given probabilities.
    
    Args:
        probabilities: Array of selection probabilities
    
    Returns:
        Selected index
    """
    r = np.random.rand()
    cumsum = np.cumsum(probabilities)
    return np.where(r <= cumsum)[0][0]

class ArtificialBeeColony:
    """
    Implementation of Artificial Bee Colony (ABC) optimization algorithm.
    
    Parameters:
    - objective_function: Function to optimize
    - dim: Number of decision variables
    - bounds: Tuple of (lower, upper) bounds for each dimension
    - population_size: Size of bee colony
    - max_iter: Maximum number of iterations
    - onlooker_ratio: Ratio of onlooker bees to population size
    - trial_limit_factor: Factor for abandonment limit
    - acceleration_bound: Upper bound for acceleration coefficient
    """
    
    def __init__(self, objective_function, dim, bounds, population_size=100, 
                 max_iter=200, onlooker_ratio=1.0, trial_limit_factor=0.6, 
                 acceleration_bound=1.0):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.n_onlookers = int(population_size * onlooker_ratio)
        self.trial_limit = round(trial_limit_factor * dim * population_size)
        self.acceleration_bound = acceleration_bound
        
        self.population = None  # Food sources (solutions)
        self.costs = None  # Cost values
        self.best_solution = None
        self.best_cost = float('inf')
        self.trial_counters = None  # Abandonment counters
        self.history = []  # History of (iteration, best_solution, best_cost)
        
    def initialize_population(self):
        """Initialize population with random solutions within bounds."""
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                          (self.population_size, self.dim))
        self.costs = np.array([self.objective_function(bee) for bee in self.population])
        self.trial_counters = np.zeros(self.population_size)
        
        # Initialize best solution
        min_idx = np.argmin(self.costs)
        if self.costs[min_idx] < self.best_cost:
            self.best_solution = self.population[min_idx].copy()
            self.best_cost = self.costs[min_idx]
    
    def employed_bee_phase(self):
        """Perform employed bee phase (exploration)."""
        for i in range(self.population_size):
            # Select random bee (k != i)
            k = np.random.choice([j for j in range(self.population_size) if j != i])
            
            # Generate acceleration coefficient
            phi = self.acceleration_bound * np.random.uniform(-1, 1, self.dim)
            
            # Create new solution
            new_position = self.population[i] + phi * (self.population[i] - self.population[k])
            new_position = np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])
            
            # Evaluate new solution
            new_cost = self.objective_function(new_position)
            
            # Greedy selection
            if new_cost <= self.costs[i]:
                self.population[i] = new_position
                self.costs[i] = new_cost
                self.trial_counters[i] = 0
            else:
                self.trial_counters[i] += 1
    
    def calculate_selection_probabilities(self):
        """Calculate fitness values and selection probabilities."""
        fitness = np.zeros(self.population_size)
        mean_cost = np.mean(self.costs)
        
        for i in range(self.population_size):
            fitness[i] = np.exp(-self.costs[i] / mean_cost)  # Convert cost to fitness
        
        return fitness / np.sum(fitness)
    
    def onlooker_bee_phase(self):
        """Perform onlooker bee phase (exploitation)."""
        probabilities = self.calculate_selection_probabilities()
        
        for _ in range(self.n_onlookers):
            # Select food source using roulette wheel
            i = roulette_wheel_selection(probabilities)
            
            # Select random bee (k != i)
            k = np.random.choice([j for j in range(self.population_size) if j != i])
            
            # Generate acceleration coefficient
            phi = self.acceleration_bound * np.random.uniform(-1, 1, self.dim)
            
            # Create new solution
            new_position = self.population[i] + phi * (self.population[i] - self.population[k])
            new_position = np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])
            
            # Evaluate new solution
            new_cost = self.objective_function(new_position)
            
            # Greedy selection
            if new_cost <= self.costs[i]:
                self.population[i] = new_position
                self.costs[i] = new_cost
                self.trial_counters[i] = 0
            else:
                self.trial_counters[i] += 1
    
    def scout_bee_phase(self):
        """Perform scout bee phase (abandonment and reinitialization)."""
        for i in range(self.population_size):
            if self.trial_counters[i] >= self.trial_limit:
                self.population[i] = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)
                self.costs[i] = self.objective_function(self.population[i])
                self.trial_counters[i] = 0
    
    def update_best_solution(self):
        """Update the best solution found so far."""
        min_idx = np.argmin(self.costs)
        if self.costs[min_idx] < self.best_cost:
            self.best_solution = self.population[min_idx].copy()
            self.best_cost = self.costs[min_idx]
    
    def optimize(self):
        """Run the Artificial Bee Colony optimization."""
        self.initialize_population()
        
        for iteration in range(self.max_iter):
            # Employed bee phase
            self.employed_bee_phase()
            
            # Onlooker bee phase
            self.onlooker_bee_phase()
            
            # Scout bee phase
            self.scout_bee_phase()
            
            # Update best solution
            self.update_best_solution()
            
            # Store iteration, best solution, and best cost
            self.history.append((iteration, self.best_solution.copy(), self.best_cost))
            
            # Display iteration information
            print(f"Iteration {iteration + 1}: Best Cost = {self.best_cost}")
        
        return self.best_solution, self.best_cost, self.history

# Example usage
if __name__ == "__main__":
    def sphere(x):
        """Sphere function for testing."""
        return np.sum(x ** 2)
    
    dim = 5
    bounds = [(-10, 10)] * dim
    abc = ArtificialBeeColony(sphere, dim, bounds)
    best_solution, best_cost, history = abc.optimize()
    
    import matplotlib.pyplot as plt
    plt.semilogy([h[2] for h in history], 'LineWidth', 2)
    plt.xlabel('Iteration')
    plt.ylabel('Best Cost')
    plt.grid(True)
    plt.savefig('abc_convergence.png')
