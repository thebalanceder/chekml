import numpy as np

class WaterwheelPlantOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, 
                 r1_range=(0, 2), r2_range=(0, 1), r3_range=(0, 2), k_initial=1.0, 
                 f_range=(-5, 5), c_range=(-5, 5)):
        """
        Initialize the Waterwheel Plant Algorithm (WWPA) optimizer.

        Parameters:
        - objective_function: Function to optimize (minimize).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of waterwheel plants (solutions).
        - max_iter: Maximum number of iterations.
        - r1_range: Range for random variable r1 in exploration phase.
        - r2_range: Range for random variable r2 in exploration phase.
        - r3_range: Range for random variable r3 in exploitation phase.
        - k_initial: Initial value for parameter K.
        - f_range: Range for random variable f in mutation.
        - c_range: Range for random variable c in mutation.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.r1_range = r1_range
        self.r2_range = r2_range
        self.r3_range = r3_range
        self.k_initial = k_initial
        self.f_range = f_range
        self.c_range = c_range

        self.waterwheels = None  # Population of waterwheel plants (solutions)
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []
        self.stagnation_counts = np.zeros(population_size)  # Track stagnation for each solution

    def initialize_waterwheels(self):
        """Initialize waterwheel plant positions randomly within bounds."""
        self.waterwheels = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                            (self.population_size, self.dim))

    def evaluate_waterwheels(self):
        """Compute fitness values for the waterwheel plant positions."""
        return np.array([self.objective_function(plant) for plant in self.waterwheels])

    def exploration_phase(self, index, t, K):
        """Simulate position identification and hunting of insects (exploration)."""
        r1 = np.random.uniform(self.r1_range[0], self.r1_range[1])
        r2 = np.random.uniform(self.r2_range[0], self.r2_range[1])
        
        # Equation (4): W = r1 * (P(t) + 2K)
        W = r1 * (self.waterwheels[index] + 2 * K)
        # Equation (5): P(t+1) = P(t) + W * (2K + r2)
        new_position = self.waterwheels[index] + W * (2 * K + r2)
        
        # Clip to bounds
        new_position = np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])
        
        # Check for stagnation and apply mutation if needed (Equation 6)
        new_value = self.objective_function(new_position)
        if new_value < self.evaluate_waterwheels()[index]:
            self.stagnation_counts[index] = 0
            return new_position
        else:
            self.stagnation_counts[index] += 1
            if self.stagnation_counts[index] >= 3:
                # Equation (6): P(t+1) = Gaussian(mu_P, sigma) + r1 * ((P(t) + 2K) / W)
                mu_P = np.mean(self.waterwheels, axis=0)
                sigma = np.std(self.waterwheels, axis=0)
                gaussian_term = np.random.normal(mu_P, sigma)
                if np.all(W != 0):  # Avoid division by zero
                    new_position = gaussian_term + r1 * ((self.waterwheels[index] + 2 * K) / W)
                else:
                    new_position = gaussian_term
                new_position = np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])
                self.stagnation_counts[index] = 0
            return new_position

    def exploitation_phase(self, index, t, K):
        """Simulate carrying the insect to the suitable tube (exploitation)."""
        r3 = np.random.uniform(self.r3_range[0], self.r3_range[1])
        
        # Equation (7): W = r3 * (K * P_best(t) + r3 * P(t))
        W = r3 * (K * self.best_solution + r3 * self.waterwheels[index])
        # Equation (8): P(t+1) = P(t) + K * W
        new_position = self.waterwheels[index] + K * W
        
        # Clip to bounds
        new_position = np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])
        
        # Check for stagnation and apply mutation if needed (Equation 9)
        new_value = self.objective_function(new_position)
        if new_value < self.evaluate_waterwheels()[index]:
            self.stagnation_counts[index] = 0
            return new_position
        else:
            self.stagnation_counts[index] += 1
            if self.stagnation_counts[index] >= 3:
                # Equation (9): P(t+1) = (r1 + K) * sin((f / c) * theta)
                r1 = np.random.uniform(self.r1_range[0], self.r1_range[1])
                f = np.random.uniform(self.f_range[0], self.f_range[1])
                c = np.random.uniform(self.c_range[0], self.c_range[1])
                theta = np.random.rand() * 2 * np.pi
                if c != 0:  # Avoid division by zero
                    new_position = (r1 + K) * np.sin((f / c) * theta)
                else:
                    new_position = self.waterwheels[index]
                new_position = np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])
                self.stagnation_counts[index] = 0
            return new_position

    def update_k(self, t, f):
        """Update the K parameter exponentially (Equation 10)."""
        return 1 + (2 * t**2 / (self.max_iter**2)) + f

    def optimize(self):
        """Run the Waterwheel Plant Algorithm optimization process."""
        self.initialize_waterwheels()
        
        for t in range(self.max_iter):
            fitness = self.evaluate_waterwheels()
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.waterwheels[min_idx].copy()
                self.best_value = fitness[min_idx]
            
            # Update parameters
            r = np.random.rand()
            f = np.random.uniform(self.f_range[0], self.f_range[1])
            K = self.update_k(t, f)
            
            # Update each waterwheel's position
            for i in range(self.population_size):
                if r < 0.5:
                    # Exploration phase
                    self.waterwheels[i] = self.exploration_phase(i, t, K)
                else:
                    # Exploitation phase
                    self.waterwheels[i] = self.exploitation_phase(i, t, K)
            
            # Update parameters for next iteration
            self.history.append((t, self.best_solution.copy(), self.best_value))
            print(f"Iteration {t + 1}: Best Value = {self.best_value}")
        
        return self.best_solution, self.best_value, self.history

# Example usage
if __name__ == "__main__":
    # Example objective function: Sphere function
    def sphere_function(x):
        return np.sum(x**2)
    
    # Define problem parameters
    dim = 30
    bounds = [(-100, 100)] * dim  # Bounds for each dimension
    population_size = 50
    max_iter = 500
    
    # Initialize and run optimizer
    optimizer = WaterwheelPlantOptimizer(
        objective_function=sphere_function,
        dim=dim,
        bounds=bounds,
        population_size=population_size,
        max_iter=max_iter
    )
    best_solution, best_value, history = optimizer.optimize()
    
    print(f"Best Solution: {best_solution}")
    print(f"Best Value: {best_value}")
