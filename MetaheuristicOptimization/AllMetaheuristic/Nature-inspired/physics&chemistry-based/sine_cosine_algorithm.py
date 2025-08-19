import numpy as np

class SineCosineAlgorithm:
    """
    Sine Cosine Algorithm (SCA) for optimization problems.
    
    Parameters:
    - objective_function: Function to optimize.
    - dim: Number of dimensions (variables).
    - bounds: Tuple of (lower, upper) bounds for each dimension or single values if same for all.
    - population_size: Number of search agents.
    - max_iter: Maximum number of iterations.
    - a: Controls the linear decrease of r1 (default=2).
    
    Reference:
    S. Mirjalili, SCA: A Sine Cosine Algorithm for solving optimization problems
    Knowledge-Based Systems, DOI: http://dx.doi.org/10.1016/j.knosys.2015.12.022
    """
    
    def __init__(self, objective_function, dim, bounds, population_size=30, max_iter=100, a=2):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Convert bounds to numpy array
        self.population_size = population_size
        self.max_iter = max_iter
        self.a = a
        
        # Handle bounds: single values or per dimension
        if self.bounds.ndim == 1:
            self.lb = np.full(dim, self.bounds[0])
            self.ub = np.full(dim, self.bounds[1])
        else:
            self.lb = self.bounds[:, 0]
            self.ub = self.bounds[:, 1]
        
        self.solutions = None  # Population of solutions
        self.best_solution = None
        self.best_fitness = float("inf")
        self.convergence_curve = []

    def initialize_solutions(self):
        """Generate initial population of random solutions."""
        if self.bounds.ndim == 1:
            self.solutions = np.random.uniform(self.lb[0], self.ub[0], 
                                            (self.population_size, self.dim))
        else:
            self.solutions = np.zeros((self.population_size, self.dim))
            for i in range(self.dim):
                self.solutions[:, i] = np.random.uniform(self.lb[i], self.ub[i], 
                                                      self.population_size)

    def evaluate_solutions(self):
        """Compute fitness values for all solutions."""
        return np.array([self.objective_function(sol) for sol in self.solutions])

    def update_position(self, t):
        """Update positions of solutions using sine-cosine equations."""
        # Eq. (3.4): Linearly decreasing r1
        r1 = self.a - t * (self.a / self.max_iter)
        
        for i in range(self.population_size):
            for j in range(self.dim):
                # Update r2, r3, r4 for Eq. (3.3)
                r2 = 2 * np.pi * np.random.rand()
                r3 = 2 * np.random.rand()
                r4 = np.random.rand()
                
                # Eq. (3.3): Update position
                if r4 < 0.5:
                    # Eq. (3.1): Sine update
                    self.solutions[i, j] = (self.solutions[i, j] + 
                                          r1 * np.sin(r2) * 
                                          abs(r3 * self.best_solution[j] - self.solutions[i, j]))
                else:
                    # Eq. (3.2): Cosine update
                    self.solutions[i, j] = (self.solutions[i, j] + 
                                          r1 * np.cos(r2) * 
                                          abs(r3 * self.best_solution[j] - self.solutions[i, j]))
            
            # Ensure solutions stay within bounds
            self.solutions[i] = np.clip(self.solutions[i], self.lb, self.ub)

    def optimize(self):
        """Run the Sine Cosine Algorithm."""
        print("SCA is optimizing your problem...")
        
        # Initialize solutions
        self.initialize_solutions()
        
        # Evaluate initial solutions and find the best
        fitness_values = self.evaluate_solutions()
        min_idx = np.argmin(fitness_values)
        self.best_solution = self.solutions[min_idx].copy()
        self.best_fitness = fitness_values[min_idx]
        self.convergence_curve.append(self.best_fitness)
        
        # Main loop
        for t in range(1, self.max_iter):
            # Update positions
            self.update_position(t)
            
            # Evaluate new solutions
            fitness_values = self.evaluate_solutions()
            
            # Update best solution if a better one is found
            min_idx = np.argmin(fitness_values)
            if fitness_values[min_idx] < self.best_fitness:
                self.best_solution = self.solutions[min_idx].copy()
                self.best_fitness = fitness_values[min_idx]
            
            # Store convergence data
            self.convergence_curve.append(self.best_fitness)
            
            # Display progress every 50 iterations
            if (t + 1) % 50 == 0:
                print(f"At iteration {t + 1}, the optimum is {self.best_fitness}")
        
        print(f"The best solution obtained by SCA is: {self.best_solution}")
        print(f"The best optimal value of the objective function found by SCA is: {self.best_fitness}")
        
        return self.best_solution, self.best_fitness, self.convergence_curve

# Example usage
if __name__ == "__main__":
    # Example objective function (Sphere function: sum of squares)
    def sphere_function(x):
        return np.sum(x ** 2)
    
    # Parameters
    dim = 30
    bounds = [(-100, 100)] * dim  # Bounds for each dimension
    population_size = 30
    max_iter = 1000
    
    # Initialize and run SCA
    sca = SineCosineAlgorithm(sphere_function, dim, bounds, population_size, max_iter)
    best_solution, best_fitness, convergence_curve = sca.optimize()
