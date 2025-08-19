import numpy as np

class GreyWolfOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=30, max_iter=100):
        """
        Initialize the Grey Wolf Optimizer (GWO).

        Parameters:
        - objective_function: Function to optimize (minimization problem).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension or single pair for all.
        - population_size: Number of search agents (wolves).
        - max_iter: Maximum number of iterations.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter
        
        # Handle bounds
        if np.isscalar(bounds[0]):  # Single bound for all dimensions
            self.bounds = np.array([[bounds[0], bounds[1]] for _ in range(dim)])
        else:  # Different bounds for each dimension
            self.bounds = np.array(bounds)
        
        self.positions = None  # Population of search agents
        self.alpha_pos = np.zeros(dim)  # Initialize as zero arrays
        self.alpha_score = float("inf")
        self.beta_pos = np.zeros(dim)
        self.beta_score = float("inf")
        self.delta_pos = np.zeros(dim)
        self.delta_score = float("inf")
        self.convergence_curve = []
        self.alpha_history = []  # Store Alpha positions for plotting

    def initialize_positions(self):
        """Initialize the positions of search agents."""
        if len(self.bounds) == 1:  # Single bound for all dimensions
            self.positions = np.random.uniform(self.bounds[0, 0], self.bounds[0, 1],
                                             (self.population_size, self.dim))
        else:  # Different bounds for each dimension
            self.positions = np.zeros((self.population_size, self.dim))
            for i in range(self.dim):
                self.positions[:, i] = np.random.uniform(self.bounds[i, 0], self.bounds[i, 1],
                                                        self.population_size)

    def enforce_bounds(self):
        """Return search agents that go beyond the boundaries to the search space."""
        for i in range(self.population_size):
            for j in range(self.dim):
                self.positions[i, j] = np.clip(self.positions[i, j], self.bounds[j, 0], self.bounds[j, 1])

    def update_hierarchy(self):
        """Update Alpha, Beta, and Delta wolves based on fitness."""
        fitness = np.array([self.objective_function(self.positions[i]) for i in range(self.population_size)])
        
        for i in range(self.population_size):
            if fitness[i] < self.alpha_score:
                self.delta_score = self.beta_score
                self.delta_pos = self.beta_pos.copy()
                self.beta_score = self.alpha_score
                self.beta_pos = self.alpha_pos.copy()
                self.alpha_score = fitness[i]
                self.alpha_pos = self.positions[i].copy()
            elif fitness[i] < self.beta_score and fitness[i] > self.alpha_score:
                self.delta_score = self.beta_score
                self.delta_pos = self.beta_pos.copy()
                self.beta_score = fitness[i]
                self.beta_pos = self.positions[i].copy()
            elif fitness[i] < self.delta_score and fitness[i] > self.beta_score:
                self.delta_score = fitness[i]
                self.delta_pos = self.positions[i].copy()

    def update_positions(self, iteration):
        """Update the positions of search agents based on Alpha, Beta, and Delta."""
        a = 2 - iteration * (2 / self.max_iter)  # Linearly decrease a from 2 to 0
        
        for i in range(self.population_size):
            for j in range(self.dim):
                # Alpha update
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i, j])
                X1 = self.alpha_pos[j] - A1 * D_alpha
                
                # Beta update
                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * self.beta_pos[j] - self.positions[i, j])
                X2 = self.beta_pos[j] - A2 * D_beta
                
                # Delta update
                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * self.delta_pos[j] - self.positions[i, j])
                X3 = self.delta_pos[j] - A3 * D_delta
                
                # Update position
                self.positions[i, j] = (X1 + X2 + X3) / 3

    def optimize(self):
        """Run the Grey Wolf Optimizer."""
        self.initialize_positions()
        self.convergence_curve = np.zeros(self.max_iter)
        self.alpha_history = []
        
        for iteration in range(self.max_iter):
            self.enforce_bounds()
            self.update_hierarchy()
            self.update_positions(iteration)
            self.convergence_curve[iteration] = self.alpha_score
            self.alpha_history.append(self.alpha_pos.copy())
            print(f"Iteration {iteration + 1}: Best Score = {self.alpha_score}")
        
        return self.alpha_pos, self.alpha_score, self.convergence_curve, self.alpha_history

