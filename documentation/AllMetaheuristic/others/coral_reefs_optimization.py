import numpy as np

class CoralReefsOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, 
                 num_reefs=10, alpha=0.1):
        """
        Initialize the Coral Reefs Optimization (CRO) algorithm.

        Parameters:
        - objective_function: Function to optimize (takes a 1D NumPy array, returns a scalar).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension (same for all dimensions).
        - population_size: Number of solutions per reef.
        - max_iter: Maximum number of iterations.
        - num_reefs: Number of reefs (subpopulations).
        - alpha: Scaling factor for local search perturbation.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [lower, upper]
        self.population_size = population_size
        self.max_iter = max_iter
        self.num_reefs = num_reefs
        self.alpha = alpha

        self.reefs = None  # List of reef populations (solutions)
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_reefs(self):
        """Generate initial reef populations randomly within bounds."""
        lower, upper = self.bounds
        self.reefs = [
            np.random.uniform(lower, upper, (self.population_size, self.dim))
            for _ in range(self.num_reefs)
        ]

    def evaluate_reefs(self):
        """Compute fitness values for all solutions in each reef."""
        fitness = np.zeros((self.num_reefs, self.population_size))
        for i in range(self.num_reefs):
            for j in range(self.population_size):
                fitness[i, j] = self.objective_function(self.reefs[i][j])
        return fitness

    def migration_phase(self):
        """Exchange solutions between reefs to simulate migration."""
        for i in range(self.num_reefs):
            for j in range(self.num_reefs):
                if i != j:
                    # Randomly select a solution to migrate from reef i
                    idx = np.random.randint(0, self.population_size)
                    solution_to_migrate = self.reefs[i][idx].copy()
                    # Replace a random solution in reef j
                    idx_replace = np.random.randint(0, self.population_size)
                    self.reefs[j][idx_replace] = solution_to_migrate

    def local_search_phase(self):
        """Perform local search by perturbing solutions in each reef."""
        lower, upper = self.bounds
        for i in range(self.num_reefs):
            for j in range(self.population_size):
                # Perturb solution with random noise
                self.reefs[i][j] += self.alpha * np.random.randn(self.dim)
                # Enforce bounds
                self.reefs[i][j] = np.clip(self.reefs[i][j], lower, upper)

    def optimize(self):
        """Run the Coral Reefs Optimization algorithm."""
        self.initialize_reefs()
        for generation in range(self.max_iter):
            # Evaluate fitness
            fitness = self.evaluate_reefs()

            # Find best solution across all reefs
            min_fitness = float("inf")
            best_reef_idx = 0
            best_solution_idx = 0
            for i in range(self.num_reefs):
                for j in range(self.population_size):
                    if fitness[i, j] < min_fitness:
                        min_fitness = fitness[i, j]
                        best_reef_idx = i
                        best_solution_idx = j
            if min_fitness < self.best_value:
                self.best_value = min_fitness
                self.best_solution = self.reefs[best_reef_idx][best_solution_idx].copy()

            # Migration phase
            self.migration_phase()

            # Local search phase
            self.local_search_phase()

            self.history.append((generation, self.best_solution.copy(), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
