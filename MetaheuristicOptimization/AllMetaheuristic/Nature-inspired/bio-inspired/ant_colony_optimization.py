import numpy as np

class AntColonyOptimizer:
    def __init__(self, objective_function, dim, bounds, max_iter=300, n_ant=40, Q=1.0, tau0=0.1, 
                 alpha=1.0, beta=0.02, rho=0.1, n_bins=10):
        """
        Initialize the Ant Colony Optimizer for continuous optimization problems.

        Parameters:
        - objective_function: Function to optimize (minimization).
        - dim: Number of dimensions (variables).
        - bounds: List of tuples [(lower, upper), ...] for each dimension.
        - max_iter: Maximum number of iterations.
        - n_ant: Number of ants (population size).
        - Q: Pheromone deposit factor.
        - tau0: Initial pheromone value.
        - alpha: Pheromone exponential weight.
        - beta: Heuristic exponential weight.
        - rho: Pheromone evaporation rate.
        - n_bins: Number of discrete bins per dimension for pheromone grid.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Shape: (dim, 2) with [lower, upper]
        self.max_iter = max_iter
        self.n_ant = n_ant
        self.Q = Q
        self.tau0 = tau0
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.n_bins = n_bins

        # Discretize each dimension into n_bins
        self.bins = [np.linspace(self.bounds[i, 0], self.bounds[i, 1], self.n_bins) 
                     for i in range(self.dim)]
        self.tau = [np.full(self.n_bins, self.tau0) for _ in range(self.dim)]  # Pheromone per dimension
        self.eta = [np.ones(self.n_bins) for _ in range(self.dim)]  # Heuristic (uniform initially)

        # Ant colony and best solution tracking
        self.ants = None
        self.best_solution = None
        self.best_cost = float('inf')
        self.history = []

    def _roulette_wheel_selection(self, P):
        """Select an index based on roulette wheel selection."""
        r = np.random.rand()
        C = np.cumsum(P)
        return np.where(r <= C)[0][0]

    def _initialize_ants(self):
        """Initialize the ant colony."""
        self.ants = [{'Tour': [], 'x': [], 'Cost': float('inf')} 
                     for _ in range(self.n_ant)]

    def optimize(self):
        """Run the Ant Colony Optimization algorithm for continuous problems."""
        self._initialize_ants()

        for it in range(self.max_iter):
            # Move ants
            for k in range(self.n_ant):
                self.ants[k]['Tour'] = []
                self.ants[k]['x'] = np.zeros(self.dim)
                
                # Construct solution by selecting a bin for each dimension
                for d in range(self.dim):
                    P = (self.tau[d] ** self.alpha) * (self.eta[d] ** self.beta)
                    P = P / np.sum(P)
                    bin_idx = self._roulette_wheel_selection(P)
                    self.ants[k]['Tour'].append(bin_idx)
                    self.ants[k]['x'][d] = self.bins[d][bin_idx]
                
                # Evaluate solution
                self.ants[k]['Cost'] = self.objective_function(self.ants[k]['x'])
                
                # Update best solution
                if self.ants[k]['Cost'] < self.best_cost:
                    self.best_cost = self.ants[k]['Cost']
                    self.best_solution = self.ants[k]['x'].copy()

            # Update pheromones
            for k in range(self.n_ant):
                for d in range(self.dim):
                    bin_idx = self.ants[k]['Tour'][d]
                    self.tau[d][bin_idx] += self.Q / (1 + self.ants[k]['Cost'] - self.best_cost)

            # Evaporation
            for d in range(self.dim):
                self.tau[d] *= (1 - self.rho)

            # Update heuristic information (optional, based on recent costs)
            for d in range(self.dim):
                self.eta[d] = np.ones(self.n_bins)  # Could be improved with problem-specific heuristics

            # Store history
            self.history.append((it, self.best_solution.copy(), self.best_cost))
            
            # Display iteration information
            print(f"Iteration {it + 1}: Best Cost = {self.best_cost}")

        return self.best_solution, self.best_cost, self.history
