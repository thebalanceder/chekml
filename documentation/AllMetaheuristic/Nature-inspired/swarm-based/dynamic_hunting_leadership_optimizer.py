import numpy as np

class DynamicHuntingLeadershipOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=30, max_iter=200, num_leaders=30, 
                 variant='V4', tolerance=5):
        """
        Initialize the Dynamic Hunting Leadership Optimizer (DHL).

        Parameters:
        - objective_function: Function to optimize (minimization problem).
        - dim: Number of dimensions (variables).
        - bounds: Tuple or list of (lower, upper) bounds for each dimension, or single (lb, ub) for all.
        - population_size: Number of search agents (N).
        - max_iter: Maximum number of iterations.
        - num_leaders: Initial number of leaders (NL).
        - variant: DHL variant ('V1', 'V2', 'V3', 'V4').
        - tolerance: Tolerance percentage for leader reduction in V4 (% of iterations).
        """
        self.objective_function = objective_function
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter
        self.num_leaders = num_leaders
        self.variant = variant.upper()
        self.tolerance = tolerance

        # Validate variant
        if self.variant not in ['V1', 'V2', 'V3', 'V4']:
            raise ValueError("Variant must be one of 'V1', 'V2', 'V3', 'V4'.")

        # Process bounds
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2 and not isinstance(bounds[0], (list, tuple)):
            # Single bound for all dimensions
            self.bounds = np.array([[bounds[0], bounds[1]]] * dim)
        else:
            # Different bounds per dimension
            self.bounds = np.array(bounds)
            if self.bounds.shape[0] != dim:
                raise ValueError("Bounds must match the number of dimensions or be a single (lb, ub) pair.")

        self.search_agents = None  # Population of search agents (SA)
        self.leaders_pos = None  # Leader positions (L_pos)
        self.leaders_fit = None  # Leader fitness values (L_fit)
        self.best_solution = None  # Best leader position (BS)
        self.best_value = float("inf")  # Best leader fitness (BF)
        self.convergence_curve = []  # Convergence curve
        self.num_leaders_history = []  # History of number of leaders

    def initialize_population(self):
        """Initialize the positions of search agents and leaders."""
        # Initialize leaders
        self.leaders_pos = self._initialize(self.num_leaders, self.dim, self.bounds)
        self.leaders_fit = np.full(self.num_leaders, float("inf"))

        # Initialize search agents
        self.search_agents = self._initialize(self.population_size, self.dim, self.bounds)

    def _initialize(self, num_agents, dim, bounds):
        """Helper function to initialize positions within bounds."""
        positions = np.zeros((num_agents, dim))
        for i in range(dim):
            lb, ub = bounds[i, 0], bounds[i, 1]
            positions[:, i] = np.random.uniform(lb, ub, num_agents)
        return positions

    def check_bounds(self, position):
        """Ensure position stays within bounds."""
        return np.clip(position, self.bounds[:, 0], self.bounds[:, 1])

    def evaluate_population(self):
        """Compute fitness values for the search agents."""
        return np.array([self.objective_function(agent) for agent in self.search_agents])

    def update_leaders(self, iter_idx):
        """Update leader positions and fitness based on search agents' fitness."""
        fitness = self.evaluate_population()
        leader_history = np.full(self.num_leaders, float("inf")) if iter_idx == 0 else self.leader_history[:, iter_idx-1]

        for i in range(self.population_size):
            sa_fit = fitness[i]
            for i_b in range(self.current_num_leaders):
                if sa_fit < self.leaders_fit[i_b]:
                    self.leaders_fit[i_b] = sa_fit
                    self.leaders_pos[i_b] = self.search_agents[i].copy()
                    break
            leader_history[i_b] = self.leaders_fit[i_b]

        return leader_history

    def update_num_leaders(self, iter_idx, leader_history):
        """Dynamically adjust the number of leaders based on the variant."""
        if self.variant == 'V1':
            # DHL_V1: Piecewise linear from 3 to NL, then 5 to 1
            AA = 1 - iter_idx / (self.max_iter / 2)
            if AA > 0:
                n_L = 3 + round(AA * (self.num_leaders - 3))
            else:
                AA = 1 - (iter_idx / self.max_iter)
                n_L = 1 + round(AA * 4)
        elif self.variant == 'V2':
            # DHL_V2: Linear from NL to 1
            AA = 1 - iter_idx / self.max_iter
            n_L = 1 + round(AA * (self.num_leaders - 1))
        elif self.variant == 'V3':
            # DHL_V3: Exponential from NL to 1
            AA = -iter_idx * 10 / self.max_iter
            n_L = self.num_leaders * np.exp(AA) + 1
            n_L = round(n_L)
        else:  # V4
            # DHL_V4: Tolerance-based reduction
            tol_iter = self.max_iter * self.tolerance / 100
            tol = [1e-5, tol_iter + 1, tol_iter]
            n_L = self.current_num_leaders
            if iter_idx >= tol[1]:
                if leader_history[n_L - 1] == float("inf"):
                    n_L -= 1
                elif abs(leader_history[n_L - 1] - 
                         self.leader_history[n_L - 1, iter_idx - int(tol[2])]) < tol[0]:
                    n_L -= 1

        # Ensure n_L is at least 1
        n_L = max(1, min(n_L, self.num_leaders))
        self.current_num_leaders = n_L
        self.num_leaders_history.append(n_L)

    def update_positions(self, iter_idx):
        """Update search agents' positions based on leaders."""
        a = 2 - iter_idx * (2 / self.max_iter)  # Linearly decrease from 2 to 0

        for i in range(self.population_size):
            new_pos = np.zeros(self.dim)
            for j in range(self.dim):
                XX = np.zeros(self.current_num_leaders)
                for i_x in range(self.current_num_leaders):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.leaders_pos[i_x, j] - self.search_agents[i, j])
                    XX[i_x] = self.leaders_pos[i_x, j] - A1 * D_alpha
                new_pos[j] = np.mean(XX)
            self.search_agents[i] = self.check_bounds(new_pos)

    def optimize(self):
        """Run the Dynamic Hunting Leadership Optimization."""
        self.initialize_population()
        self.current_num_leaders = self.num_leaders
        self.leader_history = np.zeros((self.num_leaders, self.max_iter))

        for iter_idx in range(self.max_iter):
            # Update leaders
            leader_history = self.update_leaders(iter_idx)
            self.leader_history[:, iter_idx] = leader_history

            # Update number of leaders
            self.update_num_leaders(iter_idx, leader_history)

            # Update positions
            self.update_positions(iter_idx)

            # Update best solution
            if self.leaders_fit[0] < self.best_value:
                self.best_value = self.leaders_fit[0]
                self.best_solution = self.leaders_pos[0].copy()

            # Record convergence
            self.convergence_curve.append(self.best_value)

            print(f"Iteration {iter_idx + 1}: Best Value = {self.best_value}, Num Leaders = {self.current_num_leaders}")

        return self.best_solution, self.best_value, self.convergence_curve, self.num_leaders_history

# Example usage
if __name__ == "__main__":
    def sphere_function(x):
        """Example objective function: Sphere function."""
        return np.sum(x ** 2)

    dim = 30
    bounds = [(-100, 100)] * dim  # Bounds for each dimension
    for variant in ['V1', 'V2', 'V3', 'V4']:
        print(f"\nRunning DHL_{variant}")
        optimizer = DynamicHuntingLeadershipOptimizer(
            objective_function=sphere_function,
            dim=dim,
            bounds=bounds,
            population_size=30,
            max_iter=200,
            num_leaders=30,
            variant=variant,
            tolerance=5
        )
        best_solution, best_value, convergence_curve, num_leaders_history = optimizer.optimize()
        print(f"DHL_{variant} Best Solution: {best_solution}")
        print(f"DHL_{variant} Best Value: {best_value}")
