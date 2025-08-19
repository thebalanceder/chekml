import numpy as np

class ArtificialRootForagingOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, 
                 branching_threshold=0.6, max_branching=5, min_branching=1, 
                 initial_std=1.0, final_std=0.01, max_elongation=0.1):
        """
        Initialize the Hybrid Artificial Root Foraging Optimizer (HARFO).

        Parameters:
        - objective_function: Function to optimize (minimization problem).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of root apices (solutions).
        - max_iter: Maximum number of iterations.
        - branching_threshold: Threshold for auxin concentration to trigger branching.
        - max_branching: Maximum number of new roots per branching event.
        - min_branching: Minimum number of new roots per branching event.
        - initial_std: Initial standard deviation for branching.
        - final_std: Final standard deviation for branching.
        - max_elongation: Maximum elongation length for lateral root growth.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.branching_threshold = branching_threshold
        self.max_branching = max_branching
        self.min_branching = min_branching
        self.initial_std = initial_std
        self.final_std = final_std
        self.max_elongation = max_elongation

        self.roots = None  # Population of root apices (solutions)
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_roots(self):
        """ Generate initial root apices randomly within bounds """
        self.roots = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                       (self.population_size, self.dim))

    def evaluate_fitness(self):
        """ Compute fitness values for the root population """
        return np.array([self.objective_function(root) for root in self.roots])

    def calculate_auxin_concentration(self, fitness):
        """ Calculate auxin concentration for each root based on fitness """
        f_min = np.min(fitness)
        f_max = np.max(fitness)
        if f_max == f_min:
            f_normalized = np.ones_like(fitness) / len(fitness)
        else:
            f_normalized = (fitness - f_min) / (f_max - f_min)
        auxin = f_normalized / np.sum(f_normalized)
        return auxin

    def construct_von_neumann_topology(self, current_pop_size):
        """ Construct Von Neumann topology for root-to-root communication """
        # Ensure topology accommodates current population size
        rows = int(np.sqrt(current_pop_size))
        cols = (current_pop_size + rows - 1) // rows  # Ceiling division
        topology = np.zeros((current_pop_size, 4), dtype=int) - 1  # Initialize with -1 for invalid neighbors
        for i in range(current_pop_size):
            row, col = divmod(i, cols)
            # Left neighbor
            if col > 0:
                topology[i, 0] = i - 1
            # Right neighbor
            if col < cols - 1 and i + 1 < current_pop_size:
                topology[i, 1] = i + 1
            # Up neighbor
            if row > 0:
                topology[i, 2] = i - cols
            # Down neighbor
            if row < rows - 1 and i + cols < current_pop_size:
                topology[i, 3] = i + cols
        return topology

    def main_root_regrowth(self, root_idx, local_best):
        """ Apply regrowing operator for main roots """
        local_inertia = 0.5  # Local learning inertia
        rand_coeff = np.random.rand(self.dim)
        new_position = self.roots[root_idx] + local_inertia * rand_coeff * (local_best - self.roots[root_idx])
        return np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])

    def main_root_branching(self, root_idx, auxin):
        """ Apply branching operator for main roots if condition is met """
        if auxin[root_idx] > self.branching_threshold:
            R1 = np.random.rand()
            num_new_roots = int(R1 * auxin[root_idx] * (self.max_branching - self.min_branching) + self.min_branching)
            current_iter = len(self.history)
            std = ((self.max_iter - current_iter) / self.max_iter) * (self.initial_std - self.final_std) + self.final_std
            new_roots = np.random.normal(self.roots[root_idx], std, (num_new_roots, self.dim))
            new_roots = np.clip(new_roots, self.bounds[:, 0], self.bounds[:, 1])
            return new_roots
        return None

    def lateral_root_growth(self, root_idx):
        """ Apply random walk for lateral roots """
        rand_length = np.random.rand() * self.max_elongation
        random_vector = np.random.randn(self.dim)
        growth_angle = random_vector / np.sqrt(np.sum(random_vector**2))
        new_position = self.roots[root_idx] + rand_length * growth_angle
        return np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])

    def dead_root_elimination(self, auxin):
        """ Remove roots with low auxin concentration """
        elimination_threshold = np.percentile(auxin, 10)  # Remove bottom 10%
        keep_mask = auxin > elimination_threshold
        self.roots = self.roots[keep_mask]
        return keep_mask

    def optimize(self):
        """ Run the Hybrid Artificial Root Foraging Optimization (HARFO) """
        self.initialize_roots()

        for iteration in range(self.max_iter):
            # Recompute topology for current population size
            current_pop_size = len(self.roots)
            topology = self.construct_von_neumann_topology(current_pop_size)

            fitness = self.evaluate_fitness()
            auxin = self.calculate_auxin_concentration(fitness)
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.roots[min_idx].copy()
                self.best_value = fitness[min_idx]

            # Divide roots into main and lateral based on auxin concentration
            main_root_mask = auxin > np.median(auxin)
            lateral_root_mask = ~main_root_mask

            # Process main roots
            new_roots = []
            for i in range(current_pop_size):
                if main_root_mask[i]:
                    # Find local best from Von Neumann neighbors
                    neighbor_indices = topology[i]
                    valid_neighbors = neighbor_indices[neighbor_indices >= 0]  # Filter out invalid neighbors
                    if len(valid_neighbors) > 0:
                        neighbor_fitness = fitness[valid_neighbors]
                        local_best_idx = valid_neighbors[np.argmin(neighbor_fitness)]
                        local_best = self.roots[local_best_idx]
                    else:
                        local_best = self.roots[i]  # Fallback to self if no valid neighbors

                    # Regrowing operator
                    self.roots[i] = self.main_root_regrowth(i, local_best)

                    # Branching operator
                    new_branch = self.main_root_branching(i, auxin)
                    if new_branch is not None:
                        new_roots.append(new_branch)

            # Process lateral roots
            for i in range(current_pop_size):
                if lateral_root_mask[i]:
                    self.roots[i] = self.lateral_root_growth(i)

            # Add new branched roots to population
            if new_roots:
                new_roots = np.vstack(new_roots)
                self.roots = np.vstack([self.roots, new_roots])
                # Adjust population size if exceeds limit
                if len(self.roots) > self.population_size:
                    fitness = self.evaluate_fitness()
                    keep_indices = np.argsort(fitness)[:self.population_size]
                    self.roots = self.roots[keep_indices]

            # Dead root elimination
            fitness = self.evaluate_fitness()
            auxin = self.calculate_auxin_concentration(fitness)
            keep_mask = self.dead_root_elimination(auxin)
            fitness = fitness[keep_mask]
            auxin = auxin[keep_mask]

            # Replenish population if necessary
            while len(self.roots) < self.population_size:
                new_root = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)
                self.roots = np.vstack([self.roots, new_root])

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
