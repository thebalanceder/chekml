import numpy as np

class ChickenSwarmOptimizer:
    """
    Chicken Swarm Optimization (CSO) algorithm for unconstrained optimization problems.
    Based on the MATLAB implementation by Xian-Bing Meng.
    """
    def __init__(self, objective_function, dim, bounds, population_size=100, max_iter=100, 
                 update_freq=10, rooster_ratio=0.15, hen_ratio=0.7, mother_ratio=0.5):
        """
        Initialize the CSO optimizer.

        Parameters:
        - objective_function: Function to optimize (minimize).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Total number of chickens (solutions).
        - max_iter: Maximum number of iterations.
        - update_freq: How often the swarm hierarchy is updated (G).
        - rooster_ratio: Proportion of roosters in the population.
        - hen_ratio: Proportion of hens in the population.
        - mother_ratio: Proportion of hens that are mothers.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.update_freq = update_freq
        self.rooster_ratio = rooster_ratio
        self.hen_ratio = hen_ratio
        self.mother_ratio = mother_ratio

        # Calculate group sizes
        self.rooster_num = int(np.round(population_size * rooster_ratio))
        self.hen_num = int(np.round(population_size * hen_ratio))
        self.chick_num = population_size - self.rooster_num - self.hen_num
        self.mother_num = int(np.round(self.hen_num * mother_ratio))

        # Initialize population and fitness
        self.positions = None  # Chicken positions (solutions)
        self.fitness = None  # Current fitness values
        self.best_positions = None  # Individual best positions
        self.best_fitness = None  # Individual best fitness values
        self.global_best_position = None  # Global best position
        self.global_best_fitness = float("inf")
        self.history = []

    def initialize_population(self):
        """Generate initial chicken positions randomly within bounds."""
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        self.positions = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.fitness = np.array([self.objective_function(x) for x in self.positions])
        self.best_positions = self.positions.copy()
        self.best_fitness = self.fitness.copy()
        min_idx = np.argmin(self.fitness)
        self.global_best_position = self.positions[min_idx].copy()
        self.global_best_fitness = self.fitness[min_idx]

    def apply_bounds(self, position):
        """Apply lower and upper bounds to a position."""
        return np.clip(position, self.bounds[:, 0], self.bounds[:, 1])

    def randi_tabu(self, min_val, max_val, tabu, dim=1):
        """Generate random integers excluding a tabu value."""
        values = []
        while len(values) < dim:
            temp = np.random.randint(min_val, max_val + 1)
            if temp != tabu and temp not in values:
                values.append(temp)
        return values[0] if dim == 1 else values

    def randperm_f(self, range_size, dim):
        """Generate a permutation, extending randperm for larger dimensions."""
        temp = np.random.permutation(range_size)
        if dim > range_size:
            extra = np.random.randint(1, range_size + 1, dim - range_size)
            return np.concatenate([temp, extra])
        return temp[:dim]

    def update_rooster(self, idx, sort_indices):
        """Update rooster position based on CSO rules."""
        another_rooster = self.randi_tabu(1, self.rooster_num, idx + 1)
        another_idx = sort_indices[another_rooster - 1]
        curr_idx = sort_indices[idx]

        if self.best_fitness[curr_idx] <= self.best_fitness[another_idx]:
            sigma = 1
        else:
            sigma = np.exp((self.best_fitness[another_idx] - self.best_fitness[curr_idx]) /
                           (abs(self.best_fitness[curr_idx]) + np.finfo(float).tiny))

        new_position = self.best_positions[curr_idx] * (1 + sigma * np.random.randn(self.dim))
        return self.apply_bounds(new_position)

    def update_hen(self, idx, sort_indices, mate):
        """Update hen position based on CSO rules."""
        mate_idx = sort_indices[mate[idx - self.rooster_num] - 1]
        other = self.randi_tabu(1, idx, mate[idx - self.rooster_num], 1)
        other_idx = sort_indices[other - 1]
        curr_idx = sort_indices[idx - 1]

        c1 = np.exp((self.best_fitness[curr_idx] - self.best_fitness[mate_idx]) /
                    (abs(self.best_fitness[curr_idx]) + np.finfo(float).tiny))
        c2 = np.exp(-self.best_fitness[curr_idx] + self.best_fitness[other_idx])

        new_position = (self.best_positions[curr_idx] +
                        (self.best_positions[mate_idx] - self.best_positions[curr_idx]) * c1 * np.random.rand(self.dim) +
                        (self.best_positions[other_idx] - self.best_positions[curr_idx]) * c2 * np.random.rand(self.dim))
        return self.apply_bounds(new_position)

    def update_chick(self, idx, sort_indices, mother_indices):
        """Update chick position based on CSO rules."""
        curr_idx = sort_indices[idx - 1]
        mother_idx = sort_indices[mother_indices[idx - self.rooster_num - self.hen_num] - 1]
        fl = np.random.uniform(0.5, 0.9)  # FL in [0.5, 0.9] as per CSO
        new_position = (self.best_positions[curr_idx] +
                        (self.best_positions[mother_idx] - self.best_positions[curr_idx]) * fl)
        return self.apply_bounds(new_position)

    def optimize(self):
        """Run the Chicken Swarm Optimization algorithm."""
        self.initialize_population()

        for t in range(self.max_iter):
            # Update swarm hierarchy every update_freq iterations or at start
            if t % self.update_freq == 0 or t == 0:
                sort_indices = np.argsort(self.best_fitness)
                mother_lib = np.random.permutation(self.hen_num)[:self.mother_num] + self.rooster_num
                mate = self.randperm_f(self.rooster_num, self.hen_num)
                mother_indices = mother_lib[np.random.randint(0, self.mother_num, self.chick_num)]

            # Update roosters
            for i in range(self.rooster_num):
                self.positions[sort_indices[i]] = self.update_rooster(i, sort_indices)
                self.fitness[sort_indices[i]] = self.objective_function(self.positions[sort_indices[i]])

            # Update hens
            for i in range(self.rooster_num, self.rooster_num + self.hen_num):
                self.positions[sort_indices[i]] = self.update_hen(i, sort_indices, mate)
                self.fitness[sort_indices[i]] = self.objective_function(self.positions[sort_indices[i]])

            # Update chicks
            for i in range(self.rooster_num + self.hen_num, self.population_size):
                self.positions[sort_indices[i]] = self.update_chick(i, sort_indices, mother_indices)
                self.fitness[sort_indices[i]] = self.objective_function(self.positions[sort_indices[i]])

            # Update individual and global bests
            for i in range(self.population_size):
                if self.fitness[i] < self.best_fitness[i]:
                    self.best_fitness[i] = self.fitness[i]
                    self.best_positions[i] = self.positions[i].copy()
                if self.best_fitness[i] < self.global_best_fitness:
                    self.global_best_fitness = self.best_fitness[i]
                    self.global_best_position = self.best_positions[i].copy()

            self.history.append((t, self.global_best_position.copy(), self.global_best_fitness))
            print(f"Iteration {t + 1}: Best Value = {self.global_best_fitness}")

        return self.global_best_position, self.global_best_fitness, self.history
