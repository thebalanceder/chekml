import numpy as np

class FruitFlyOptimizationAlgorithm:
    def __init__(self, objective_function, dim, bounds, population_size=20, max_iter=100,
                 search_range=1.0, mode='FOA', delta=0.5, chaos_factor=0.5):
        """
        Initialize the Fruit Fly Optimization Algorithm (FOA) optimizer.

        Parameters:
        - objective_function: Function to optimize (expects numpy array input, returns scalar).
        - dim: Number of dimensions (variables).
        - bounds: Array of (lower, upper) bounds for each dimension [(low, high), ...].
        - population_size: Number of fruit flies in the swarm.
        - max_iter: Maximum number of iterations.
        - search_range: Range for random direction and distance in smell-based search.
        - mode: Optimization mode ('FOA', 'MFOA', or 'CFOA').
        - delta: Parameter for MFOA to adjust smell concentration (0 ≤ delta ≤ 1).
        - chaos_factor: Scaling factor for CFOA chaotic search.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Shape: (dim, 2)
        self.population_size = population_size
        self.max_iter = max_iter
        self.search_range = search_range
        self.mode = mode.upper()
        self.delta = delta
        self.chaos_factor = chaos_factor

        self.swarm = None  # Population of fruit flies (solutions)
        self.swarm_axis = None  # Center of the swarm (X_axis, Y_axis, [Z_axis for MFOA])
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

        # Validate mode
        if self.mode not in ['FOA', 'MFOA', 'CFOA']:
            raise ValueError("Mode must be 'FOA', 'MFOA', or 'CFOA'")

    def initialize_swarm(self):
        """Initialize the fruit fly swarm with random positions."""
        # Initialize swarm center (X_axis, Y_axis, [Z_axis for MFOA])
        if self.mode == 'MFOA':
            self.swarm_axis = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (3,))
        else:
            self.swarm_axis = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (2,))

        # Initialize swarm positions
        self.swarm = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            self.swarm[i] = self.smell_based_search(i)

    def smell_based_search(self, index):
        """Simulate smell-based random search for a fruit fly."""
        if self.mode == 'CFOA':
            return self.chaos_based_search(index)
        elif self.mode == 'MFOA':
            return self.modified_smell_search(index)
        else:
            return self.basic_smell_search(index)

    def basic_smell_search(self, index):
        """Basic FOA smell-based search (2D)."""
        # Random direction and distance
        random_value = self.search_range * (2 * np.random.rand(self.dim) - 1)
        new_position = self.swarm_axis[:self.dim] + random_value
        return np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])

    def modified_smell_search(self, index):
        """Modified FOA smell-based search (3D)."""
        # Random direction and distance in 3D
        random_value = self.search_range * (2 * np.random.rand(3) - 1)
        new_position = self.swarm_axis + random_value
        # Project to problem dimensions (if dim < 3)
        new_position = new_position[:self.dim]
        return np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])

    def chaos_based_search(self, index):
        """CFOA chaos-based search."""
        # Chaotic sequence for global search
        chaotic_value = self.chaos_factor * (2 * np.random.rand(self.dim) - 1)
        max_diff = np.maximum(self.bounds[:, 1] - self.swarm_axis[:self.dim],
                              self.swarm_axis[:self.dim] - self.bounds[:, 0])
        new_position = self.swarm_axis[:self.dim] + chaotic_value * max_diff
        return np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])

    def evaluate_swarm(self):
        """Compute smell concentration (fitness) for the swarm."""
        smell = np.zeros(self.population_size)
        for i in range(self.population_size):
            # Calculate distance to origin (Dist)
            dist = np.sqrt(np.sum(self.swarm[i]**2))
            # Calculate smell concentration judgment value (S)
            if self.mode == 'MFOA':
                delta_adjust = dist * (0.5 - self.delta)
                s = 1 / dist + delta_adjust
            else:
                s = 1 / dist if dist != 0 else 1e-10  # Avoid division by zero
            # Evaluate fitness (smell concentration)
            smell[i] = self.objective_function(s * np.ones(self.dim))  # Scale to dimension
        return smell

    def optimize(self):
        """Run the Fruit Fly Optimization Algorithm."""
        self.initialize_swarm()
        smell_best = float("inf")  # Track best smell concentration

        for generation in range(self.max_iter):
            # Smell-based search
            for i in range(self.population_size):
                self.swarm[i] = self.smell_based_search(i)

            # Evaluate swarm
            smell = self.evaluate_swarm()
            min_idx = np.argmin(smell)  # Minimize fitness
            current_best_smell = smell[min_idx]

            # Update best solution
            if current_best_smell < smell_best:
                smell_best = current_best_smell
                self.best_solution = self.swarm[min_idx].copy()
                self.best_value = self.objective_function(self.best_solution)
                # Update swarm center (vision-based movement)
                if self.mode == 'MFOA':
                    self.swarm_axis = np.array([self.best_solution[0], self.best_solution[1], self.best_solution[0]]) \
                        if self.dim >= 2 else np.array([self.best_solution[0], self.best_solution[0], self.best_solution[0]])
                else:
                    self.swarm_axis = np.array([self.best_solution[0], self.best_solution[1]]) \
                        if self.dim >= 2 else np.array([self.best_solution[0], self.best_solution[0]])

            # CFOA local search (if applicable)
            if self.mode == 'CFOA' and generation >= self.max_iter // 2:
                self.local_search()

            self.history.append((generation, self.best_solution.copy(), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

    def local_search(self):
        """CFOA local search using golden ratio (from document page 131)."""
        for i in range(self.population_size):
            # Local search using golden ratio
            xl = 0.618 * self.swarm_axis[0] + 0.382 * self.swarm[i][0]
            yl = 0.618 * self.swarm_axis[1] + 0.382 * self.swarm[i][1]
            local_position = np.array([xl, yl])[:self.dim]
            local_position = np.clip(local_position, self.bounds[:, 0], self.bounds[:, 1])
            # Evaluate local position
            dist = np.sqrt(np.sum(local_position**2))
            s = 1 / dist if dist != 0 else 1e-10
            local_smell = self.objective_function(s * np.ones(self.dim))
            # Update if better
            if local_smell < self.objective_function(self.swarm[i]):
                self.swarm[i] = local_position
