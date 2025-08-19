import numpy as np

class AntLionOptimization:
    def __init__(self, objective_function, dim, bounds, population_size=40, max_iter=500):
        """
        Initialize the Ant Lion Optimizer (ALO).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: List of tuples [(lower, upper), ...] for each dimension or single values if same for all.
        - population_size: Number of search agents (antlions and ants).
        - max_iter: Maximum number of iterations.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter
        
        # Handle bounds
        if isinstance(bounds, (list, tuple, np.ndarray)) and len(bounds) == 2 and not isinstance(bounds[0], (list, tuple, np.ndarray)):
            self.bounds = np.array([[bounds[0], bounds[1]] for _ in range(dim)])
        else:
            self.bounds = np.array(bounds)
        
        self.antlion_positions = None
        self.ant_positions = None
        self.elite_antlion_position = None
        self.elite_antlion_fitness = float("inf")
        self.sorted_antlions = None
        self.history = []

    def initialize_positions(self):
        """ Generate initial random positions for antlions and ants """
        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]
        
        if len(lb) == 1:
            self.antlion_positions = np.random.uniform(lb[0], ub[0], (self.population_size, self.dim))
            self.ant_positions = np.random.uniform(lb[0], ub[0], (self.population_size, self.dim))
        else:
            self.antlion_positions = np.zeros((self.population_size, self.dim))
            self.ant_positions = np.zeros((self.population_size, self.dim))
            for i in range(self.dim):
                self.antlion_positions[:, i] = np.random.uniform(lb[i], ub[i], self.population_size)
                self.ant_positions[:, i] = np.random.uniform(lb[i], ub[i], self.population_size)

    def evaluate_fitness(self, positions):
        """ Compute fitness values for given positions """
        return np.array([self.objective_function(pos) for pos in positions])

    def roulette_wheel_selection(self, weights):
        """ Perform roulette wheel selection based on weights """
        accumulation = np.cumsum(weights)
        p = np.random.rand() * accumulation[-1]
        for idx, val in enumerate(accumulation):
            if val > p:
                return idx
        return 0  # Default to first index if no selection

    def random_walk_around_antlion(self, antlion, current_iter):
        """ Create random walks around a given antlion """
        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]

        # Adjust ratio I based on iteration
        I = 1
        if current_iter > self.max_iter / 10:
            I = 1 + 100 * (current_iter / self.max_iter)
        if current_iter > self.max_iter / 2:
            I = 1 + 1000 * (current_iter / self.max_iter)
        if current_iter > self.max_iter * 3 / 4:
            I = 1 + 10000 * (current_iter / self.max_iter)
        if current_iter > self.max_iter * 0.9:
            I = 1 + 100000 * (current_iter / self.max_iter)
        if current_iter > self.max_iter * 0.95:
            I = 1 + 1000000 * (current_iter / self.max_iter)

        # Decrease boundaries
        lb = lb / I
        ub = ub / I

        # Move interval around antlion
        if np.random.rand() < 0.5:
            lb = lb + antlion
        else:
            lb = -lb + antlion

        if np.random.rand() >= 0.5:
            ub = ub + antlion
        else:
            ub = -ub + antlion

        # Generate random walks
        walks = np.zeros((self.max_iter + 1, self.dim))
        for i in range(self.dim):
            X = np.cumsum(2 * (np.random.rand(self.max_iter) > 0.5) - 1)
            X = np.insert(X, 0, 0)  # Start at 0
            a, b = np.min(X), np.max(X)
            c, d = lb[i], ub[i]
            X_norm = ((X - a) * (d - c)) / (b - a + 1e-10) + c  # Normalize
            walks[:, i] = X_norm

        return walks

    def optimize(self):
        """ Run the Ant Lion Optimization algorithm """
        # Initialize positions
        self.initialize_positions()
        
        # Evaluate initial antlion fitness
        antlions_fitness = self.evaluate_fitness(self.antlion_positions)
        
        # Sort antlions
        sorted_indices = np.argsort(antlions_fitness)
        self.sorted_antlions = self.antlion_positions[sorted_indices]
        antlions_fitness = antlions_fitness[sorted_indices]
        
        # Set elite antlion
        self.elite_antlion_position = self.sorted_antlions[0].copy()
        self.elite_antlion_fitness = antlions_fitness[0]
        
        # Store initial best solution in history
        self.history.append((0, self.elite_antlion_position.copy(), self.elite_antlion_fitness))
        
        # Main loop
        for current_iter in range(1, self.max_iter):
            # Simulate random walks for each ant
            for i in range(self.population_size):
                # Select antlion using roulette wheel
                roulette_idx = self.roulette_wheel_selection(1 / (antlions_fitness + 1e-10))
                if roulette_idx == -1:
                    roulette_idx = 0
                
                # Random walk around selected antlion
                RA = self.random_walk_around_antlion(self.sorted_antlions[roulette_idx], current_iter)
                
                # Random walk around elite antlion
                RE = self.random_walk_around_antlion(self.elite_antlion_position, current_iter)
                
                # Update ant position (Equation 2.13)
                self.ant_positions[i] = (RA[current_iter] + RE[current_iter]) / 2
            
            # Boundary checking
            for i in range(self.population_size):
                self.ant_positions[i] = np.clip(self.ant_positions[i], self.bounds[:, 0], self.bounds[:, 1])
            
            # Evaluate ant fitness
            ants_fitness = self.evaluate_fitness(self.ant_positions)
            
            # Combine populations and update antlions
            double_population = np.vstack((self.sorted_antlions, self.ant_positions))
            double_fitness = np.concatenate((antlions_fitness, ants_fitness))
            
            sorted_indices = np.argsort(double_fitness)
            double_population = double_population[sorted_indices]
            double_fitness = double_fitness[sorted_indices]
            
            antlions_fitness = double_fitness[:self.population_size]
            self.sorted_antlions = double_population[:self.population_size]
            
            # Update elite if better solution found
            if antlions_fitness[0] < self.elite_antlion_fitness:
                self.elite_antlion_position = self.sorted_antlions[0].copy()
                self.elite_antlion_fitness = antlions_fitness[0]
            
            # Ensure elite is in population
            self.sorted_antlions[0] = self.elite_antlion_position
            antlions_fitness[0] = self.elite_antlion_fitness
            
            # Store history
            self.history.append((current_iter, self.elite_antlion_position.copy(), self.elite_antlion_fitness))
            
            # Display progress every 50 iterations
            if (current_iter + 1) % 50 == 0:
                print(f"At iteration {current_iter + 1}, the elite fitness is {self.elite_antlion_fitness}")

        return self.elite_antlion_position, self.elite_antlion_fitness, self.history

