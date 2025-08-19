import numpy as np

class MothFlameOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=30, max_iter=1000):
        """
        Initialize the Moth-Flame Optimization (MFO) algorithm.

        Parameters:
        - objective_function: Function to optimize (minimize).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension or single pair for all.
        - population_size: Number of moths (search agents).
        - max_iter: Maximum number of iterations.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...] or single [low, high]
        self.population_size = population_size
        self.max_iter = max_iter

        # Handle bounds: single pair or per dimension
        if self.bounds.ndim == 1:
            self.lb = np.full(dim, self.bounds[0])
            self.ub = np.full(dim, self.bounds[1])
        else:
            self.lb = self.bounds[:, 0]
            self.ub = self.bounds[:, 1]

        self.moth_pos = None  # Moth positions (population)
        self.best_flame_pos = None  # Best solution found
        self.best_flame_score = float("inf")  # Best fitness value
        self.convergence_curve = np.zeros(max_iter)  # History of best scores

    def initialize_moths(self):
        """Initialize the positions of moths randomly within bounds."""
        self.moth_pos = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))

    def evaluate_moths(self):
        """Compute fitness values for all moths."""
        return np.array([self.objective_function(pos) for pos in self.moth_pos])

    def clip_to_bounds(self):
        """Ensure moth positions stay within bounds."""
        self.moth_pos = np.clip(self.moth_pos, self.lb, self.ub)

    def optimize(self):
        """Run the Moth-Flame Optimization algorithm."""
        print("MFO is optimizing your problem")
        self.initialize_moths()
        previous_population = None
        previous_fitness = None
        best_flames = None
        best_flame_fitness = None

        for iteration in range(self.max_iter):
            # Number of flames (Eq. 3.14)
            flame_no = round(self.population_size - iteration * ((self.population_size - 1) / self.max_iter))

            # Clip moth positions to bounds
            self.clip_to_bounds()

            # Calculate fitness
            moth_fitness = self.evaluate_moths()

            if iteration == 0:
                # Sort initial population
                sorted_indices = np.argsort(moth_fitness)
                fitness_sorted = moth_fitness[sorted_indices]
                sorted_population = self.moth_pos[sorted_indices]

                # Initialize flames
                best_flames = sorted_population
                best_flame_fitness = fitness_sorted
            else:
                # Combine previous and best flames
                double_population = np.vstack((previous_population, best_flames))
                double_fitness = np.concatenate((previous_fitness, best_flame_fitness))

                # Sort combined population
                sorted_indices = np.argsort(double_fitness)
                double_fitness_sorted = double_fitness[sorted_indices]
                double_sorted_population = double_population[sorted_indices]

                # Update flames
                fitness_sorted = double_fitness_sorted[:self.population_size]
                sorted_population = double_sorted_population[:self.population_size]
                best_flames = sorted_population
                best_flame_fitness = fitness_sorted

            # Update best flame
            self.best_flame_score = fitness_sorted[0]
            self.best_flame_pos = sorted_population[0].copy()

            # Store previous population
            previous_population = self.moth_pos.copy()
            previous_fitness = moth_fitness.copy()

            # Linearly decrease 'a' from -1 to -2 (Eq. 3.12)
            a = -1 + iteration * (-1 / self.max_iter)

            # Update moth positions
            for i in range(self.population_size):
                for j in range(self.dim):
                    if i < flame_no:  # Update w.r.t. corresponding flame
                        distance_to_flame = abs(sorted_population[i, j] - self.moth_pos[i, j])
                        b = 1
                        t = (a - 1) * np.random.rand() + 1
                        self.moth_pos[i, j] = (distance_to_flame * np.exp(b * t) * 
                                               np.cos(t * 2 * np.pi) + sorted_population[i, j])
                    else:  # Update w.r.t. best flame
                        distance_to_flame = abs(sorted_population[flame_no - 1, j] - self.moth_pos[i, j])
                        b = 1
                        t = (a - 1) * np.random.rand() + 1
                        self.moth_pos[i, j] = (distance_to_flame * np.exp(b * t) * 
                                               np.cos(t * 2 * np.pi) + sorted_population[flame_no - 1, j])

            # Store convergence
            self.convergence_curve[iteration] = self.best_flame_score

            # Display progress every 50 iterations
            if iteration % 50 == 0:
                print(f"At iteration {iteration}, the best fitness is {self.best_flame_score}")

        return self.best_flame_pos, self.best_flame_score, self.convergence_curve

