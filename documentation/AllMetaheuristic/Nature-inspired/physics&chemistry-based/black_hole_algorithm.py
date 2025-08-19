import numpy as np

class BlackHoleOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100):
        """
        Initialize the Black Hole Algorithm optimizer.

        Parameters:
        - objective_function: Function to optimize (minimize).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of stars (solutions).
        - max_iter: Maximum number of iterations.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.stars = None  # Population of stars (solutions)
        self.black_hole = None  # Best solution (black hole position)
        self.black_hole_idx = None  # Index of black hole
        self.best_value = None  # Best fitness value
        self.fitness_history = []  # Track best fitness per iteration
        self.num_evaluations = 0  # Count function evaluations

    def initialize_stars(self):
        """Generate initial star positions randomly within bounds."""
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        self.stars = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = self.evaluate_stars()
        self.num_evaluations += self.population_size
        self.black_hole_idx = np.argmin(fitness)
        self.black_hole = self.stars[self.black_hole_idx].copy()
        self.best_value = fitness[self.black_hole_idx]

    def evaluate_stars(self):
        """Compute fitness values for all stars."""
        return np.array([self.objective_function(star) for star in self.stars])

    def bound_stars(self):
        """Ensure stars stay within bounds."""
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        for i in range(self.population_size):
            self.stars[i] = np.clip(self.stars[i], lb, ub)

    def update_star_positions(self):
        """Update star positions based on black hole attraction."""
        for i in range(self.population_size):
            if i != self.black_hole_idx:
                landa = np.random.rand()
                # Move star towards black hole
                self.stars[i] += np.random.rand(self.dim) * (self.black_hole - self.stars[i])

    def replace_with_better_black_hole(self):
        """Replace black hole if a star has better fitness."""
        fitness = self.evaluate_stars()
        self.num_evaluations += self.population_size
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.best_value:
            self.black_hole_idx = min_idx
            self.black_hole = self.stars[min_idx].copy()
            self.best_value = fitness[min_idx]

    def new_star_generation(self):
        """Replace stars that cross the event horizon with new random stars."""
        fitness = self.evaluate_stars()
        self.num_evaluations += self.population_size
        # Calculate event horizon radius
        R = fitness[self.black_hole_idx] / np.sum(fitness)
        distances = np.linalg.norm(self.stars - self.black_hole, axis=1)
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        for i in range(self.population_size):
            if distances[i] < R and i != self.black_hole_idx:
                # Generate new star at random location
                self.stars[i] = np.random.uniform(lb, ub, self.dim)
                fitness[i] = self.objective_function(self.stars[i])
                self.num_evaluations += 1

    def optimize(self, display_flag=True):
        """Run the Black Hole Algorithm optimization."""
        self.initialize_stars()
        for iteration in range(self.max_iter):
            # Update star positions
            self.update_star_positions()
            # Ensure stars are within bounds
            self.bound_stars()
            # Check for new black hole
            self.replace_with_better_black_hole()
            # Replace stars within event horizon
            self.new_star_generation()
            # Ensure stars are within bounds again
            self.bound_stars()
            # Store best fitness for this iteration
            self.fitness_history.append(self.best_value)
            if display_flag:
                print(f"Iteration {iteration + 1}: Best Fitness = {self.best_value}")

        return self.black_hole, self.best_value, self.fitness_history, self.num_evaluations

def run_multiple_trials(objective_function, dim, bounds, population_size=50, max_iter=500, runs=10, display_flag=True):
    """Run the Black Hole Algorithm multiple times and collect statistics."""
    best_positions = np.zeros((runs, dim))
    best_fitnesses = np.zeros(runs)
    fitness_evolutions = np.zeros((runs, max_iter))
    
    for run in range(runs):
        optimizer = BlackHoleOptimizer(objective_function, dim, bounds, population_size, max_iter)
        best_x, best_fitness, fitness_history, _ = optimizer.optimize(display_flag=display_flag)
        best_positions[run] = best_x
        best_fitnesses[run] = best_fitness
        fitness_evolutions[run] = fitness_history
    
    # Calculate statistics
    min_fitness = np.min(best_fitnesses)
    mean_fitness = np.mean(best_fitnesses)
    median_fitness = np.median(best_fitnesses)
    max_fitness = np.max(best_fitnesses)
    std_fitness = np.std(best_fitnesses)
    
    print(f"MIN={min_fitness:.6f}  MEAN={mean_fitness:.6f}  MEDIAN={median_fitness:.6f} "
          f"MAX={max_fitness:.6f}  SD={std_fitness:.6f}")
    
    return best_positions, best_fitnesses, fitness_evolutions
