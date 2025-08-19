import numpy as np

class SquirrelSearchAlgorithm:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, 
                 max_gliding_distance=1.11, min_gliding_distance=0.5, gliding_constant=1.9,
                 num_food_sources=4, hickory_nut_tree=1, acorn_nut_tree=3, no_food_trees=46):
        """
        Initialize the Squirrel Search Algorithm (SSA) optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of squirrels (solutions).
        - max_iter: Maximum number of iterations.
        - max_gliding_distance: Maximum gliding distance (Fmax).
        - min_gliding_distance: Minimum gliding distance (Fmin).
        - gliding_constant: Gliding constant (Gc).
        - num_food_sources: Number of food resources (nfs).
        - hickory_nut_tree: Number of hickory nut trees (hnt).
        - acorn_nut_tree: Number of acorn nut trees (ant).
        - no_food_trees: Number of no-food trees (noft).
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.Fmax = max_gliding_distance
        self.Fmin = min_gliding_distance
        self.Gc = gliding_constant
        self.nfs = num_food_sources
        self.hnt = hickory_nut_tree
        self.ant = acorn_nut_tree
        self.noft = no_food_trees

        self.squirrels = None  # Population of squirrels (solutions)
        self.best_solution = None
        self.best_value = float("inf")
        self.convergence_curve = np.zeros(max_iter)
        self.fitness = np.zeros(population_size)
        self.velocities = np.zeros((population_size, dim))
        self.gliding_distances = np.zeros(population_size)
        self.pulse_flying_rate = np.random.rand(population_size)

    def initialize_squirrels(self):
        """ Generate initial squirrel positions randomly """
        if len(self.bounds.shape) == 1:  # Single boundary for all dimensions
            self.squirrels = np.random.rand(self.population_size, self.dim) * \
                           (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        else:  # Different boundaries for each dimension
            self.squirrels = np.zeros((self.population_size, self.dim))
            for i in range(self.dim):
                ub_i, lb_i = self.bounds[i, 1], self.bounds[i, 0]
                self.squirrels[:, i] = np.random.rand(self.population_size) * (ub_i - lb_i) + lb_i

        # Ensure initial positions are within bounds
        self.squirrels = np.clip(self.squirrels, self.bounds[:, 0], self.bounds[:, 1])

        # Evaluate initial fitness
        for i in range(self.population_size):
            self.fitness[i] = self.objective_function(self.squirrels[i, :])

        # Randomly assign tree types: 1 (acorn), 2 (normal), 3 (hickory)
        self.tree_types = np.random.randint(1, 4, size=self.population_size)

        # Find initial best solution
        min_idx = np.argmin(self.fitness)
        self.best_value = self.fitness[min_idx]
        self.best_solution = self.squirrels[min_idx, :].copy()

    def enforce_bounds(self, position):
        """ Enforce boundary constraints on a single position """
        return np.clip(position, self.bounds[:, 0], self.bounds[:, 1])

    def update_squirrel_position(self, idx):
        """ Update squirrel position based on tree type """
        # Random gliding distance
        self.gliding_distances[idx] = self.Fmin + (self.Fmax - self.Fmin) * np.random.rand()

        # Update velocity based on tree type
        if self.tree_types[idx] == 1:  # Acorn tree
            self.velocities[idx, :] += self.gliding_distances[idx] * self.Gc * \
                                       (self.squirrels[idx, :] - self.best_solution) * 1
        elif self.tree_types[idx] == 2:  # Normal tree
            self.velocities[idx, :] += self.gliding_distances[idx] * self.Gc * \
                                       (self.squirrels[idx, :] - self.best_solution) * 2
        else:  # Hickory tree
            self.velocities[idx, :] += self.gliding_distances[idx] * self.Gc * \
                                       (self.squirrels[idx, :] - self.best_solution) * 3

        # Update position
        self.squirrels[idx, :] += self.velocities[idx, :]

        # Enforce bounds after velocity update
        self.squirrels[idx, :] = self.enforce_bounds(self.squirrels[idx, :])

        # Random flying condition
        if np.random.rand() > self.pulse_flying_rate[idx]:
            eps = -1 + (1 - (-1)) * np.random.rand()
            A = np.random.rand(self.population_size, 1)
            self.squirrels[idx, :] = self.best_solution + eps * np.mean(A)
            # Enforce bounds after random flying
            self.squirrels[idx, :] = self.enforce_bounds(self.squirrels[idx, :])

        # Evaluate new fitness
        new_fitness = self.objective_function(self.squirrels[idx, :])
        if new_fitness <= self.best_value:
            self.best_solution = self.squirrels[idx, :].copy()
            self.best_value = new_fitness

        return new_fitness

    def optimize(self):
        """ Run the Squirrel Search Algorithm """
        self.initialize_squirrels()
        for iteration in range(self.max_iter):
            for i in range(self.population_size):
                self.fitness[i] = self.update_squirrel_position(i)

            # Store convergence curve
            self.convergence_curve[iteration] = self.best_value

            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.convergence_curve
