import numpy as np

class BacterialColonyOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=1000,
                 chemotaxis_step_max=0.2, chemotaxis_step_min=0.01, elimination_ratio=0.2,
                 reproduction_threshold=0.5, migration_probability=0.1, communication_prob=0.5):
        """
        Initialize the Bacterial Colony Optimization (BCO) algorithm.

        Parameters:
        - objective_function: Function to optimize (minimization).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of bacteria (solutions).
        - max_iter: Maximum number of iterations.
        - chemotaxis_step_max: Maximum chemotaxis step size.
        - chemotaxis_step_min: Minimum chemotaxis step size.
        - elimination_ratio: Percentage of worst bacteria eliminated per iteration.
        - reproduction_threshold: Energy threshold for reproduction eligibility.
        - migration_probability: Probability of migration for bacteria.
        - communication_prob: Probability of information exchange between bacteria.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.chemotaxis_step_max = chemotaxis_step_max
        self.chemotaxis_step_min = chemotaxis_step_min
        self.elimination_ratio = elimination_ratio
        self.reproduction_threshold = reproduction_threshold
        self.migration_probability = migration_probability
        self.communication_prob = communication_prob

        self.bacteria = None  # Population of bacteria (solutions)
        self.energy_levels = None  # Energy levels based on fitness
        self.best_solution = None
        self.best_value = float("inf")
        self.global_best = None
        self.history = []

    def initialize_bacteria(self):
        """ Generate initial bacteria population randomly """
        self.bacteria = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                         (self.population_size, self.dim))
        self.energy_levels = np.zeros(self.population_size)

    def evaluate_bacteria(self):
        """ Compute fitness values for the bacteria population """
        return np.array([self.objective_function(bacterium) for bacterium in self.bacteria])

    def compute_chemotaxis_step(self, iteration):
        """ Compute adaptive chemotaxis step size (linearly decreasing) """
        return self.chemotaxis_step_min + (self.chemotaxis_step_max - self.chemotaxis_step_min) * \
               ((self.max_iter - iteration) / self.max_iter)

    def chemotaxis_and_communication(self, iteration):
        """ Perform chemotaxis and communication phase """
        new_bacteria = self.bacteria.copy()
        fitness = self.evaluate_bacteria()
        chemotaxis_step = self.compute_chemotaxis_step(iteration)

        for i in range(self.population_size):
            # Tumbling (exploration with random direction)
            if np.random.rand() < 0.5:
                turbulent = np.random.randn(self.dim)
                direction = chemotaxis_step * (
                    0.5 * (self.global_best - self.bacteria[i]) +
                    0.5 * (self.bacteria[i] - self.bacteria[i]) + turbulent
                )
            # Swimming (exploitation towards best solutions)
            else:
                direction = chemotaxis_step * (
                    0.5 * (self.global_best - self.bacteria[i]) +
                    0.5 * (self.bacteria[i] - self.bacteria[i])
                )

            new_bacteria[i] += direction
            new_bacteria[i] = np.clip(new_bacteria[i], self.bounds[:, 0], self.bounds[:, 1])

            # Communication (individual or group exchange)
            if np.random.rand() < self.communication_prob:
                # Individual exchange (dynamic neighbor or random)
                if np.random.rand() < 0.5:
                    # Dynamic neighbor oriented
                    neighbor_idx = (i + np.random.choice([-1, 1])) % self.population_size
                else:
                    # Random oriented
                    neighbor_idx = np.random.randint(self.population_size)
                
                neighbor_fitness = self.objective_function(self.bacteria[neighbor_idx])
                if neighbor_fitness < fitness[i]:
                    new_bacteria[i] = self.bacteria[neighbor_idx].copy()
                else:
                    # Group exchange (replace with global best if worse)
                    if fitness[i] > self.best_value:
                        new_bacteria[i] = self.global_best.copy()

        self.bacteria = new_bacteria

    def elimination_and_reproduction(self):
        """ Perform elimination and reproduction phase """
        fitness = self.evaluate_bacteria()
        sorted_indices = np.argsort(fitness)
        
        # Update energy levels based on fitness
        self.energy_levels = 1 / (1 + fitness)  # Higher fitness -> higher energy

        # Elimination: remove worst bacteria
        num_eliminate = int(self.elimination_ratio * self.population_size)
        for idx in sorted_indices[-num_eliminate:]:
            if self.energy_levels[idx] < self.reproduction_threshold:
                self.bacteria[idx] = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)

        # Reproduction: replicate best bacteria
        num_reproduce = num_eliminate // 2
        for idx in sorted_indices[:num_reproduce]:
            if self.energy_levels[idx] >= self.reproduction_threshold:
                self.bacteria[sorted_indices[-(idx + 1)]] = self.bacteria[idx].copy()

    def migration(self):
        """ Perform migration phase """
        for i in range(self.population_size):
            if np.random.rand() < self.migration_probability:
                # Check migration conditions (e.g., low energy or high similarity)
                if self.energy_levels[i] < self.reproduction_threshold or \
                   np.linalg.norm(self.bacteria[i] - self.global_best) < 1e-3:
                    self.bacteria[i] = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)

    def optimize(self):
        """ Run the Bacterial Colony Optimization algorithm """
        self.initialize_bacteria()
        self.global_best = self.bacteria[0].copy()

        for iteration in range(self.max_iter):
            fitness = self.evaluate_bacteria()
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.bacteria[min_idx].copy()
                self.best_value = fitness[min_idx]
                self.global_best = self.best_solution.copy()

            # Chemotaxis and Communication
            self.chemotaxis_and_communication(iteration)

            # Elimination and Reproduction
            self.elimination_and_reproduction()

            # Migration
            self.migration()

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

# Example usage:
if __name__ == "__main__":
    # Example objective function (Sphere function)
    def sphere_function(x):
        return np.sum(x**2)

    # Parameters
    dim = 15
    bounds = [(-100, 100)] * dim
    bco = BacterialColonyOptimizer(
        objective_function=sphere_function,
        dim=dim,
        bounds=bounds,
        population_size=100,
        max_iter=2000,
        chemotaxis_step_max=0.2,
        chemotaxis_step_min=0.01,
        elimination_ratio=0.2,
        reproduction_threshold=0.5,
        migration_probability=0.1,
        communication_prob=0.5
    )

    best_solution, best_value, history = bco.optimize()
    print(f"Best Solution: {best_solution}")
    print(f"Best Value: {best_value}")
