import numpy as np

class BumbleBeeMatingOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100,
                 queen_factor=0.3, drone_selection=0.2, worker_improvement=1.35,
                 brood_distribution=0.46, mating_resistance=1.2, replacement_ratio=0.23):
        """
        Initialize the Bumble Bee Mating Optimization (BBMO) optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of bees (solutions).
        - max_iter: Maximum number of iterations.
        - queen_factor: Controls queen's influence in mating.
        - drone_selection: Controls drone selection probability.
        - worker_improvement: Improvement factor for worker phase.
        - brood_distribution: Distribution coefficient for broods.
        - mating_resistance: Resistance coefficient for mating process.
        - replacement_ratio: Percentage of worst solutions replaced per iteration.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.queen_factor = queen_factor
        self.drone_selection = drone_selection
        self.worker_improvement = worker_improvement
        self.brood_distribution = brood_distribution
        self.mating_resistance = mating_resistance
        self.replacement_ratio = replacement_ratio

        self.bees = None  # Population of bees (solutions)
        self.queen = None  # Best solution (queen bee)
        self.queen_value = float("inf")
        self.history = []

    def initialize_bees(self):
        """ Generate initial bee population randomly """
        self.bees = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                      (self.population_size, self.dim))

    def evaluate_bees(self):
        """ Compute fitness values for the bee population """
        return np.array([self.objective_function(bee) for bee in self.bees])

    def queen_selection_phase(self):
        """ Select the best bee as the queen """
        fitness = self.evaluate_bees()
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.queen_value:
            self.queen = self.bees[min_idx]
            self.queen_value = fitness[min_idx]

    def mating_phase(self, index):
        """ Simulate mating process with crossover operators """
        r1, r2 = np.random.rand(), np.random.rand()
        # Modified: Select crossover types based on dimension
        if self.dim >= 3:
            crossover_type = np.random.choice(['one_point', 'two_point', 'three_point', 'blend_alpha'])
        elif self.dim == 2:
            crossover_type = np.random.choice(['one_point', 'blend_alpha'])
        else:  # dim == 1
            crossover_type = 'blend_alpha'  # Only blend_alpha is safe for dim=1
        drone = self.bees[np.random.randint(0, self.population_size)]

        if r1 < self.drone_selection:
            Vi = (self.queen_factor ** (2/3)) / self.mating_resistance * r1
        else:
            Vi = (self.queen_factor ** (2/3)) / self.mating_resistance * r2

        if crossover_type == 'one_point' and self.dim >= 2:
            cut = np.random.randint(1, self.dim)
            new_solution = np.concatenate((self.queen[:cut], drone[cut:]))
        elif crossover_type == 'two_point' and self.dim >= 3:
            cut1, cut2 = np.sort(np.random.choice(self.dim, 2, replace=False))
            new_solution = np.concatenate((self.queen[:cut1], drone[cut1:cut2], self.queen[cut2:]))
        elif crossover_type == 'three_point' and self.dim >= 4:
            cuts = np.sort(np.random.choice(self.dim, 3, replace=False))
            new_solution = np.concatenate((self.queen[:cuts[0]], drone[cuts[0]:cuts[1]],
                                          self.queen[cuts[1]:cuts[2]], drone[cuts[2]:]))
        else:  # blend_alpha (or fallback for low dim)
            new_solution = self.blend_alpha_crossover(self.queen, drone)

        new_solution = self.queen + (new_solution - self.queen) * Vi * np.random.rand(self.dim)
        return np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])

    def blend_alpha_crossover(self, queen, drone, alpha=0.5):
        """ Implement blend-alpha crossover for continuous optimization """
        # Generate offspring by blending queen and drone within an extended range
        lower = np.minimum(queen, drone) - alpha * abs(queen - drone)
        upper = np.maximum(queen, drone) + alpha * abs(queen - drone)
        new_solution = np.random.uniform(lower, upper, self.dim)
        return new_solution

    def worker_phase(self, index):
        """ Simulate worker phase for local search and brood improvement """
        r3, r4 = np.random.rand(), np.random.rand()
        CFR = 9.435 * np.random.gamma(0.85, 2.5)

        if r3 < self.brood_distribution:
            Vi2 = (self.worker_improvement ** (2/3)) / (2 * CFR) * r3
        else:
            Vi2 = (self.worker_improvement ** (2/3)) / (2 * CFR) * r4

        Improve = np.sign(self.queen_value - self.evaluate_bees()[index]) * \
                  (self.queen - self.bees[index]) * np.random.rand(self.dim)

        new_solution = self.queen + (self.queen - self.bees[index]) * Vi2 + Improve
        return np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])

    def replacement_phase(self):
        """ Replace worst solutions with new random solutions """
        fitness = self.evaluate_bees()
        worst_indices = np.argsort(fitness)[-int(self.replacement_ratio * self.population_size):]
        for i in worst_indices:
            self.bees[i] = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)

    def optimize(self):
        """ Run the Bumble Bee Mating Optimization """
        self.initialize_bees()
        for generation in range(self.max_iter):
            # Queen selection
            self.queen_selection_phase()

            # Mating phase (global search)
            for i in range(self.population_size):
                self.bees[i] = self.mating_phase(i)

            # Worker phase (local search)
            for i in range(self.population_size):
                self.bees[i] = self.worker_phase(i)

            # Replacement phase
            self.replacement_phase()

            self.history.append((generation, self.queen.copy(), self.queen_value))
            print(f"Iteration {generation + 1}: Best Value = {self.queen_value}")

        return self.queen, self.queen_value, self.history
