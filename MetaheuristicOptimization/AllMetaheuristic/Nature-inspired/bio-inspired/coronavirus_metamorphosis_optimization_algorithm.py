import numpy as np

class CoronavirusMetamorphosisOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, 
                 mutation_rate=0.3, crossover_rate=0.5):
        """
        Initialize the CMOA optimizer.

        Parameters:
        - objective_function: Function to optimize (minimize).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of viruses (solutions).
        - max_iter: Maximum number of iterations.
        - mutation_rate: Probability of non-genetic mutation.
        - crossover_rate: Probability of genetic recombination.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.viruses = None  # Population of viruses (solutions)
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_viruses(self):
        """Generate initial virus population randomly."""
        self.viruses = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                        (self.population_size, self.dim))

    def evaluate_viruses(self):
        """Compute fitness values for the virus population."""
        return np.array([self.objective_function(virus) for virus in self.viruses])

    def find_closest_virus(self, index):
        """Find the index of the closest virus to the virus at the given index."""
        current_virus = self.viruses[index]
        distances = np.abs(np.sum(self.viruses - current_virus, axis=1))
        distances[index] = np.inf  # Exclude the virus itself
        return np.argmin(distances)

    def genetic_recombination(self, index):
        """Simulate genetic recombination by moving towards the closest virus."""
        if np.random.rand() < self.crossover_rate:
            closest_idx = self.find_closest_virus(index)
            closest_virus = self.viruses[closest_idx]
            new_virus = self.viruses[index] + np.random.rand(self.dim) * (closest_virus - self.viruses[index])
            return np.clip(new_virus, self.bounds[:, 0], self.bounds[:, 1])
        return self.viruses[index]

    def cross_activation(self, index):
        """Simulate cross-activation by moving towards the best solution."""
        if self.best_solution is not None:
            new_virus = self.viruses[index] + np.random.rand(self.dim) * (self.best_solution - self.viruses[index])
            return np.clip(new_virus, self.bounds[:, 0], self.bounds[:, 1])
        return self.viruses[index]

    def incremental_reactivation(self, index, iteration):
        """Simulate incremental reactivation with rapid movement."""
        t = iteration / self.max_iter
        evolutionary_operator = np.cos(np.pi * t)  # Dynamic operator based on iteration
        if self.best_solution is not None:
            new_virus = self.viruses[index] + evolutionary_operator * np.random.rand(self.dim) * \
                        (self.best_solution - self.viruses[index])
            return np.clip(new_virus, self.bounds[:, 0], self.bounds[:, 1])
        return self.viruses[index]

    def non_genetic_mutation(self, index):
        """Simulate non-genetic mutation by random perturbation."""
        if np.random.rand() < self.mutation_rate:
            new_virus = self.viruses[index] + np.random.uniform(-0.1, 0.1, self.dim) * \
                        (self.bounds[:, 1] - self.bounds[:, 0])
            return np.clip(new_virus, self.bounds[:, 0], self.bounds[:, 1])
        return self.viruses[index]

    def genotypic_mixing(self, index):
        """Simulate genotypic mixing by combining information from two random viruses."""
        idx1, idx2 = np.random.choice(self.population_size, size=2, replace=False)
        new_virus = self.viruses[index] + np.random.rand(self.dim) * \
                    (self.viruses[idx1] - self.viruses[idx2])
        return np.clip(new_virus, self.bounds[:, 0], self.bounds[:, 1])

    def optimize(self):
        """Run the Coronavirus Metamorphosis Optimization Algorithm."""
        self.initialize_viruses()
        for iteration in range(self.max_iter):
            fitness = self.evaluate_viruses()
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.viruses[min_idx].copy()
                self.best_value = fitness[min_idx]

            # Apply CMOA phases
            new_viruses = np.zeros_like(self.viruses)
            for i in range(self.population_size):
                # Genetic recombination
                new_viruses[i] = self.genetic_recombination(i)
                # Cross-activation
                new_viruses[i] = self.cross_activation(i)
                # Incremental reactivation
                new_viruses[i] = self.incremental_reactivation(i, iteration)
                # Non-genetic mutation
                new_viruses[i] = self.non_genetic_mutation(i)
                # Genotypic mixing
                new_viruses[i] = self.genotypic_mixing(i)

            self.viruses = new_viruses

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
