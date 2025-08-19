import numpy as np

class BottlenoseDolphinForagingOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, 
                 exploration_factor=0.5, adjustment_rate=0.3, elimination_ratio=0.2):
        """
        Initialize the Bottlenose Dolphin Foraging Optimizer for continuous optimization.

        Parameters:
        - objective_function: Function to optimize (minimize).
        - dim: Number of dimensions (variables).
        - bounds: List of (lower, upper) bounds for each dimension.
        - population_size: Number of solutions in the population.
        - max_iter: Maximum number of iterations.
        - exploration_factor: Controls the magnitude of random exploration.
        - adjustment_rate: Controls the step size for local adjustments.
        - elimination_ratio: Percentage of worst solutions replaced per iteration.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.exploration_factor = exploration_factor
        self.adjustment_rate = adjustment_rate
        self.elimination_ratio = elimination_ratio

        self.population = None  # Population of solutions
        self.best_solution = None  # Best solution found
        self.best_value = float("inf")  # Best objective value
        self.history = []  # Track (iteration, best_solution, best_value)

    def initialize_population(self):
        """Generate initial population of random solutions."""
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                           (self.population_size, self.dim))

    def evaluate_population(self):
        """Compute objective values for all solutions in the population."""
        return np.array([self.objective_function(solution) for solution in self.population])

    def bottom_grubbing_phase(self, index):
        """
        Perform bottom grubbing-inspired adjustment on a solution.

        Splits the solution into two segments, evaluates their impact, and adjusts toward better regions.
        """
        solution = self.population[index].copy()
        current_fitness = self.objective_function(solution)
        
        # Split solution into two segments (e.g., for 2D: [x1], [x2])
        half_dim = self.dim // 2
        segment1 = solution[:half_dim]
        segment2 = solution[half_dim:]

        # Perturb each segment to estimate its contribution
        perturbation = np.random.uniform(-0.1, 0.1, self.dim)
        perturbed1 = solution.copy()
        perturbed1[:half_dim] += perturbation[:half_dim]
        perturbed1 = np.clip(perturbed1, self.bounds[:, 0], self.bounds[:, 1])
        fitness1 = self.objective_function(perturbed1)

        perturbed2 = solution.copy()
        perturbed2[half_dim:] += perturbation[half_dim:]
        perturbed2 = np.clip(perturbed2, self.bounds[:, 0], self.bounds[:, 1])
        fitness2 = self.objective_function(perturbed2)

        # Move toward the best solution or blend segments
        if self.best_solution is not None:
            direction = self.best_solution - solution
            if fitness1 < current_fitness:  # Segment 1 is promising
                solution[:half_dim] += self.adjustment_rate * direction[:half_dim]
            if fitness2 < current_fitness:  # Segment 2 is promising
                solution[half_dim:] += self.adjustment_rate * direction[half_dim:]
        else:
            # Blend with a random solution if no best solution yet
            random_idx = np.random.randint(self.population_size)
            solution[:half_dim] += self.adjustment_rate * (self.population[random_idx][:half_dim] - segment1)
            solution[half_dim:] += self.adjustment_rate * (self.population[random_idx][half_dim:] - segment2)

        # Clip to bounds
        solution = np.clip(solution, self.bounds[:, 0], self.bounds[:, 1])
        return solution

    def exploration_phase(self, index):
        """Simulate exploration by introducing random perturbations."""
        solution = self.population[index].copy()
        perturbation = self.exploration_factor * np.random.uniform(-1, 1, self.dim) * (self.bounds[:, 1] - self.bounds[:, 0])
        new_solution = solution + perturbation
        new_solution = np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])
        return new_solution

    def elimination_phase(self):
        """Replace worst solutions with new random ones."""
        fitness = self.evaluate_population()
        worst_indices = np.argsort(fitness)[-int(self.elimination_ratio * self.population_size):]
        for i in worst_indices:
            self.population[i] = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)

    def optimize(self):
        """Run the Bottlenose Dolphin Foraging Optimization."""
        self.initialize_population()
        for iteration in range(self.max_iter):
            fitness = self.evaluate_population()
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.population[min_idx].copy()
                self.best_value = fitness[min_idx]

            # Bottom grubbing phase
            for i in range(self.population_size):
                self.population[i] = self.bottom_grubbing_phase(i)

            # Exploration phase
            for i in range(self.population_size):
                if np.random.rand() < 0.3:  # Apply exploration probabilistically
                    self.population[i] = self.exploration_phase(i)

            # Elimination phase
            self.elimination_phase()

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
