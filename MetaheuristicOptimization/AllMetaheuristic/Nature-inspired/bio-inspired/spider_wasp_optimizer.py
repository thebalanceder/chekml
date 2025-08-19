import numpy as np
from scipy.special import gamma

class SpiderWaspOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=100, max_iter=50000,
                 trade_off=0.3, crossover_prob=0.2, min_population=20):
        """
        Initialize the Spider Wasp Optimizer (SWO).

        Parameters:
        - objective_function: Function to optimize (minimization problem).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of spider wasps (solutions).
        - max_iter: Maximum number of iterations/function evaluations.
        - trade_off: Trade-off probability between hunting and mating behaviors (TR).
        - crossover_prob: Crossover probability for mating behavior (Cr).
        - min_population: Minimum population size for reduction (N_min).
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.trade_off = trade_off
        self.crossover_prob = crossover_prob
        self.min_population = min_population

        self.positions = None  # Population of spider wasps (solutions)
        self.best_solution = None  # Best-so-far solution
        self.best_score = float("inf")  # Best-so-far fitness score
        self.convergence_curve = np.zeros(max_iter)  # Convergence history
        self.fitness = None  # Fitness values of population

    def initialize_positions(self):
        """Initialize the positions of spider wasps randomly within bounds."""
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        if len(lb) == 1:  # Single bound for all dimensions
            self.positions = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        else:  # Different bounds for each dimension
            self.positions = np.zeros((self.population_size, self.dim))
            for i in range(self.dim):
                self.positions[:, i] = np.random.rand(self.population_size) * (ub[i] - lb[i]) + lb[i]

    def evaluate_positions(self):
        """Compute fitness values for the spider wasp positions."""
        return np.array([self.objective_function(pos) for pos in self.positions])

    def levy_flight(self, d):
        """Generate a Levy flight sample."""
        beta = 3/2
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(d) * sigma
        v = np.random.randn(d)
        step = u / np.abs(v) ** (1 / beta)
        return 0.05 * step

    def hunting_behavior(self, i, t, JK):
        """Simulate hunting and nesting behavior."""
        r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
        p = np.random.rand()
        a = 2 - 2 * (t / self.max_iter)  # Linearly decreases from 2 to 0
        a2 = -1 - (t / self.max_iter)  # Linearly decreases from -1 to -2
        k = 1 - (t / self.max_iter)  # Linearly decreases from 1 to 0
        C = a * (2 * r1 - 1)  # Eq. (11)
        l = (a2 - 1) * np.random.rand() + 1  # Parameter for Eqs. (7) and (8)
        L = self.levy_flight(1)  # Levy-based number
        vc = np.random.uniform(-k, k, self.dim)  # Vector in Eq. (12)
        rn1 = np.random.randn()  # Normal distribution-based number

        original_pos = self.positions[i].copy()
        new_pos = self.positions[i].copy()

        for j in range(self.dim):
            if i < k * self.population_size:
                if p < (1 - t / self.max_iter):  # Searching stage (Exploration)
                    if r1 < r2:
                        m1 = np.abs(rn1) * r1  # Eq. (5)
                        new_pos[j] = new_pos[j] + m1 * (self.positions[JK[0], j] - self.positions[JK[1], j])  # Eq. (4)
                    else:
                        B = 1 / (1 + np.exp(l))  # Eq. (8)
                        m2 = B * np.cos(l * 2 * np.pi)  # Eq. (7)
                        new_pos[j] = self.positions[JK[i], j] + m2 * (self.bounds[j, 0] + np.random.rand() * (self.bounds[j, 1] - self.bounds[j, 0]))  # Eq. (6)
                else:  # Following and escaping stage
                    if r1 < r2:
                        new_pos[j] = new_pos[j] + C * np.abs(2 * np.random.rand() * self.positions[JK[2], j] - new_pos[j])  # Eq. (10)
                    else:
                        new_pos[j] = new_pos[j] * vc[j]  # Eq. (12)
            else:
                if r1 < r2:
                    new_pos[j] = self.best_solution[j] + np.cos(2 * l * np.pi) * (self.best_solution[j] - new_pos[j])  # Eq. (16)
                else:
                    new_pos[j] = self.positions[JK[0], j] + r3 * np.abs(L) * (self.positions[JK[0], j] - new_pos[j]) + \
                                 (1 - r3) * (np.random.rand() > np.random.rand()) * (self.positions[JK[2], j] - self.positions[JK[1], j])  # Eq. (17)

        # Bound checking
        new_pos = np.clip(new_pos, self.bounds[:, 0], self.bounds[:, 1])
        new_fitness = self.objective_function(new_pos)

        # Update position and best solution
        if new_fitness < self.fitness[i]:
            self.fitness[i] = new_fitness
            self.positions[i] = new_pos
            if new_fitness < self.best_score:
                self.best_score = new_fitness
                self.best_solution = new_pos.copy()
        else:
            self.positions[i] = original_pos

        return 1  # Increment function evaluation counter

    def mating_behavior(self, i, t, JK):
        """Simulate mating behavior."""
        a2 = -1 - (t / self.max_iter)  # Linearly decreases from -1 to -2
        l = (a2 - 1) * np.random.rand() + 1  # Parameter for Eqs. (7) and (8)
        original_pos = self.positions[i].copy()
        new_pos = self.positions[i].copy()

        # Step sizes for male spider wasp
        if self.fitness[JK[0]] < self.fitness[i]:  # Eq. (23)
            v1 = self.positions[JK[0]] - self.positions[i]
        else:
            v1 = self.positions[i] - self.positions[JK[0]]

        if self.fitness[JK[1]] < self.fitness[JK[2]]:  # Eq. (24)
            v2 = self.positions[JK[1]] - self.positions[JK[2]]
        else:
            v2 = self.positions[JK[2]] - self.positions[JK[1]]

        rn1, rn2 = np.random.randn(), np.random.randn()
        male_pos = np.zeros(self.dim)
        for j in range(self.dim):
            male_pos[j] = new_pos[j] + (np.exp(l)) * np.abs(rn1) * v1[j] + (1 - np.exp(l)) * np.abs(rn2) * v2[j]  # Eq. (22)
            if np.random.rand() < self.crossover_prob:  # Eq. (21)
                new_pos[j] = male_pos[j]

        # Bound checking
        new_pos = np.clip(new_pos, self.bounds[:, 0], self.bounds[:, 1])
        new_fitness = self.objective_function(new_pos)

        # Update position and best solution
        if new_fitness < self.fitness[i]:
            self.fitness[i] = new_fitness
            self.positions[i] = new_pos
            if new_fitness < self.best_score:
                self.best_score = new_fitness
                self.best_solution = new_pos.copy()
        else:
            self.positions[i] = original_pos

        return 1  # Increment function evaluation counter

    def optimize(self):
        """Run the Spider Wasp Optimization algorithm."""
        self.initialize_positions()
        self.fitness = self.evaluate_positions()
        min_idx = np.argmin(self.fitness)
        self.best_score = self.fitness[min_idx]
        self.best_solution = self.positions[min_idx].copy()

        t = 0  # Function evaluation counter
        current_population = self.population_size

        while t < self.max_iter:
            a = 2 - 2 * (t / self.max_iter)  # Used in hunting behavior
            JK = np.random.permutation(current_population)  # Random permutation of indices

            if np.random.rand() < self.trade_off:  # Hunting and nesting behavior
                for i in range(current_population):
                    t += self.hunting_behavior(i, t, JK)
                    if t >= self.max_iter:
                        break
                    self.convergence_curve[t] = self.best_score
            else:  # Mating behavior
                for i in range(current_population):
                    t += self.mating_behavior(i, t, JK)
                    if t >= self.max_iter:
                        break
                    self.convergence_curve[t] = self.best_score

            # Population reduction
            current_population = int(self.min_population + (self.population_size - self.min_population) * ((self.max_iter - t) / self.max_iter))
            if current_population < self.min_population:
                current_population = self.min_population

            # Sort and trim population
            if current_population < len(self.positions):
                sorted_indices = np.argsort(self.fitness)
                self.positions = self.positions[sorted_indices[:current_population]]
                self.fitness = self.fitness[sorted_indices[:current_population]]

        self.convergence_curve[t - 1] = self.best_score
        return self.best_solution, self.best_score, self.convergence_curve
