import numpy as np

class HunterPreyOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=30, max_iter=100, constriction_coeff=0.1):
        """
        Initialize the Hunter-Prey Optimizer (HPO).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of hunters/preys (solutions).
        - max_iter: Maximum number of iterations.
        - constriction_coeff: Constriction coefficient (B parameter).
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.constriction_coeff = constriction_coeff

        self.hp_positions = None  # Population of hunter/prey positions
        self.target_position = None  # Best solution (target)
        self.target_score = float("inf")  # Best fitness value
        self.convergence_curve = np.zeros(max_iter)  # Track convergence
        self.history = []  # Track best solution path

    def initialize_positions(self):
        """ Generate initial hunter/prey positions randomly """
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        self.hp_positions = np.random.uniform(lb, ub, (self.population_size, self.dim))

    def evaluate_positions(self):
        """ Compute fitness values for the hunter/prey positions """
        return np.array([self.objective_function(pos) for pos in self.hp_positions])

    def update_position(self, i, c, kbest):
        """ Update position of a hunter/prey based on HPO rules """
        r1 = np.random.rand(self.dim) < c
        r2 = np.random.rand()
        r3 = np.random.rand(self.dim)
        idx = (r1 == 0)
        z = r2 * idx + r3 * (~idx)

        if np.random.rand() < self.constriction_coeff:
            # Safe mode: Move towards mean and selected individual
            xi = np.mean(self.hp_positions, axis=0)
            dist = np.sqrt(np.sum((xi - self.hp_positions) ** 2, axis=1))
            idxsortdist = np.argsort(dist)
            SI = self.hp_positions[idxsortdist[kbest - 1]]  # kbest-th closest
            self.hp_positions[i] = (self.hp_positions[i] + 0.5 * (
                (2 * c * z * SI - self.hp_positions[i]) +
                (2 * (1 - c) * z * xi - self.hp_positions[i])))
        else:
            # Attack mode: Move towards target with cosine perturbation
            new_pos = np.zeros(self.dim)
            for j in range(self.dim):
                rr = -1 + 2 * z[j]
                new_pos[j] = 2 * z[j] * np.cos(2 * np.pi * rr) * (self.target_position[j] - self.hp_positions[i][j]) + self.target_position[j]
            self.hp_positions[i] = new_pos

        # Ensure bounds
        self.hp_positions[i] = np.clip(self.hp_positions[i], self.bounds[:, 0], self.bounds[:, 1])

    def optimize(self):
        """ Run the Hunter-Prey Optimization algorithm """
        # Initialization
        self.initialize_positions()
        hp_fitness = self.evaluate_positions()
        min_idx = np.argmin(hp_fitness)
        self.target_position = self.hp_positions[min_idx].copy()
        self.target_score = hp_fitness[min_idx]
        self.convergence_curve[0] = self.target_score
        self.history.append((0, self.target_position.copy(), self.target_score))

        # Main loop
        for it in range(1, self.max_iter):
            c = 1 - it * (0.98 / self.max_iter)  # Update C parameter
            kbest = round(self.population_size * c)  # Update kbest

            for i in range(self.population_size):
                self.update_position(i, c, kbest)
                fitness = self.objective_function(self.hp_positions[i])
                # Update target if better solution found
                if fitness < self.target_score:
                    self.target_position = self.hp_positions[i].copy()
                    self.target_score = fitness

            self.convergence_curve[it] = self.target_score
            self.history.append((it, self.target_position.copy(), self.target_score))
            print(f"Iteration: {it + 1}, Best Cost = {self.target_score}")

        return self.target_position, self.target_score, self.convergence_curve, self.history
