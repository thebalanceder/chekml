import numpy as np
import matplotlib.pyplot as plt
import time

class WhaleOptimizationAlgorithm:
    def __init__(self, objective_function, dim, bounds, population_size=30, max_iter=100):
        """
        Initialize the Whale Optimization Algorithm (WOA).

        Parameters:
        - objective_function: Function to optimize (minimize).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of search agents (whales).
        - max_iter: Maximum number of iterations.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter

        self.positions = None  # Population of search agents
        self.leader_pos = None  # Best solution found
        self.leader_score = float("inf")  # Best objective value
        self.search_history = []  # Store positions of all agents per iteration

    def initialize_positions(self):
        """Initialize the first population of search agents."""
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        if lb.size == 1:  # Single boundary for all dimensions
            self.positions = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        else:  # Different boundaries for each dimension
            self.positions = np.zeros((self.population_size, self.dim))
            for i in range(self.dim):
                self.positions[:, i] = np.random.rand(self.population_size) * (ub[i] - lb[i]) + lb[i]

    def enforce_bounds(self):
        """Return search agents that go beyond the boundaries to valid space."""
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        for i in range(self.population_size):
            flag4ub = self.positions[i, :] > ub
            flag4lb = self.positions[i, :] < lb
            self.positions[i, :] = (self.positions[i, :] * (~(flag4ub + flag4lb))) + ub * flag4ub + lb * flag4lb

    def optimize(self):
        """Run the Whale Optimization Algorithm."""
        self.initialize_positions()
        self.leader_pos = np.zeros(self.dim)

        for t in range(self.max_iter):
            # Store current positions in history
            self.search_history.append(self.positions.copy())

            # Evaluate and update leader
            for i in range(self.population_size):
                self.enforce_bounds()
                fitness = self.objective_function(self.positions[i, :])
                if fitness < self.leader_score:
                    self.leader_score = fitness
                    self.leader_pos = self.positions[i, :].copy()

            # Update parameters
            a = 2 - t * (2 / self.max_iter)  # Linearly decreases from 2 to 0
            a2 = -1 + t * (-1 / self.max_iter)  # Linearly decreases from -1 to -2

            # Update positions of search agents
            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                A = 2 * a * r1 - a  # Eq. (2.3)
                C = 2 * r2  # Eq. (2.4)
                b = 1  # Parameter in Eq. (2.5)
                l = (a2 - 1) * np.random.rand() + 1  # Parameter in Eq. (2.5)
                p = np.random.rand()  # Random number for strategy selection

                for j in range(self.dim):
                    if p < 0.5:
                        if abs(A) >= 1:  # Search for prey (exploration)
                            rand_leader_index = np.random.randint(self.population_size)
                            X_rand = self.positions[rand_leader_index, :]
                            D_X_rand = abs(C * X_rand[j] - self.positions[i, j])  # Eq. (2.7)
                            self.positions[i, j] = X_rand[j] - A * D_X_rand  # Eq. (2.8)
                        else:  # Encircling prey (exploitation)
                            D_Leader = abs(C * self.leader_pos[j] - self.positions[i, j])  # Eq. (2.1)
                            self.positions[i, j] = self.leader_pos[j] - A * D_Leader  # Eq. (2.2)
                    else:  # Spiral bubble-net attack
                        distance2Leader = abs(self.leader_pos[j] - self.positions[i, j])
                        self.positions[i, j] = distance2Leader * np.exp(b * l) * np.cos(l * 2 * np.pi) + self.leader_pos[j]  # Eq. (2.5)

            print(f"Iteration {t + 1}: Best Score = {self.leader_score}")

        return self.leader_pos, self.leader_score,  self.search_history
