import numpy as np
from scipy.special import gamma  # Import gamma function from scipy.special

class WalrusOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=100, max_iter=100, 
                 female_proportion=0.4, base=7):
        """
        Initialize the Walrus Optimizer (WO).

        Parameters:
        - objective_function: Function to optimize (minimize).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of walruses (solutions).
        - max_iter: Maximum number of iterations.
        - female_proportion: Proportion of female walruses in the population.
        - base: Base value for Halton sequence in male position updates.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.female_proportion = female_proportion
        self.base = base

        self.positions = None  # Population of walrus positions
        self.best_position = np.zeros(dim)
        self.best_score = float("inf")
        self.second_position = np.zeros(dim)
        self.second_score = float("inf")
        self.global_best_positions = None
        self.convergence_curve = np.zeros(max_iter)
        
        # Population division
        self.female_count = round(population_size * female_proportion)
        self.male_count = self.female_count
        self.child_count = population_size - self.female_count - self.male_count

    def initialize_positions(self):
        """Initialize the positions of walruses using random uniform distribution."""
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.global_best_positions = np.tile(self.best_position, (self.population_size, 1))

    def halton_sequence(self, index, base):
        """Generate Halton sequence for male position updates."""
        result = 0
        f = 1 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i = i // base
            f /= base
        return result

    def levy_flight(self, dim):
        """Generate Levy flight step for child position updates."""
        beta = 3 / 2
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, dim)
        v = np.random.normal(0, 1, dim)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def enforce_bounds(self):
        """Ensure all positions are within bounds."""
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        self.positions = np.clip(self.positions, lb, ub)

    def optimize(self):
        """Run the Walrus Optimizer algorithm."""
        self.initialize_positions()

        for t in range(self.max_iter):
            # Evaluate fitness and update best and second-best positions
            for i in range(self.population_size):
                fitness = self.objective_function(self.positions[i, :])
                if fitness < self.best_score:
                    self.best_score = fitness
                    self.best_position = self.positions[i, :].copy()
                if self.best_score < fitness < self.second_score:
                    self.second_score = fitness
                    self.second_position = self.positions[i, :].copy()

            # Update parameters
            alpha = 1 - t / self.max_iter
            beta = 1 - 1 / (1 + np.exp((0.5 * self.max_iter - t) / self.max_iter * 10))
            A = 2 * alpha
            r1 = np.random.rand()
            R = 2 * r1 - 1
            danger_signal = A * R
            safety_signal = np.random.rand()

            if abs(danger_signal) >= 1:
                # Migration phase
                r3 = np.random.rand()
                indices = np.random.permutation(self.population_size)
                migration_step = (beta * r3 ** 2) * (self.positions[indices, :] - self.positions)
                self.positions += migration_step
            else:
                if safety_signal >= 0.5:
                    # Male position updates using Halton sequence
                    for i in range(self.male_count):
                        halton_val = self.halton_sequence(i + 1, self.base)
                        male_pos = self.bounds[:, 0] + halton_val * (self.bounds[:, 1] - self.bounds[:, 0])
                        self.positions[i, :] = male_pos
                    # Female position updates
                    for j in range(self.male_count, self.male_count + self.female_count):
                        self.positions[j, :] += alpha * (self.positions[j - self.male_count, :] - self.positions[j, :]) + \
                                               (1 - alpha) * (self.global_best_positions[j, :] - self.positions[j, :])
                    # Child position updates with Levy flight
                    for i in range(self.population_size - self.child_count, self.population_size):
                        P = np.random.rand()
                        levy_step = self.levy_flight(self.dim)
                        o = self.global_best_positions[i, :] + self.positions[i, :] * levy_step
                        self.positions[i, :] = P * (o - self.positions[i, :])
                elif safety_signal < 0.5 and abs(danger_signal) >= 0.5:
                    # Position adjustment
                    for i in range(self.population_size):
                        r4 = np.random.rand()
                        self.positions[i, :] = self.positions[i, :] * R - \
                                              abs(self.global_best_positions[i, :] - self.positions[i, :]) * r4 ** 2
                else:
                    # Exploitation around best and second-best positions
                    for i in range(self.population_size):
                        for j in range(self.dim):
                            theta1 = np.random.rand()
                            a1 = beta * np.random.rand() - beta
                            b1 = np.tan(theta1 * np.pi)
                            X1 = self.best_position[j] - a1 * b1 * abs(self.best_position[j] - self.positions[i, j])

                            theta2 = np.random.rand()
                            a2 = beta * np.random.rand() - beta
                            b2 = np.tan(theta2 * np.pi)
                            X2 = self.second_position[j] - a2 * b2 * abs(self.second_position[j] - self.positions[i, j])

                            self.positions[i, j] = (X1 + X2) / 2

            self.enforce_bounds()
            self.convergence_curve[t] = self.best_score
            self.global_best_positions = np.tile(self.best_position, (self.population_size, 1))

        return self.best_position, self.best_score, self.convergence_curve.tolist()
