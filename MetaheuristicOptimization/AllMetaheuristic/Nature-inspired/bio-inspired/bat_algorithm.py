import numpy as np

class BatAlgorithm:
    def __init__(self, objective_function, dim, bounds, population_size=20, max_iter=1000, 
                 A=1.0, r0=1.0, alpha=0.97, gamma=0.1, Freq_min=0.0, Freq_max=2.0):
        """
        Initialize the Bat Algorithm optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: List of tuples [(lower, upper), ...] for each dimension.
        - population_size: Number of bats (solutions).
        - max_iter: Maximum number of iterations.
        - A: Initial loudness (constant or decreasing).
        - r0: Initial pulse rate (constant or decreasing).
        - alpha: Parameter alpha for loudness decay.
        - gamma: Parameter gamma for pulse rate increase.
        - Freq_min: Minimum frequency.
        - Freq_max: Maximum frequency.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.A = A
        self.r0 = r0
        self.alpha = alpha
        self.gamma = gamma
        self.Freq_min = Freq_min
        self.Freq_max = Freq_max

        self.bats = None  # Population of bats (solutions)
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_bats(self):
        """Generate initial bat population randomly within bounds."""
        self.bats = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                      (self.population_size, self.dim))

    def evaluate_bats(self):
        """Compute fitness values for the bat population."""
        return np.array([self.objective_function(bat) for bat in self.bats])

    def simplebounds(self, s):
        """Apply bounds to a solution."""
        return np.clip(s, self.bounds[:, 0], self.bounds[:, 1])

    def optimize(self):
        """Run the Bat Algorithm optimization."""
        self.initialize_bats()
        A = self.A  # Current loudness
        r = self.r0  # Current pulse rate
        Freq = np.zeros(self.population_size)  # Frequency array
        v = np.zeros((self.population_size, self.dim))  # Velocities

        # Evaluate initial population
        fitness = self.evaluate_bats()
        min_idx = np.argmin(fitness)
        self.best_solution = self.bats[min_idx].copy()
        self.best_value = fitness[min_idx]

        # Main loop
        for t in range(self.max_iter):
            # Update loudness and pulse rate
            r = self.r0 * (1 - np.exp(-self.gamma * t))
            A = self.alpha * A

            # Loop over all bats
            for i in range(self.population_size):
                # Update frequency and velocity
                Freq[i] = self.Freq_min + (self.Freq_max - self.Freq_min) * np.random.rand()
                v[i, :] = v[i, :] + (self.bats[i, :] - self.best_solution) * Freq[i]
                S = self.bats[i, :] + v[i, :]

                # Apply local search with probability r
                if np.random.rand() < r:
                    S = self.best_solution + 0.1 * np.random.randn(self.dim) * A

                # Apply bounds
                S = self.simplebounds(S)

                # Evaluate new solution
                Fnew = self.objective_function(S)

                # Update if solution improves or not too loud
                if (Fnew <= fitness[i]) and (np.random.rand() > A):
                    self.bats[i, :] = S
                    fitness[i] = Fnew

                # Update global best
                if Fnew <= self.best_value:
                    self.best_solution = S.copy()
                    self.best_value = Fnew

            # Store history
            self.history.append((t, self.best_solution.copy(), self.best_value))

            # Display progress every 100 iterations
            if t % 100 == 0:
                print(f"Iteration {t + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

if __name__ == "__main__":
    # Example usage
    def example_function(x):
        return np.sum((x - 2) ** 2)  # Optimal at x = [2, 2, ...]

    bounds = [(-5, 5)] * 2  # 2D example
    ba = BatAlgorithm(objective_function=example_function, dim=2, bounds=bounds)
    best_solution, best_value, history = ba.optimize()
    print(f"Best solution: {best_solution}, Best value: {best_value}")
