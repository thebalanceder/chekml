import numpy as np

class FireflyAlgorithm:
    def __init__(self, objective_function, dim, bounds, population_size=20, max_iter=100,
                 alpha=1.0, beta0=1.0, gamma=0.01, theta=0.97):
        """
        Initialize the Firefly Algorithm (FA) optimizer.

        Parameters:
        - objective_function: Function to optimize (minimize).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of fireflies (solutions).
        - max_iter: Maximum number of iterations.
        - alpha: Randomness strength (0 to 1).
        - beta0: Attractiveness constant.
        - gamma: Absorption coefficient.
        - theta: Randomness reduction factor.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.theta = theta

        self.fireflies = None  # Population of fireflies (solutions)
        self.light_intensity = None  # Objective values (light intensities)
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_fireflies(self):
        """ Generate initial firefly positions randomly """
        self.fireflies = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                          (self.population_size, self.dim))
        self.light_intensity = np.array([self.objective_function(fly) for fly in self.fireflies])

    def find_limits(self, solutions):
        """ Ensure solutions are within bounds """
        return np.clip(solutions, self.bounds[:, 0], self.bounds[:, 1])

    def update_fireflies(self, iteration):
        """ Update firefly positions based on attractiveness and randomness """
        self.alpha *= self.theta  # Reduce randomness over time
        scale = np.abs(self.bounds[:, 1] - self.bounds[:, 0])  # Problem scale

        for i in range(self.population_size):
            for j in range(self.population_size):
                # Update light intensity for firefly i
                self.light_intensity[i] = self.objective_function(self.fireflies[i])
                # Move firefly i towards j if j is brighter
                if self.light_intensity[i] >= self.light_intensity[j]:
                    r = np.sqrt(np.sum((self.fireflies[i] - self.fireflies[j]) ** 2))
                    beta = self.beta0 * np.exp(-self.gamma * r ** 2)  # Attractiveness
                    steps = self.alpha * (np.random.rand(self.dim) - 0.5) * scale
                    # Update position
                    self.fireflies[i] += beta * (self.fireflies[j] - self.fireflies[i]) + steps

        # Ensure new positions are within bounds
        self.fireflies = self.find_limits(self.fireflies)

    def rank_fireflies(self):
        """ Sort fireflies by light intensity and update positions """
        indices = np.argsort(self.light_intensity)
        self.light_intensity = self.light_intensity[indices]
        self.fireflies = self.fireflies[indices]

    def optimize(self):
        """ Run the Firefly Algorithm """
        self.initialize_fireflies()
        for iteration in range(self.max_iter):
            # Update firefly positions
            self.update_fireflies(iteration)
            # Rank fireflies by light intensity
            self.rank_fireflies()
            # Update best solution
            if self.light_intensity[0] < self.best_value:
                self.best_solution = self.fireflies[0].copy()
                self.best_value = self.light_intensity[0]
            # Store history
            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

