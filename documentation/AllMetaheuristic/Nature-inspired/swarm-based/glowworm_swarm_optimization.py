import numpy as np

class GlowwormSwarmOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=100, max_iter=100,
                 luciferin_initial=5.0, decision_range_initial=3.0, luciferin_decay=0.4,
                 luciferin_enhancement=0.6, neighbor_threshold=0.08, step_size=0.6,
                 sensor_range=10.0, neighbor_count=10):
        """
        Initialize the Glowworm Swarm Optimization (GSO) algorithm.

        Parameters:
        - objective_function: Function to optimize (minimization).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of glowworms (solutions).
        - max_iter: Maximum number of iterations.
        - luciferin_initial: Initial luciferin value for all glowworms.
        - decision_range_initial: Initial decision range for movement.
        - luciferin_decay: Decay rate of luciferin.
        - luciferin_enhancement: Enhancement factor for luciferin update.
        - neighbor_threshold: Threshold for dynamic decision range update.
        - step_size: Step size for movement towards brighter glowworms.
        - sensor_range: Maximum sensor range for detecting neighbors.
        - neighbor_count: Desired number of neighbors for decision range update.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.luciferin_initial = luciferin_initial
        self.decision_range_initial = decision_range_initial
        self.luciferin_decay = luciferin_decay
        self.luciferin_enhancement = luciferin_enhancement
        self.neighbor_threshold = neighbor_threshold
        self.step_size = step_size
        self.sensor_range = sensor_range
        self.neighbor_count = neighbor_count

        self.glowworms = None  # Population of glowworm positions
        self.luciferin = None  # Luciferin levels of glowworms
        self.decision_range = None  # Decision ranges for each glowworm
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def convert_to_min(self, x):
        """Convert objective function value to minimization form."""
        fcn = self.objective_function(x)
        if fcn >= 0:
            return 1 / (1 + fcn)
        return 1 + abs(fcn)

    def initialize_glowworms(self):
        """Generate initial glowworm positions and luciferin levels."""
        self.glowworms = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                          (self.population_size, self.dim))
        self.luciferin = self.luciferin_initial * np.ones(self.population_size)
        self.decision_range = self.decision_range_initial * np.ones(self.population_size)

    def euclidean_distance(self, pos1, pos2):
        """Compute Euclidean distance between two positions."""
        return np.sqrt(np.sum((pos1 - pos2) ** 2))

    def evaluate_glowworms(self):
        """Compute fitness values for all glowworms."""
        return np.array([self.convert_to_min(glowworm) for glowworm in self.glowworms])

    def select_by_roulette(self, probabilities):
        """Select an index using roulette wheel selection."""
        cum_prob = np.cumsum(probabilities)
        rn = np.random.rand()
        return np.where(cum_prob >= rn)[0][0]

    def optimize(self):
        """Run the Glowworm Swarm Optimization algorithm."""
        self.initialize_glowworms()

        for iteration in range(self.max_iter):
            # Update luciferin levels
            fitness = self.evaluate_glowworms()
            self.luciferin = (1 - self.luciferin_decay) * self.luciferin + self.luciferin_enhancement * fitness

            # Find best glowworm
            best_idx = np.argmax(self.luciferin)
            if fitness[best_idx] < self.best_value:
                self.best_solution = self.glowworms[best_idx].copy()
                self.best_value = fitness[best_idx]

            # Move glowworms
            for i in range(self.population_size):
                current_pos = self.glowworms[i].copy()
                current_luciferin = self.luciferin[i]
                distances = np.array([self.euclidean_distance(current_pos, other_pos)
                                      for other_pos in self.glowworms])
                neighbors = np.where((distances < self.decision_range[i]) & (self.luciferin > current_luciferin))[0]

                if len(neighbors) == 0:
                    self.glowworms[i] = current_pos
                else:
                    neighbor_luciferin = self.luciferin[neighbors]
                    neighbor_positions = self.glowworms[neighbors]
                    probs = (neighbor_luciferin - current_luciferin) / np.sum(neighbor_luciferin - current_luciferin)
                    selected_idx = self.select_by_roulette(probs)
                    selected_pos = neighbor_positions[selected_idx]
                    distance = self.euclidean_distance(selected_pos, current_pos)
                    if distance > 0:
                        self.glowworms[i] = current_pos + self.step_size * (selected_pos - current_pos) / distance

                # Update decision range
                neighbor_count = len(neighbors)
                self.decision_range[i] = min(
                    self.sensor_range,
                    max(0, self.decision_range[i] + self.neighbor_threshold * (self.neighbor_count - neighbor_count))
                )

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

# Example usage
if __name__ == "__main__":
    def sphere_function(x):
        return np.sum(x ** 2)

    dim = 2
    bounds = [(-20, 20)] * dim
    gso = GlowwormSwarmOptimizer(sphere_function, dim, bounds)
    best_solution, best_value, history = gso.optimize()
    print(f"Best Solution: {best_solution}")
    print(f"Best Value: {best_value}")
