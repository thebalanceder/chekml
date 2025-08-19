import numpy as np

class ArtificialFishSwarmOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=30, max_iter=100,
                 visual=0.3, step=0.1, try_number=5, delta=0.618):
        """
        Initialize the Artificial Fish Swarm Algorithm (AFSA).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of artificial fish.
        - max_iter: Maximum number of iterations.
        - visual: Visual distance (perception range).
        - step: Maximum movement step size.
        - try_number: Number of attempts to find a better position.
        - delta: Crowding factor (0 < delta < 1).
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)
        self.population_size = population_size
        self.max_iter = max_iter
        self.visual = visual
        self.step = step
        self.try_number = try_number
        self.delta = delta

        self.fish_positions = None
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_fish(self):
        """ Generate initial fish positions randomly """
        self.fish_positions = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                                (self.population_size, self.dim))

    def evaluate(self, positions):
        """ Evaluate fitness for a set of positions """
        return np.array([self.objective_function(pos) for pos in positions])

    def move_towards(self, Xi, Xj):
        """ Move fish Xi towards Xj with step size """
        direction = Xj - Xi
        norm = np.linalg.norm(direction)
        if norm == 0:
            return Xi
        step_vec = self.step * direction / norm
        new_pos = Xi + step_vec * np.random.rand()
        return np.clip(new_pos, self.bounds[:, 0], self.bounds[:, 1])

    def prey_behavior(self, i):
        """ AFSA prey behavior: random search in visual range """
        Xi = self.fish_positions[i]
        best_candidate = Xi.copy()
        best_value = self.objective_function(Xi)

        for _ in range(self.try_number):
            candidate = Xi + self.visual * (np.random.rand(self.dim) * 2 - 1)
            candidate = np.clip(candidate, self.bounds[:, 0], self.bounds[:, 1])
            candidate_value = self.objective_function(candidate)
            if candidate_value < best_value:
                best_candidate = candidate
                best_value = candidate_value

        if best_value < self.objective_function(Xi):
            return self.move_towards(Xi, best_candidate)
        return Xi

    def swarm_behavior(self, i):
        """ AFSA swarm behavior: move towards center if less crowded """
        Xi = self.fish_positions[i]
        neighbors = []
        for j in range(self.population_size):
            if i != j and np.linalg.norm(self.fish_positions[j] - Xi) < self.visual:
                neighbors.append(self.fish_positions[j])

        if not neighbors:
            return self.prey_behavior(i)

        center = np.mean(neighbors, axis=0)
        center_value = self.objective_function(center)
        avg_density = len(neighbors)

        if center_value / avg_density < self.objective_function(Xi) * self.delta:
            return self.move_towards(Xi, center)
        return self.prey_behavior(i)

    def follow_behavior(self, i):
        """ AFSA follow behavior: follow the best neighbor if advantageous """
        Xi = self.fish_positions[i]
        best_neighbor = Xi.copy()
        best_value = self.objective_function(Xi)

        for j in range(self.population_size):
            if i != j and np.linalg.norm(self.fish_positions[j] - Xi) < self.visual:
                neighbor_value = self.objective_function(self.fish_positions[j])
                if neighbor_value < best_value:
                    best_neighbor = self.fish_positions[j]
                    best_value = neighbor_value

        if best_value < self.objective_function(Xi) * self.delta:
            return self.move_towards(Xi, best_neighbor)
        return self.prey_behavior(i)

    def optimize(self):
        """ Run the Artificial Fish Swarm Optimization """
        self.initialize_fish()
        for generation in range(self.max_iter):
            for i in range(self.population_size):
                behavior = np.random.choice(["prey", "swarm", "follow"])
                if behavior == "prey":
                    self.fish_positions[i] = self.prey_behavior(i)
                elif behavior == "swarm":
                    self.fish_positions[i] = self.swarm_behavior(i)
                else:
                    self.fish_positions[i] = self.follow_behavior(i)

            fitness = self.evaluate(self.fish_positions)
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.fish_positions[min_idx]
                self.best_value = fitness[min_idx]

            self.history.append((generation, self.best_solution.copy(), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history