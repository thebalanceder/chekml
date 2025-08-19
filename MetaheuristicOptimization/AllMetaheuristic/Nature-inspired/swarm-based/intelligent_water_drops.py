"""
INTELLIGENT WATER DROPS (IWD) ALGORITHM
ALGORITHM: Shah-Hosseini
ADAPTED TO OPTIMIZATION FRAMEWORK: Grok 3
"""

import numpy as np
import random

class IntelligentWaterDropOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=10,
                 a_s=1.0, b_s=0.01, c_s=1.0, a_v=1.0, b_v=0.01, c_v=1.0,
                 init_vel=20, p_n=0.9, p_iwd=0.9, initial_soil=100):
        """
        Initialize the IWD optimizer.

        Parameters:
        - objective_function: Function to optimize (minimize).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of intelligent water drops.
        - max_iter: Maximum number of iterations.
        - a_s, b_s, c_s: Soil updating parameters.
        - a_v, b_v, c_v: Velocity updating parameters.
        - init_vel: Initial velocity of water drops.
        - p_n: Soil update probability.
        - p_iwd: Soil reinforcement probability.
        - initial_soil: Initial amount of soil on paths.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.a_s, self.b_s, self.c_s = a_s, b_s, c_s
        self.a_v, self.b_v, self.c_v = a_v, b_v, c_v
        self.init_vel = init_vel
        self.p_n = p_n
        self.p_iwd = p_iwd
        self.initial_soil = initial_soil

        self.water_drops = None  # Population of water drops
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []
        self.soil = None  # Soil matrix
        self.HUD = None  # Heuristic undesirability matrix

    def initialize_water_drops(self):
        """ Generate initial water drop positions randomly """
        self.water_drops = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                            (self.population_size, self.dim))
        for i in range(self.population_size):
            self.water_drops[i] = np.clip(self.water_drops[i], self.bounds[:, 0], self.bounds[:, 1])

    def initialize_soil_and_hud(self):
        """ Initialize soil and heuristic undesirability matrices """
        # Initialize soil matrix as a population_size x population_size matrix
        self.soil = np.full((self.population_size, self.population_size), self.initial_soil, dtype=float)
        # Heuristic undesirability (HUD) based on Euclidean distance between solutions
        self.HUD = np.zeros((self.population_size, self.population_size))
        for i in range(self.population_size):
            for j in range(self.population_size):
                if i != j:
                    self.HUD[i, j] = np.sqrt(np.sum((self.water_drops[i] - self.water_drops[j]) ** 2))

    def evaluate_water_drops(self):
        """ Compute fitness values for the water drops """
        return np.array([self.objective_function(drop) for drop in self.water_drops])

    def f_soil(self, i, j, visited):
        """ Compute f(soil) for probability calculation """
        epsilon_s = 0.0001
        return 1.0 / (epsilon_s + self.g_soil(i, j, visited))

    def g_soil(self, i, j, visited):
        """ Compute g(soil) for probability calculation """
        minimum = float("inf")
        for l in range(self.population_size):
            if l not in visited:
                if self.soil[i, l] < minimum:
                    minimum = self.soil[i, l]
        if minimum >= 0:
            return self.soil[i, j]
        return self.soil[i, j] - minimum

    def probability_of_choosing_j(self, i, j, visited):
        """ Compute probability of choosing water drop j from i """
        sum_fsoil = 0.0
        for k in range(self.population_size):
            if k not in visited:
                sum_fsoil += self.f_soil(i, k, visited)
        if sum_fsoil == 0:
            return 0.0
        return self.f_soil(i, j, visited) / sum_fsoil

    def time(self, i, j, vel):
        """ Compute time to travel from i to j """
        return self.HUD[i, j] / vel

    def update_velocity(self, iwd, j):
        """ Update velocity of water drop """
        return iwd['velocity'] + self.a_v / (self.b_v + self.c_v * self.soil[iwd['current'], j] ** 2)

    def update_soil(self, iwd, j, updated_velocity):
        """ Update soil between current node and j """
        delta_soil = self.a_s / (self.b_s + self.c_s * self.time(iwd['current'], j, updated_velocity) ** 2)
        self.soil[iwd['current'], j] = (1 - self.p_n) * self.soil[iwd['current'], j] - self.p_n * delta_soil
        iwd['amount_of_soil'] += delta_soil

    def move_water_drop(self, iwd):
        """ Move water drop to next position """
        visited = iwd['visited']
        current = iwd['current']
        probabilities = {}
        node_selected = False
        next_node = current

        # Calculate probabilities for unvisited nodes
        for j in range(self.population_size):
            if j not in visited:
                probabilities[j] = self.probability_of_choosing_j(current, j, visited)

        # Select next node based on probabilities
        random_number = random.random()
        probability_sum = 0.0
        for j in probabilities:
            probability_sum += probabilities[j]
            if random_number < probability_sum:
                next_node = j
                node_selected = True
                break

        if node_selected:
            # Update velocity and soil
            updated_velocity = self.update_velocity(iwd, next_node)
            iwd['velocity'] = updated_velocity
            self.update_soil(iwd, next_node, updated_velocity)
            # Move water drop (interpolate position)
            iwd['position'] = (self.water_drops[current] + self.water_drops[next_node]) / 2
            iwd['position'] = np.clip(iwd['position'], self.bounds[:, 0], self.bounds[:, 1])
            iwd['current'] = next_node
            iwd['visited'].append(next_node)

        return node_selected

    def quality(self, position):
        """ Compute quality as inverse of objective function value """
        value = self.objective_function(position)
        return 1.0 / value if value != 0 else float("inf")

    def optimize(self):
        """ Run the Intelligent Water Drop Optimization """
        self.initialize_water_drops()
        self.initialize_soil_and_hud()

        for iteration in range(self.max_iter):
            # Initialize water drops for this iteration
            iwds = [{
                'id': i,
                'current': i,
                'position': self.water_drops[i].copy(),
                'velocity': self.init_vel,
                'amount_of_soil': 0.0,
                'visited': [i]
            } for i in range(self.population_size)]

            qualities = []
            for iwd in iwds:
                # Move water drop until all nodes are visited or no valid move
                while len(iwd['visited']) < self.population_size:
                    if not self.move_water_drop(iwd):
                        break
                # Complete cycle by returning to start
                start = iwd['id']
                if start not in iwd['visited']:
                    updated_velocity = self.update_velocity(iwd, start)
                    iwd['velocity'] = updated_velocity
                    self.update_soil(iwd, start, updated_velocity)
                    iwd['current'] = start
                qualities.append(self.quality(iwd['position']))

            # Find iteration best
            max_quality_idx = np.argmax(qualities)
            iteration_best = iwds[max_quality_idx]
            iteration_quality = qualities[max_quality_idx]

            # Update soil for iteration best path
            visited = iteration_best['visited']
            for i in range(len(visited) - 1):
                prev, curr = visited[i], visited[i + 1]
                self.soil[prev, curr] = (1 + self.p_iwd) * self.soil[prev, curr] - \
                                        self.p_iwd * (1 / (self.population_size - 1)) * iteration_best['amount_of_soil']

            # Update global best
            current_value = self.objective_function(iteration_best['position'])
            if current_value < self.best_value:
                self.best_solution = iteration_best['position'].copy()
                self.best_value = current_value

            # Update water drops positions
            self.water_drops = np.array([iwd['position'] for iwd in iwds])
            self.initialize_soil_and_hud()  # Reinitialize HUD for new positions

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

# Example usage
if __name__ == "__main__":
    # Example objective function: Sphere function
    def sphere_function(x):
        return np.sum(x ** 2)

    dim = 5
    bounds = [(-5, 5)] * dim
    iwd_optimizer = IntelligentWaterDropOptimizer(
        objective_function=sphere_function,
        dim=dim,
        bounds=bounds,
        population_size=50,
        max_iter=100
    )
    best_solution, best_value, history = iwd_optimizer.optimize()
    print(f"Best Solution: {best_solution}")
    print(f"Best Value: {best_value}")
