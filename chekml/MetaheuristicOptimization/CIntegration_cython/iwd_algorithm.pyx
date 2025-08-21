# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport sqrt, pow

# Define types for NumPy arrays
ctypedef cnp.float64_t DTYPE_t

# Class definition
cdef class IntelligentWaterDropOptimizer:
    cdef object objective_function
    cdef int dim
    cdef cnp.ndarray bounds
    cdef int population_size
    cdef int max_iter
    cdef double a_s, b_s, c_s
    cdef double a_v, b_v, c_v
    cdef double init_vel
    cdef double p_n, p_iwd
    cdef double initial_soil
    cdef cnp.ndarray water_drops
    cdef cnp.ndarray best_solution
    cdef double best_value
    cdef list history
    cdef cnp.ndarray soil
    cdef cnp.ndarray HUD

    def __init__(self, objective_function, int dim, bounds, int population_size=50, int max_iter=1000,
                 double a_s=1.0, double b_s=0.01, double c_s=1.0,
                 double a_v=1.0, double b_v=0.01, double c_v=1.0,
                 double init_vel=200, double p_n=0.9, double p_iwd=0.9, double initial_soil=10000):
        """
        Initialize the IWD optimizer.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.float64)
        self.population_size = population_size
        self.max_iter = max_iter
        self.a_s = a_s
        self.b_s = b_s
        self.c_s = c_s
        self.a_v = a_v
        self.b_v = b_v
        self.c_v = c_v
        self.init_vel = init_vel
        self.p_n = p_n
        self.p_iwd = p_iwd
        self.initial_soil = initial_soil
        self.best_value = float("inf")
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def initialize_water_drops(self):
        """Generate initial water drop positions randomly."""
        cdef cnp.ndarray[DTYPE_t, ndim=2] water_drops = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], (self.population_size, self.dim)
        )
        cdef int i
        for i in range(self.population_size):
            water_drops[i] = np.clip(water_drops[i], self.bounds[:, 0], self.bounds[:, 1])
        self.water_drops = water_drops

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def initialize_soil_and_hud(self):
        """Initialize soil and heuristic undesirability matrices."""
        self.soil = np.full((self.population_size, self.population_size), self.initial_soil, dtype=np.float64)
        self.HUD = np.zeros((self.population_size, self.population_size), dtype=np.float64)
        cdef int i, j, k
        cdef double dist
        for i in range(self.population_size):
            for j in range(self.population_size):
                if i != j:
                    dist = 0.0
                    for k in range(self.dim):
                        dist += (self.water_drops[i, k] - self.water_drops[j, k]) ** 2
                    self.HUD[i, j] = sqrt(dist)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def evaluate_water_drops(self):
        """Compute fitness values for the water drops."""
        cdef cnp.ndarray[DTYPE_t, ndim=1] fitness = np.empty(self.population_size, dtype=np.float64)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.water_drops[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double f_soil(self, int i, int j, set visited):
        """Compute f(soil) for probability calculation."""
        cdef double epsilon_s = 0.0001
        return 1.0 / (epsilon_s + self.g_soil(i, j, visited))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double g_soil(self, int i, int j, set visited):
        """Compute g(soil) for probability calculation."""
        cdef double minimum = float("inf")
        cdef int l
        for l in range(self.population_size):
            if l not in visited:
                if self.soil[i, l] < minimum:
                    minimum = self.soil[i, l]
        if minimum >= 0:
            return self.soil[i, j]
        return self.soil[i, j] - minimum

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double probability_of_choosing_j(self, int i, int j, set visited):
        """Compute probability of choosing water drop j from i."""
        cdef double sum_fsoil = 0.0
        cdef int k
        for k in range(self.population_size):
            if k not in visited:
                sum_fsoil += self.f_soil(i, k, visited)
        if sum_fsoil == 0:
            return 0.0
        return self.f_soil(i, j, visited) / sum_fsoil

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double time(self, int i, int j, double vel):
        """Compute time to travel from i to j."""
        return self.HUD[i, j] / vel

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double update_velocity(self, dict iwd, int j):
        """Update velocity of water drop."""
        return iwd['velocity'] + self.a_v / (self.b_v + self.c_v * self.soil[iwd['current'], j] ** 2)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_soil(self, dict iwd, int j, double updated_velocity):
        """Update soil between current node and j."""
        cdef double delta_soil = self.a_s / (self.b_s + self.c_s * self.time(iwd['current'], j, updated_velocity) ** 2)
        self.soil[iwd['current'], j] = (1 - self.p_n) * self.soil[iwd['current'], j] - self.p_n * delta_soil
        iwd['amount_of_soil'] += delta_soil

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef bint move_water_drop(self, dict iwd):
        """Move water drop to next position."""
        cdef set visited = iwd['visited']
        cdef int current = iwd['current']
        cdef dict probabilities = {}
        cdef bint node_selected = False
        cdef int next_node = current
        cdef int j
        cdef double random_number, probability_sum

        # Calculate probabilities for unvisited nodes
        for j in range(self.population_size):
            if j not in visited:
                probabilities[j] = self.probability_of_choosing_j(current, j, visited)

        # Select next node based on probabilities
        random_number = rand() / <double>RAND_MAX
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
            iwd['visited'].add(next_node)

        return node_selected

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double quality(self, cnp.ndarray[DTYPE_t, ndim=1] position):
        """Compute quality as inverse of objective function value."""
        cdef double value = self.objective_function(position)
        return 1.0 / value if value != 0 else float("inf")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """Run the Intelligent Water Drop Optimization."""
        self.initialize_water_drops()
        self.initialize_soil_and_hud()

        cdef int iteration, i, prev, curr
        cdef list iwds
        cdef cnp.ndarray[DTYPE_t, ndim=1] qualities
        cdef int max_quality_idx
        cdef double iteration_quality, current_value
        cdef dict iteration_best

        for iteration in range(self.max_iter):
            # Initialize water drops for this iteration
            iwds = [{
                'id': i,
                'current': i,
                'position': self.water_drops[i].copy(),
                'velocity': self.init_vel,
                'amount_of_soil': 0.0,
                'visited': {i}
            } for i in range(self.population_size)]

            qualities = np.empty(self.population_size, dtype=np.float64)
            for i in range(self.population_size):
                # Move water drop until all nodes are visited or no valid move
                while len(iwds[i]['visited']) < self.population_size:
                    if not self.move_water_drop(iwds[i]):
                        break
                # Complete cycle by returning to start
                start = iwds[i]['id']
                if start not in iwds[i]['visited']:
                    updated_velocity = self.update_velocity(iwds[i], start)
                    iwds[i]['velocity'] = updated_velocity
                    self.update_soil(iwds[i], start, updated_velocity)
                    iwds[i]['current'] = start
                qualities[i] = self.quality(iwds[i]['position'])

            # Find iteration best
            max_quality_idx = np.argmax(qualities)
            iteration_best = iwds[max_quality_idx]
            iteration_quality = qualities[max_quality_idx]

            # Update soil for iteration best path
            visited = list(iteration_best['visited'])
            for i in range(len(visited) - 1):
                prev = visited[i]
                curr = visited[i + 1]
                self.soil[prev, curr] = (1 + self.p_iwd) * self.soil[prev, curr] - \
                                        self.p_iwd * (1 / (self.population_size - 1)) * iteration_best['amount_of_soil']

            # Update global best
            current_value = self.objective_function(iteration_best['position'])
            if current_value < self.best_value:
                self.best_solution = iteration_best['position'].copy()
                self.best_value = current_value

            # Update water drops positions
            self.water_drops = np.array([iwd['position'] for iwd in iwds], dtype=np.float64)
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
