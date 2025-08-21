# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, fmax, fmin, fabs

# Define numpy types for Cython
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef class GlowwormSwarmOptimizer:
    cdef public:
        object objective_function
        int dim
        np.ndarray bounds
        int population_size
        int max_iter
        double luciferin_initial
        double decision_range_initial
        double luciferin_decay
        double luciferin_enhancement
        double neighbor_threshold
        double step_size
        double sensor_range
        int neighbor_count
        np.ndarray glowworms
        np.ndarray luciferin
        np.ndarray decision_range
        np.ndarray best_solution
        double best_value
        list history

    def __init__(self, objective_function, int dim, bounds, int population_size=100, int max_iter=100,
                 double luciferin_initial=5.0, double decision_range_initial=3.0, double luciferin_decay=0.4,
                 double luciferin_enhancement=0.6, double neighbor_threshold=0.08, double step_size=0.6,
                 double sensor_range=10.0, int neighbor_count=10):
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
        self.bounds = np.array(bounds, dtype=DTYPE)
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
        self.glowworms = None
        self.luciferin = None
        self.decision_range = None
        self.best_solution = None
        self.best_value = np.inf
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double convert_to_min(self, np.ndarray[DTYPE_t, ndim=1] x):
        """Convert objective function value to minimization form."""
        cdef double fcn = self.objective_function(x)
        if fcn >= 0:
            return 1.0 / (1.0 + fcn)
        return 1.0 + fabs(fcn)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_glowworms(self):
        """Generate initial glowworm positions and luciferin levels."""
        cdef int i
        self.glowworms = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                          (self.population_size, self.dim)).astype(DTYPE)
        self.luciferin = self.luciferin_initial * np.ones(self.population_size, dtype=DTYPE)
        self.decision_range = self.decision_range_initial * np.ones(self.population_size, dtype=DTYPE)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double euclidean_distance(self, np.ndarray[DTYPE_t, ndim=1] pos1, np.ndarray[DTYPE_t, ndim=1] pos2):
        """Compute Euclidean distance between two positions."""
        cdef double dist = 0.0
        cdef int j
        for j in range(self.dim):
            dist += (pos1[j] - pos2[j]) ** 2
        return sqrt(dist)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[DTYPE_t, ndim=1] evaluate_glowworms(self):
        """Compute fitness values for all glowworms."""
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = np.empty(self.population_size, dtype=DTYPE)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.convert_to_min(self.glowworms[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int select_by_roulette(self, np.ndarray[DTYPE_t, ndim=1] probabilities):
        """Select an index using roulette wheel selection."""
        cdef double cum_prob = 0.0
        cdef double rn = np.random.rand()
        cdef int i
        for i in range(probabilities.shape[0]):
            cum_prob += probabilities[i]
            if cum_prob >= rn:
                return i
        return probabilities.shape[0] - 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """Run the Glowworm Swarm Optimization algorithm."""
        self.initialize_glowworms()
        cdef np.ndarray[DTYPE_t, ndim=1] fitness, distances, neighbor_luciferin, probs
        cdef np.ndarray[DTYPE_t, ndim=2] neighbor_positions
        cdef np.ndarray[np.npy_intp, ndim=1] neighbors
        cdef double current_luciferin, distance, neighbor_count
        cdef int iteration, i, j, best_idx, selected_idx
        cdef np.ndarray[DTYPE_t, ndim=1] current_pos, selected_pos

        for iteration in range(self.max_iter):
            # Update luciferin levels
            fitness = self.evaluate_glowworms()
            for i in range(self.population_size):
                self.luciferin[i] = (1.0 - self.luciferin_decay) * self.luciferin[i] + \
                                    self.luciferin_enhancement * fitness[i]

            # Find best glowworm
            best_idx = np.argmax(self.luciferin)
            if fitness[best_idx] < self.best_value:
                self.best_solution = self.glowworms[best_idx].copy()
                self.best_value = fitness[best_idx]

            # Move glowworms
            for i in range(self.population_size):
                current_pos = self.glowworms[i].copy()
                current_luciferin = self.luciferin[i]
                distances = np.empty(self.population_size, dtype=DTYPE)
                for j in range(self.population_size):
                    distances[j] = self.euclidean_distance(current_pos, self.glowworms[j])
                
                # Find neighbors
                neighbors = np.where((distances < self.decision_range[i]) & 
                                     (self.luciferin > current_luciferin))[0]
                neighbor_count = len(neighbors)

                if neighbor_count == 0:
                    self.glowworms[i] = current_pos
                else:
                    neighbor_luciferin = self.luciferin[neighbors]
                    neighbor_positions = self.glowworms[neighbors]
                    probs = (neighbor_luciferin - current_luciferin) / \
                            np.sum(neighbor_luciferin - current_luciferin)
                    selected_idx = self.select_by_roulette(probs)
                    selected_pos = neighbor_positions[selected_idx]
                    distance = self.euclidean_distance(selected_pos, current_pos)
                    if distance > 0:
                        for j in range(self.dim):
                            self.glowworms[i, j] = current_pos[j] + \
                                                   self.step_size * (selected_pos[j] - current_pos[j]) / distance

                # Update decision range
                self.decision_range[i] = fmin(
                    self.sensor_range,
                    fmax(0.0, self.decision_range[i] + 
                         self.neighbor_threshold * (self.neighbor_count - neighbor_count))
                )

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self
