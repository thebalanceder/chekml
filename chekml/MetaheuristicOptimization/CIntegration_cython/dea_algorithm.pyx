import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport fabs

# Define numpy types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef class DolphinEcholocationOptimizer:
    cdef object objective_function
    cdef int dim
    cdef np.ndarray bounds
    cdef int population_size
    cdef int max_loops
    cdef double convergence_power
    cdef double effective_radius_factor
    cdef double probability_threshold
    cdef np.ndarray locations
    cdef np.ndarray best_location
    cdef double best_value
    cdef list history
    cdef double effective_radius
    cdef list accumulative_fitness
    cdef list alternatives

    def __init__(self, objective_function, int dim, bounds, int population_size=50, int max_loops=100,
                 double convergence_power=1.0, double effective_radius_factor=0.25, double probability_threshold=0.1):
        """
        Initialize the Dolphin Echolocation Algorithm (DEA) optimizer.

        Parameters:
        - objective_function: Function to optimize (minimize).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of dolphin locations (solutions).
        - max_loops: Maximum number of iterations (loops).
        - convergence_power: Power parameter for the convergence curve.
        - effective_radius_factor: Factor to determine effective radius (< 1/4 search space).
        - probability_threshold: Minimum probability threshold for location selection.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)
        self.population_size = population_size
        self.max_loops = max_loops
        self.convergence_power = convergence_power
        self.effective_radius_factor = effective_radius_factor
        self.probability_threshold = probability_threshold
        self.best_value = np.inf
        self.history = []
        self.accumulative_fitness = []
        self.alternatives = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def initialize_locations(self):
        """ Generate initial dolphin locations randomly """
        cdef np.ndarray[DTYPE_t, ndim=2] locations
        locations = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                     (self.population_size, self.dim)).astype(DTYPE)
        self.locations = locations

        # Calculate effective radius based on search space size
        cdef np.ndarray[DTYPE_t, ndim=1] search_space_size = np.abs(self.bounds[:, 1] - self.bounds[:, 0])
        self.effective_radius = self.effective_radius_factor * np.min(search_space_size)

        # Initialize alternatives matrix (discretized search space per dimension)
        self.alternatives = [np.linspace(self.bounds[i, 0], self.bounds[i, 1], 100, dtype=DTYPE) for i in range(self.dim)]
        self.accumulative_fitness = [np.zeros(100, dtype=DTYPE) for _ in range(self.dim)]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def evaluate_locations(self):
        """ Compute fitness values for the dolphin locations """
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = np.empty(self.population_size, dtype=DTYPE)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.locations[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void calculate_accumulative_fitness(self, np.ndarray[DTYPE_t, ndim=1] fitness):
        """
        Calculate accumulative fitness for alternatives based on dolphin rules.
        """
        # Reset accumulative fitness
        cdef int j
        for j in range(self.dim):
            self.accumulative_fitness[j][:] = 0.0

        cdef int i, k, neighbor_idx, alt_idx
        cdef double loc_value, fit, weight
        cdef np.ndarray[DTYPE_t, ndim=1] alternatives_j
        for i in range(self.population_size):
            for j in range(self.dim):
                alternatives_j = self.alternatives[j]
                loc_value = self.locations[i, j]
                # Find closest alternative index
                alt_idx = 0
                min_diff = fabs(alternatives_j[0] - loc_value)
                for k in range(1, 100):
                    diff = fabs(alternatives_j[k] - loc_value)
                    if diff < min_diff:
                        min_diff = diff
                        alt_idx = k
                fit = fitness[i]
                for k in range(-10, 11):
                    neighbor_idx = alt_idx + k
                    if 0 <= neighbor_idx < 100:
                        weight = (10 - abs(k)) / 10.0
                        self.accumulative_fitness[j][neighbor_idx] += weight * fit

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double get_convergence_probability(self, int loop):
        """ Compute the predefined probability curve value for the current loop """
        cdef double t = loop / <double>self.max_loops
        return t ** self.convergence_power

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def update_probabilities(self, int loop):
        """ Allocate probabilities to alternatives based on accumulative fitness """
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = self.evaluate_locations()
        cdef int min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.best_value:
            self.best_location = self.locations[min_idx].copy()
            self.best_value = fitness[min_idx]

        self.calculate_accumulative_fitness(fitness)
        probabilities = [np.zeros(100, dtype=DTYPE) for _ in range(self.dim)]
        cdef double convergence_prob = self.get_convergence_probability(loop)

        # Find best alternative indices
        cdef int[:] best_alt_indices = np.zeros(self.dim, dtype=np.int32)
        cdef int j, k
        cdef double loc_value
        cdef np.ndarray[DTYPE_t, ndim=1] alternatives_j
        for j in range(self.dim):
            alternatives_j = self.alternatives[j]
            loc_value = self.best_location[j]
            best_alt_indices[j] = np.argmin(np.abs(alternatives_j - loc_value))

        # Assign probabilities
        cdef double total_af, prob, prob_sum, remaining_prob
        for j in range(self.dim):
            total_af = np.sum(self.accumulative_fitness[j])
            if total_af == 0:
                probabilities[j][:] = 1.0 / 100
            else:
                probabilities[j][best_alt_indices[j]] = convergence_prob
                remaining_prob = 1.0 - convergence_prob
                for k in range(100):
                    if k != best_alt_indices[j]:
                        prob = (self.accumulative_fitness[j][k] / total_af) * remaining_prob if total_af > 0 else 0
                        probabilities[j][k] = max(prob, self.probability_threshold)
                prob_sum = np.sum(probabilities[j])
                if prob_sum > 0:
                    probabilities[j] /= prob_sum
        return probabilities

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def select_new_locations(self, list probabilities):
        """ Select new locations based on calculated probabilities """
        cdef np.ndarray[DTYPE_t, ndim=2] new_locations = np.zeros((self.population_size, self.dim), dtype=DTYPE)
        cdef int i, j, selected_idx
        cdef np.ndarray[DTYPE_t, ndim=1] prob_j, alternatives_j
        for i in range(self.population_size):
            for j in range(self.dim):
                prob_j = probabilities[j]
                alternatives_j = self.alternatives[j]
                # Simple random choice based on probabilities
                selected_idx = self._random_choice(prob_j)
                new_locations[i, j] = alternatives_j[selected_idx]
        self.locations = np.clip(new_locations, self.bounds[:, 0], self.bounds[:, 1])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int _random_choice(self, np.ndarray[DTYPE_t, ndim=1] probs):
        """ Custom random choice based on probabilities """
        cdef double r = rand() / <double>RAND_MAX
        cdef double cumsum = 0.0
        cdef int i
        for i in range(probs.shape[0]):
            cumsum += probs[i]
            if r <= cumsum:
                return i
        return probs.shape[0] - 1  # Fallback

    def optimize(self):
        """ Run the Dolphin Echolocation Algorithm """
        self.initialize_locations()
        cdef int loop
        for loop in range(self.max_loops):
            probabilities = self.update_probabilities(loop)
            self.select_new_locations(probabilities)
            self.history.append((loop, self.best_location.copy(), self.best_value))
            print(f"Loop {loop + 1}: Best Value = {self.best_value}")
            if loop > 0 and abs(self.history[loop][2] - self.history[loop-1][2]) < 1e-6:
                print("Convergence reached.")
                break
        return self.best_location, self.best_value, self.history

# Example usage remains Python-compatible
if __name__ == "__main__":
    def objective_function(x):
        return np.sum(x ** 2)
    dim = 2
    bounds = [(-5, 5), (-5, 5)]
    optimizer = DolphinEcholocationOptimizer(
        objective_function=objective_function,
        dim=dim,
        bounds=bounds,
        population_size=50,
        max_loops=100,
        convergence_power=1.0,
        effective_radius_factor=0.25,
        probability_threshold=0.1
    )
    best_solution, best_value, history = optimizer.optimize()
    print(f"Best Solution: {best_solution}")
    print(f"Best Value: {best_value}")
