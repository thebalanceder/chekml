import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt, exp

# Ensure NumPy C API is initialized
cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double cost_function(double[:] x):
    """ Modified sphere function: sum((x_i - 1)^2), global min = 0 at x = [1,1,...,1] """
    cdef int i
    cdef double total = 0.0
    for i in range(x.shape[0]):
        total += (x[i] - 1.0) ** 2
    return total

cdef class FireflyAlgorithm:
    cdef object objective_function
    cdef int dim
    cdef cnp.ndarray bounds
    cdef int population_size
    cdef int max_iter
    cdef double alpha
    cdef double beta0
    cdef double gamma
    cdef double theta
    cdef cnp.ndarray fireflies
    cdef cnp.ndarray light_intensity
    cdef cnp.ndarray best_solution
    cdef double best_value
    cdef list history

    def __init__(self, objective_function, int dim, bounds, int population_size=20, int max_iter=500,
                 double alpha=1.0, double beta0=1.0, double gamma=0.01, double theta=0.97):
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
        self.objective_function = objective_function if objective_function is not None else cost_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.double)
        self.population_size = population_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.theta = theta
        self.best_value = float("inf")
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def initialize_fireflies(self):
        """ Generate initial firefly positions randomly """
        cdef cnp.ndarray[cnp.double_t, ndim=2] fireflies = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], (self.population_size, self.dim))
        cdef cnp.ndarray[cnp.double_t, ndim=1] light_intensity = np.empty(self.population_size, dtype=np.double)
        cdef int i
        for i in range(self.population_size):
            light_intensity[i] = self.objective_function(fireflies[i])
        self.fireflies = fireflies
        self.light_intensity = light_intensity

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def find_limits(self, cnp.ndarray[cnp.double_t, ndim=2] solutions):
        """ Ensure solutions are within bounds """
        return np.clip(solutions, self.bounds[:, 0], self.bounds[:, 1])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def update_fireflies(self, int iteration):
        """ Update firefly positions based on attractiveness and randomness """
        cdef double alpha = self.alpha * self.theta  # Reduce randomness
        self.alpha = alpha
        cdef cnp.ndarray[cnp.double_t, ndim=1] scale = np.abs(self.bounds[:, 1] - self.bounds[:, 0])
        cdef cnp.ndarray[cnp.double_t, ndim=2] fireflies = self.fireflies
        cdef cnp.ndarray[cnp.double_t, ndim=1] light_intensity = self.light_intensity
        cdef int i, j, k
        cdef double r, beta
        cdef cnp.ndarray[cnp.double_t, ndim=1] steps = np.empty(self.dim, dtype=np.double)
        cdef double dist

        for i in range(self.population_size):
            light_intensity[i] = self.objective_function(fireflies[i])  # Update intensity
            for j in range(self.population_size):
                if light_intensity[i] >= light_intensity[j]:  # Move if j is brighter
                    dist = 0.0
                    for k in range(self.dim):
                        dist += (fireflies[i, k] - fireflies[j, k]) ** 2
                    r = sqrt(dist)
                    beta = self.beta0 * exp(-self.gamma * r * r)
                    for k in range(self.dim):
                        steps[k] = alpha * (np.random.rand() - 0.5) * scale[k]
                        fireflies[i, k] += beta * (fireflies[j, k] - fireflies[i, k]) + steps[k]

        self.fireflies = self.find_limits(fireflies)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def rank_fireflies(self):
        """ Sort fireflies by light intensity and update positions """
        cdef cnp.ndarray[cnp.intp_t, ndim=1] indices = np.argsort(self.light_intensity)
        self.light_intensity = self.light_intensity[indices]
        self.fireflies = self.fireflies[indices]

    def optimize(self):
        """ Run the Firefly Algorithm """
        self.initialize_fireflies()
        cdef int iteration
        for iteration in range(self.max_iter):
            self.update_fireflies(iteration)
            self.rank_fireflies()
            if self.light_intensity[0] < self.best_value:
                self.best_solution = self.fireflies[0].copy()
                self.best_value = self.light_intensity[0]
            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

