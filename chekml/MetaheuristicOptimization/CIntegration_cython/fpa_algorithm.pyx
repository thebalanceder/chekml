# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython
from scipy.special import gamma
from libc.math cimport sin, cos, pi, pow, abs, sqrt, exp

# Define NumPy array types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class FlowerPollinationAlgorithm:
    cdef public:
        object objective_function
        int dim
        np.ndarray bounds
        int population_size
        int max_iter
        double switch_prob
        np.ndarray flowers
        np.ndarray best_solution
        double best_value
        list history
        np.ndarray fitness

    def __init__(self, object objective_function, int dim, bounds, int population_size=20, 
                 int max_iter=5000, double switch_prob=0.8):
        """
        Initialize the Flower Pollination Algorithm (FPA).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of flowers/solutions.
        - max_iter: Maximum number of iterations.
        - switch_prob: Probability of switching between global and local pollination.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)
        self.population_size = population_size
        self.max_iter = max_iter
        self.switch_prob = switch_prob
        self.flowers = None
        self.best_solution = None
        self.best_value = np.inf
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def initialize_flowers(self):
        """ Generate initial population of flowers randomly within bounds """
        cdef np.ndarray[DTYPE_t, ndim=2] flowers = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], (self.population_size, self.dim))
        self.flowers = flowers
        self.fitness = np.zeros(self.population_size, dtype=DTYPE)
        cdef int i
        for i in range(self.population_size):
            self.fitness[i] = self.objective_function(flowers[i])
        cdef int min_idx = np.argmin(self.fitness)
        self.best_solution = flowers[min_idx].copy()
        self.best_value = self.fitness[min_idx]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[DTYPE_t, ndim=1] levy_flight(self):
        """ Draw samples from a Levy distribution """
        cdef double beta = 1.5  # 3/2 as a float for consistency
        cdef double sigma = (gamma(1 + beta) * sin(pi * beta / 2) /
                             (gamma((1 + beta) / 2) * beta * pow(2, (beta - 1) / 2))) ** (1 / beta)
        cdef np.ndarray[DTYPE_t, ndim=1] u = np.random.randn(self.dim) * sigma
        cdef np.ndarray[DTYPE_t, ndim=1] v = np.random.randn(self.dim)
        cdef np.ndarray[DTYPE_t, ndim=1] step = np.zeros(self.dim, dtype=DTYPE)
        cdef int i
        for i in range(self.dim):
            step[i] = u[i] / pow(abs(v[i]), 1 / beta)
        for i in range(self.dim):
            step[i] *= 0.01
        return step

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[DTYPE_t, ndim=1] simple_bounds(self, np.ndarray[DTYPE_t, ndim=1] solution):
        """ Apply lower and upper bounds to a solution """
        return np.clip(solution, self.bounds[:, 0], self.bounds[:, 1])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[DTYPE_t, ndim=1] global_pollination(self, np.ndarray[DTYPE_t, ndim=1] flower):
        """ Perform global pollination using Levy flight """
        cdef np.ndarray[DTYPE_t, ndim=1] step_size = self.levy_flight()
        cdef np.ndarray[DTYPE_t, ndim=1] delta = step_size * (flower - self.best_solution)
        cdef np.ndarray[DTYPE_t, ndim=1] new_solution = flower + delta
        return self.simple_bounds(new_solution)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[DTYPE_t, ndim=1] local_pollination(self, np.ndarray[DTYPE_t, ndim=1] flower, 
                                                       np.ndarray[np.int64_t, ndim=1] flower_indices):
        """ Perform local pollination using two random flowers """
        cdef double epsilon = np.random.rand()
        cdef np.ndarray[DTYPE_t, ndim=1] flower_j = self.flowers[flower_indices[0]]
        cdef np.ndarray[DTYPE_t, ndim=1] flower_k = self.flowers[flower_indices[1]]
        cdef np.ndarray[DTYPE_t, ndim=1] new_solution = flower + epsilon * (flower_j - flower_k)
        return self.simple_bounds(new_solution)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """ Run the Flower Pollination Algorithm """
        self.initialize_flowers()
        self.best_value = self.fitness[np.argmin(self.fitness)]
        self.best_solution = self.flowers[np.argmin(self.fitness)].copy()
        
        cdef np.ndarray[DTYPE_t, ndim=2] new_flowers = np.zeros((self.population_size, self.dim), dtype=DTYPE)
        cdef int iteration, i
        cdef double new_fitness
        cdef np.ndarray[np.int64_t, ndim=1] indices

        for iteration in range(self.max_iter):
            new_flowers[:] = self.flowers
            for i in range(self.population_size):
                if np.random.rand() > self.switch_prob:
                    # Global pollination
                    new_flowers[i] = self.global_pollination(self.flowers[i])
                else:
                    # Local pollination
                    indices = np.random.permutation(self.population_size)[:2]
                    new_flowers[i] = self.local_pollination(self.flowers[i], indices)

                # Evaluate new solution
                new_fitness = self.objective_function(new_flowers[i])
                if new_fitness <= self.fitness[i]:
                    self.flowers[i] = new_flowers[i]
                    self.fitness[i] = new_fitness

                # Update global best
                if new_fitness <= self.best_value:
                    self.best_solution = new_flowers[i].copy()
                    self.best_value = new_fitness

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            
            # Display progress every 100 iterations
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Best Value = {self.best_value}")

        print(f"Total number of evaluations: {self.max_iter * self.population_size}")
        print(f"Best solution: {self.best_solution}")
        print(f"Best value: {self.best_value}")
        return self.best_solution, self.best_value, self.history

# Example usage with the Ackley function
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double ackley_function(np.ndarray[DTYPE_t, ndim=1] x) except? -1:
    """ Ackley function for testing the optimizer """
    cdef int d = x.shape[0]
    cdef double sum_sq = 0.0
    cdef double sum_cos = 0.0
    cdef int i
    for i in range(d):
        sum_sq += x[i] * x[i]
        sum_cos += cos(2.0 * pi * x[i])
    cdef double term1 = -20.0 * exp(-0.2 * sqrt(sum_sq / d))
    cdef double term2 = -exp(sum_cos / d)
    return term1 + term2 + 20.0 + exp(1.0)

# Wrapper for Python compatibility
def py_ackley_function(np.ndarray[DTYPE_t, ndim=1] x):
    return ackley_function(x)

if __name__ == "__main__":
    dim = 10
    bounds = [(-10, 10)] * dim
    fpa = FlowerPollinationAlgorithm(
        objective_function=py_ackley_function,
        dim=dim,
        bounds=bounds,
        population_size=20,
        max_iter=5000,
        switch_prob=0.8
    )
    best_solution, best_value, history = fpa.optimize()
