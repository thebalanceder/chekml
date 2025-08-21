# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython
from cython cimport boundscheck, wraparound

# Define numpy types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef class BatAlgorithm:
    cdef object objective_function
    cdef int dim
    cdef np.ndarray bounds
    cdef int population_size
    cdef int max_iter
    cdef double A
    cdef double r0
    cdef double alpha
    cdef double gamma
    cdef double Freq_min
    cdef double Freq_max
    cdef np.ndarray bats
    cdef np.ndarray best_solution
    cdef double best_value
    cdef list history

    def __init__(self, objective_function, int dim, bounds, int population_size=20, 
                 int max_iter=1000, double A=1.0, double r0=1.0, double alpha=0.97, 
                 double gamma=0.1, double Freq_min=0.0, double Freq_max=2.0):
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
        self.bounds = np.array(bounds, dtype=DTYPE)
        self.population_size = population_size
        self.max_iter = max_iter
        self.A = A
        self.r0 = r0
        self.alpha = alpha
        self.gamma = gamma
        self.Freq_min = Freq_min
        self.Freq_max = Freq_max
        self.bats = None
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    @boundscheck(False)
    @wraparound(False)
    cdef void initialize_bats(self):
        """Generate initial bat population randomly within bounds."""
        self.bats = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                      (self.population_size, self.dim))

    @boundscheck(False)
    @wraparound(False)
    def evaluate_bats(self):
        """Compute fitness values for the bat population."""
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = np.empty(self.population_size, dtype=DTYPE)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.bats[i])
        return fitness

    @boundscheck(False)
    @wraparound(False)
    cdef np.ndarray[DTYPE_t, ndim=1] simplebounds(self, np.ndarray[DTYPE_t, ndim=1] s):
        """Apply bounds to a solution."""
        return np.clip(s, self.bounds[:, 0], self.bounds[:, 1])

    @boundscheck(False)
    @wraparound(False)
    def optimize(self):
        """Run the Bat Algorithm optimization."""
        self.initialize_bats()
        cdef double A = self.A
        cdef double r = self.r0
        cdef np.ndarray[DTYPE_t, ndim=1] Freq = np.zeros(self.population_size, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2] v = np.zeros((self.population_size, self.dim), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] S
        cdef np.ndarray[DTYPE_t, ndim=1] fitness
        cdef int t, i, min_idx
        cdef double Fnew, rand_val

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
                rand_val = np.random.rand()
                if rand_val < r:
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
