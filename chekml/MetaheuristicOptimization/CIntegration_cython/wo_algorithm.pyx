# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython
from scipy.special import gamma

# Define NumPy types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef class WalrusOptimizer:
    cdef public:
        object objective_function
        int dim
        np.ndarray bounds
        int population_size
        int max_iter
        double female_proportion
        int base
        np.ndarray positions
        np.ndarray best_position
        double best_score
        np.ndarray second_position
        double second_score
        np.ndarray global_best_positions
        np.ndarray convergence_curve
        int female_count
        int male_count
        int child_count

    def __init__(self, objective_function, int dim, bounds, int population_size=100, 
                 int max_iter=2000, double female_proportion=0.4, int base=7):
        """
        Initialize the Walrus Optimizer (WO).

        Parameters:
        - objective_function: Function to optimize (minimize).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of walruses (solutions).
        - max_iter: Maximum number of iterations.
        - female_proportion: Proportion of female walruses in the population.
        - base: Base value for Halton sequence in male position updates.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)
        self.population_size = population_size
        self.max_iter = max_iter
        self.female_proportion = female_proportion
        self.base = base

        self.positions = None
        self.best_position = np.zeros(dim, dtype=DTYPE)
        self.best_score = float("inf")
        self.second_position = np.zeros(dim, dtype=DTYPE)
        self.second_score = float("inf")
        self.global_best_positions = None
        self.convergence_curve = np.zeros(max_iter, dtype=DTYPE)
        
        self.female_count = round(population_size * female_proportion)
        self.male_count = self.female_count
        self.child_count = population_size - self.female_count - self.male_count

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_positions(self):
        """Initialize the positions of walruses using random uniform distribution."""
        cdef np.ndarray[DTYPE_t, ndim=1] lb = self.bounds[:, 0]
        cdef np.ndarray[DTYPE_t, ndim=1] ub = self.bounds[:, 1]
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim)).astype(DTYPE)
        self.global_best_positions = np.tile(self.best_position, (self.population_size, 1)).astype(DTYPE)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double halton_sequence(self, int index, int base):
        """Generate Halton sequence for male position updates."""
        cdef double result = 0.0
        cdef double f = 1.0 / base
        cdef int i = index
        while i > 0:
            result += f * (i % base)
            i = i // base
            f /= base
        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[DTYPE_t, ndim=1] levy_flight(self, int dim):
        """Generate Levy flight step for child position updates."""
        cdef double beta = 3.0 / 2.0
        cdef double sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        cdef np.ndarray[DTYPE_t, ndim=1] u = np.random.normal(0, sigma, dim).astype(DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] v = np.random.normal(0, 1, dim).astype(DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] step = u / np.abs(v) ** (1 / beta)
        return step

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void enforce_bounds(self):
        """Ensure all positions are within bounds."""
        cdef np.ndarray[DTYPE_t, ndim=1] lb = self.bounds[:, 0]
        cdef np.ndarray[DTYPE_t, ndim=1] ub = self.bounds[:, 1]
        self.positions = np.clip(self.positions, lb, ub)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """Run the Walrus Optimizer algorithm."""
        self.initialize_positions()
        cdef int t, i, j
        cdef double fitness, alpha, beta, A, r1, R, danger_signal, safety_signal, r3, r4, P
        cdef np.ndarray[DTYPE_t, ndim=2] positions = self.positions
        cdef np.ndarray[DTYPE_t, ndim=2] global_best_positions = self.global_best_positions
        cdef np.ndarray[DTYPE_t, ndim=1] best_position = self.best_position
        cdef np.ndarray[DTYPE_t, ndim=1] second_position = self.second_position
        cdef double best_score = self.best_score
        cdef double second_score = self.second_score
        cdef np.ndarray[DTYPE_t, ndim=1] convergence_curve = self.convergence_curve
        cdef np.ndarray[DTYPE_t, ndim=1] lb = self.bounds[:, 0]
        cdef np.ndarray[DTYPE_t, ndim=1] ub = self.bounds[:, 1]
        cdef np.ndarray[long, ndim=1] indices
        cdef np.ndarray[DTYPE_t, ndim=1] levy_step
        cdef np.ndarray[DTYPE_t, ndim=1] o

        for t in range(self.max_iter):
            # Evaluate fitness and update best and second-best positions
            for i in range(self.population_size):
                fitness = self.objective_function(positions[i, :])
                if fitness < best_score:
                    best_score = fitness
                    best_position[:] = positions[i, :].copy()
                if best_score < fitness < second_score:
                    second_score = fitness
                    second_position[:] = positions[i, :].copy()

            # Update parameters
            alpha = 1.0 - t / self.max_iter
            beta = 1.0 - 1.0 / (1.0 + np.exp((0.5 * self.max_iter - t) / self.max_iter * 10))
            A = 2.0 * alpha
            r1 = np.random.rand()
            R = 2.0 * r1 - 1.0
            danger_signal = A * R
            safety_signal = np.random.rand()

            if abs(danger_signal) >= 1.0:
                # Migration phase
                r3 = np.random.rand()
                indices = np.random.permutation(self.population_size)
                positions += (beta * r3 ** 2) * (positions[indices, :] - positions)
            else:
                if safety_signal >= 0.5:
                    # Male position updates using Halton sequence
                    for i in range(self.male_count):
                        halton_val = self.halton_sequence(i + 1, self.base)
                        positions[i, :] = lb + halton_val * (ub - lb)
                    # Female position updates
                    for j in range(self.male_count, self.male_count + self.female_count):
                        positions[j, :] += alpha * (positions[j - self.male_count, :] - positions[j, :]) + \
                                           (1.0 - alpha) * (global_best_positions[j, :] - positions[j, :])
                    # Child position updates with Levy flight
                    for i in range(self.population_size - self.child_count, self.population_size):
                        P = np.random.rand()
                        levy_step = self.levy_flight(self.dim)
                        o = global_best_positions[i, :] + positions[i, :] * levy_step
                        positions[i, :] = P * (o - positions[i, :])
                elif safety_signal < 0.5 and abs(danger_signal) >= 0.5:
                    # Position adjustment
                    for i in range(self.population_size):
                        r4 = np.random.rand()
                        positions[i, :] = positions[i, :] * R - \
                                          abs(global_best_positions[i, :] - positions[i, :]) * r4 ** 2
                else:
                    # Exploitation around best and second-best positions
                    for i in range(self.population_size):
                        for j in range(self.dim):
                            theta1 = np.random.rand()
                            a1 = beta * np.random.rand() - beta
                            b1 = np.tan(theta1 * np.pi)
                            X1 = best_position[j] - a1 * b1 * abs(best_position[j] - positions[i, j])

                            theta2 = np.random.rand()
                            a2 = beta * np.random.rand() - beta
                            b2 = np.tan(theta2 * np.pi)
                            X2 = second_position[j] - a2 * b2 * abs(second_position[j] - positions[i, j])

                            positions[i, j] = (X1 + X2) / 2.0

            self.enforce_bounds()
            convergence_curve[t] = best_score
            global_best_positions[:] = np.tile(best_position, (self.population_size, 1))

        self.best_position = best_position
        self.best_score = best_score
        self.second_position = second_position
        self.second_score = second_score
        self.positions = positions
        self.global_best_positions = global_best_positions
        self.convergence_curve = convergence_curve

        return self.best_position, self.best_score, self.convergence_curve.tolist()
