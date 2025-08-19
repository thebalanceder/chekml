# cython: language_level=3
# distutils: language=c
import cython
import numpy as np
cimport numpy as cnp
cimport libc.math as math
from cython cimport floating
import random
import time

# Declare NumPy types
cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double CoefCalculate(double F, double T) nogil:
    """
    Calculate the atmospheric absorption coefficient.

    Parameters:
    - F: Frequency (Hz)
    - T: Temperature (°C)

    Returns:
    - AbsorbCoef: Rounded absorption coefficient (dB)
    """
    cdef double pres = 1.0  # Atmospheric pressure (atm)
    cdef double relh = 50.0  # Relative humidity (%)
    cdef double freq_hum = F
    cdef double temp = T + 273.0  # Convert to Kelvin

    # Calculate humidity
    cdef double C_humid = 4.6151 - 6.8346 * (math.pow(273.15 / temp, 1.261))
    cdef double hum = relh * (math.pow(10.0, C_humid)) * pres

    # Temperature ratio
    cdef double tempr = temp / 293.15

    # Oxygen and nitrogen relaxation frequencies
    cdef double frO = pres * (24.0 + 4.04e4 * hum * (0.02 + hum) / (0.391 + hum))
    cdef double frN = pres * (math.pow(tempr, -0.5)) * (9.0 + 280.0 * hum * math.exp(-4.17 * (math.pow(tempr, -1.0/3.0) - 1.0)))

    # Absorption coefficient calculation
    cdef double alpha = 8.686 * freq_hum * freq_hum * (
        1.84e-11 * (1.0 / pres) * math.sqrt(tempr) +
        (math.pow(tempr, -2.5)) * (
            0.01275 * (math.exp(-2239.1 / temp) * 1.0 / (frO + freq_hum * freq_hum / frO)) +
            0.1068 * (math.exp(-3352.0 / temp) * 1.0 / (frN + freq_hum * freq_hum / frN))
        )
    )

    # Round to 3 decimal places
    cdef double db_humi = math.round(alpha * 1000.0) / 1000.0

    return db_humi

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class CricketAlgorithm:
    cdef public:
        object fun
        int dim
        cnp.ndarray bounds
        int population_size
        double alpha
        int max_iter
        double tol
        double beta_min
        double Q_min
        cnp.ndarray solutions
        cnp.ndarray fitness
        cnp.ndarray best_solution
        double best_fitness
        int N_iter

    def __init__(self, fun, int dim, bounds, int population_size=25, double alpha=0.5, int max_iter=1000, double tol=1e-6):
        """
        Initialize the Cricket Algorithm optimizer.

        Parameters:
        - fun: Objective function to minimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of crickets (solutions).
        - alpha: Scaling factor for randomization.
        - max_iter: Maximum number of iterations.
        - tol: Stopping tolerance for convergence.
        """
        self.fun = fun
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.float64)
        self.population_size = population_size
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.beta_min = 0.2
        self.Q_min = 0.0
        self.N_iter = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void initialize_solutions(self):
        """Initialize the population of solutions (crickets)."""
        self.solutions = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                          (self.population_size, self.dim))
        self.fitness = np.array([self.fun(sol) for sol in self.solutions])
        cdef int min_idx = np.argmin(self.fitness)
        self.best_solution = self.solutions[min_idx].copy()
        self.best_fitness = self.fitness[min_idx]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef cnp.ndarray simple_bounds(self, cnp.ndarray s):
        """Apply boundary constraints to a solution."""
        return np.clip(s, self.bounds[:, 0], self.bounds[:, 1])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double alpha_new(self, double alpha):
        """Update alpha parameter."""
        cdef double delta = 0.97
        return delta * alpha

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple optimize(self):
        """Run the Cricket Algorithm optimization."""
        cdef double start_time = time.time()
        self.initialize_solutions()
        cdef cnp.ndarray Q = np.zeros(self.population_size, dtype=np.float64)
        cdef cnp.ndarray v = np.zeros((self.population_size, self.dim), dtype=np.float64)
        cdef cnp.ndarray scale = self.bounds[:, 1] - self.bounds[:, 0]
        cdef list history = []
        cdef int i, j, k
        cdef cnp.ndarray N, T, C, V, Z, F, S, M, new_solution
        cdef double SumF, SumT, gamma, distance, PS, Lp, Aatm, RLP, K, beta, Fnew
        cdef cnp.ndarray tmpf

        while self.best_fitness > self.tol and self.N_iter < self.max_iter * self.population_size:
            for i in range(self.population_size):
                # Simulate cricket parameters
                N = np.random.randint(0, 121, self.dim).astype(np.float64)
                T = 0.891797 * N + 40.0252
                T = np.clip(T, 55.0, 180.0)
                C = (5.0 / 9.0) * (T - 32.0)
                V = 20.1 * np.sqrt(273.0 + C)
                V = np.sqrt(V) / 1000.0
                Z = self.solutions[i] - self.best_solution

                # Calculate frequency
                F = np.zeros(self.dim, dtype=np.float64)
                for j in range(self.dim):
                    if Z[j] != 0:
                        F[j] = V[j] / Z[j]

                # Compute Q[i] as a scalar (use mean of F to reduce to scalar)
                Q[i] = self.Q_min + np.mean(F - self.Q_min) * np.random.rand()
                v[i] = v[i] + (self.solutions[i] - self.best_solution) * Q[i] + V
                S = self.solutions[i] + v[i]

                # Calculate gamma using CoefCalculate
                SumF = np.mean(F) + 10000.0
                SumT = np.mean(C)
                gamma = CoefCalculate(SumF, SumT)

                # Update solution based on fitness comparison
                M = np.zeros(self.dim, dtype=np.float64)
                for j in range(self.population_size):
                    if self.fitness[i] < self.fitness[j]:
                        distance = math.sqrt(np.sum((self.solutions[i] - self.solutions[j]) ** 2))
                        PS = self.fitness[i] * (4.0 * math.pi * (distance ** 2))
                        Lp = PS + 10.0 * math.log10(1.0 / (4.0 * math.pi * (distance ** 2)))
                        Aatm = (7.4 * (np.mean(F) ** 2 * distance) / (50.0 * (10.0 ** (-8.0))))
                        RLP = Lp - Aatm
                        K = RLP * math.exp(-gamma * distance ** 2)
                        beta = K + self.beta_min
                        tmpf = self.alpha * (np.random.rand(self.dim) - 0.5) * scale
                        M = self.solutions[i] * (1.0 - beta) + self.solutions[j] * beta + tmpf
                    else:
                        M = self.best_solution + 0.01 * np.random.randn(self.dim)

                # Select new solution
                new_solution = S if np.random.rand() > gamma else M
                new_solution = self.simple_bounds(new_solution)

                # Evaluate new solution
                Fnew = self.fun(new_solution)
                self.N_iter += 1

                if Fnew <= self.fitness[i]:
                    self.solutions[i] = new_solution
                    self.fitness[i] = Fnew

                    if Fnew <= self.best_fitness:
                        self.best_solution = new_solution
                        self.best_fitness = Fnew

                self.alpha = self.alpha_new(self.alpha)

            # Store history for this iteration
            history.append((self.N_iter // self.population_size, self.best_solution.copy(), self.best_fitness))
            print(f"Iteration {self.N_iter // self.population_size + 1}: Best Fitness = {self.best_fitness}")

        cdef double elapsed_time = time.time() - start_time
        print(f"Number of evaluations: {self.N_iter}")
        print(f"Best solution: {self.best_solution}")
        print(f"Best fitness: {self.best_fitness}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        return self.best_solution, self.best_fitness, history
