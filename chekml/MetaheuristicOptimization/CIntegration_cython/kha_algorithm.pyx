# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt, exp, cos, pi
from libc.stdlib cimport rand, RAND_MAX

cnp.import_array()

cdef class KrillHerdOptimizer:
    cdef public:
        object objective_function
        int dim
        cnp.ndarray bounds
        int population_size
        int max_iter
        double Vf
        double Dmax
        double Nmax
        bint crossover_flag
        cnp.ndarray krill_positions
        cnp.ndarray best_position
        double best_value
        list history
        cnp.ndarray N
        cnp.ndarray F
        cnp.ndarray D
        double Dt

    def __init__(self, object objective_function=None, int dim=2, object bounds=[(-5, 5)], 
                 int population_size=25, int max_iter=200, double Vf=0.02, 
                 double Dmax=0.005, double Nmax=0.01, bint crossover_flag=True):
        """
        Initialize the Krill Herd Optimizer.
        """
        self.objective_function = objective_function if objective_function else self.ackley_function
        self.dim = dim
        # Handle bounds
        if isinstance(bounds, tuple):
            bounds = [bounds] * dim
        elif len(bounds) != dim:
            raise ValueError(f"Length of bounds ({len(bounds)}) must match dim ({dim})")
        self.bounds = np.array(bounds, dtype=np.double)
        if self.bounds.shape[0] != dim or self.bounds.shape[1] != 2:
            raise ValueError(f"Bounds must be a list of (low, high) tuples with shape ({dim}, 2)")
        self.population_size = population_size
        self.max_iter = max_iter
        self.Vf = Vf
        self.Dmax = Dmax
        self.Nmax = Nmax
        self.crossover_flag = crossover_flag

        self.best_value = float("inf")
        self.history = []
        self.N = np.zeros((self.dim, self.population_size), dtype=np.double)
        self.F = np.zeros((self.dim, self.population_size), dtype=np.double)
        self.D = np.zeros(self.population_size, dtype=np.double)
        self.Dt = np.mean(np.abs(self.bounds[:, 1] - self.bounds[:, 0])) / 2

    cdef double ackley_function(self, double[:] X):
        """Cythonized Ackley function."""
        cdef int n = self.dim
        cdef double a = 20.0
        cdef double b = 0.2
        cdef double c = 2 * pi
        cdef double s1 = 0.0
        cdef double s2 = 0.0
        cdef int i
        for i in range(n):
            s1 += X[i] * X[i]
            s2 += cos(c * X[i])
        return -a * exp(-b * sqrt(s1 / n)) - exp(s2 / n) + a + exp(1)

    cdef void initialize_krill_positions(self):
        """Generate initial krill positions randomly."""
        cdef double[:, :] bounds_view = self.bounds
        cdef double[:, :] positions = np.empty((self.dim, self.population_size), dtype=np.double)
        cdef int i, j
        for i in range(self.dim):
            for j in range(self.population_size):
                positions[i, j] = bounds_view[i, 0] + (bounds_view[i, 1] - bounds_view[i, 0]) * (<double>rand() / RAND_MAX)
        self.krill_positions = np.asarray(positions)

    cdef double[:] evaluate_krill_positions(self):
        """Compute fitness values for the krill positions."""
        cdef double[:] fitness = np.empty(self.population_size, dtype=np.double)
        cdef int i
        cdef double[:, :] positions = self.krill_positions
        for i in range(self.population_size):
            fitness[i] = self.objective_function(positions[:, i])
        return fitness

    cdef double[:, :] find_limits(self, double[:, :] positions, double[:] best):
        """Evolutionary Boundary Constraint Handling Scheme."""
        cdef double[:, :] ns = positions.copy()
        cdef double lb, ub, A, B
        cdef int i, j
        for i in range(self.dim):
            lb = self.bounds[i, 0]
            ub = self.bounds[i, 1]
            for j in range(positions.shape[1]):
                A = <double>rand() / RAND_MAX
                B = <double>rand() / RAND_MAX
                if ns[i, j] < lb:
                    ns[i, j] = A * lb + (1 - A) * best[i]
                elif ns[i, j] > ub:
                    ns[i, j] = B * ub + (1 - B) * best[i]
        return ns

    cdef double compute_norm(self, double[:] vec):
        """Compute Euclidean norm of a vector."""
        cdef double s = 0.0
        cdef int i
        for i in range(self.dim):
            s += vec[i] * vec[i]
        return sqrt(s)

    cpdef tuple optimize(self):
        """Run the Krill Herd Optimization Algorithm."""
        self.initialize_krill_positions()
        cdef double[:] K = self.evaluate_krill_positions()
        cdef double[:] Kib = K.copy()
        cdef double[:, :] Xib = self.krill_positions.copy()
        cdef int best_idx = np.argmin(K)
        self.best_position = self.krill_positions[:, best_idx].copy()
        self.best_value = K[best_idx]
        cdef list Kgb = [self.best_value]

        cdef int iteration, i, n, nn, j, NK4Cr
        cdef double w, Kw_Kgb, ds, Kf, alpha_b, alpha_n, Beta_f, Beta_b, C_rate
        cdef double[:] Xf = np.empty(self.dim, dtype=np.double)
        cdef double[:] Sf = np.empty(self.dim, dtype=np.double)
        cdef double[:, :] RR, DX
        cdef double[:] Rf = np.empty(self.dim, dtype=np.double)
        cdef double[:] Rgb = np.empty(self.dim, dtype=np.double)
        cdef double[:] Rib = np.empty(self.dim, dtype=np.double)
        cdef double[:] R = np.empty(self.population_size, dtype=np.double)
        cdef double[:] Cr = np.empty(self.dim, dtype=np.double)
        cdef double sum_inv_K

        for iteration in range(self.max_iter):
            # Virtual food location
            sum_inv_K = 0.0
            for i in range(self.population_size):
                sum_inv_K += 1.0 / K[i]
            for i in range(self.dim):
                Sf[i] = 0.0
                for n in range(self.population_size):
                    Sf[i] += self.krill_positions[i, n] / K[n]
            for i in range(self.dim):
                Xf[i] = Sf[i] / sum_inv_K
            # Reshape Xf for find_limits
            Xf_2d = np.array([Xf], dtype=np.double).T
            Xf_2d = self.find_limits(Xf_2d, self.best_position)
            for i in range(self.dim):
                Xf[i] = Xf_2d[i, 0]
            Kf = self.objective_function(Xf)

            if iteration > 0 and Kf < self.history[-1][2]:
                for i in range(self.dim):
                    Xf[i] = self.history[-1][1][i]
                Kf = self.history[-1][2]

            Kw_Kgb = np.max(K) - self.best_value
            w = 0.1 + 0.8 * (1 - iteration / self.max_iter)

            for i in range(self.population_size):
                # Calculate distances
                for n in range(self.dim):
                    Rf[n] = Xf[n] - self.krill_positions[n, i]
                    Rgb[n] = self.best_position[n] - self.krill_positions[n, i]
                RR = self.krill_positions - self.krill_positions[:, i:i+1]
                for n in range(self.population_size):
                    R[n] = 0.0
                    for j in range(self.dim):
                        R[n] += RR[j, n] * RR[j, n]
                    R[n] = sqrt(R[n])

                # Movement Induced
                alpha_b = 0.0
                if self.best_value < K[i]:
                    alpha_b = -2 * (1 + (<double>rand() / RAND_MAX) * (iteration / self.max_iter)) * \
                              (self.best_value - K[i]) / Kw_Kgb / self.compute_norm(Rgb)
                    for n in range(self.dim):
                        self.N[n, i] = w * self.N[n, i] + self.Nmax * alpha_b * Rgb[n]
                else:
                    for n in range(self.dim):
                        self.N[n, i] = w * self.N[n, i]

                alpha_n = 0.0
                nn = 0
                ds = np.mean(R) / 5
                for n in range(self.population_size):
                    if R[n] < ds and n != i:
                        nn += 1
                        if nn <= 4 and K[i] != K[n]:
                            alpha_n -= (K[n] - K[i]) / Kw_Kgb / R[n]
                            for j in range(self.dim):
                                self.N[j, i] += self.Nmax * alpha_n * RR[j, n]
                        if nn >= 4:
                            break

                # Foraging Motion
                Beta_f = 0.0
                if Kf < K[i]:
                    Beta_f = -2 * (1 - iteration / self.max_iter) * (Kf - K[i]) / \
                             Kw_Kgb / self.compute_norm(Rf)
                    for n in range(self.dim):
                        self.F[n, i] = w * self.F[n, i] + self.Vf * Beta_f * Rf[n]
                else:
                    for n in range(self.dim):
                        self.F[n, i] = w * self.F[n, i]

                for n in range(self.dim):
                    Rib[n] = Xib[n, i] - self.krill_positions[n, i]
                Beta_b = 0.0
                if Kib[i] < K[i]:
                    Beta_b = -(Kib[i] - K[i]) / Kw_Kgb / self.compute_norm(Rib)
                    for n in range(self.dim):
                        self.F[n, i] += self.Vf * Beta_b * Rib[n]

                # Physical Diffusion
                self.D[i] = self.Dmax * (1 - iteration / self.max_iter) * \
                            (<double>rand() / RAND_MAX + (K[i] - self.best_value) / Kw_Kgb)
                for n in range(self.dim):
                    self.D[i] *= (2 * (<double>rand() / RAND_MAX) - 1)

                # Motion Process
                DX = np.empty((self.dim, 1), dtype=np.double)
                for n in range(self.dim):
                    DX[n, 0] = self.Dt * (self.N[n, i] + self.F[n, i])

                # Crossover
                if self.crossover_flag:
                    C_rate = 0.8 + 0.2 * (K[i] - self.best_value) / Kw_Kgb
                    for n in range(self.dim):
                        Cr[n] = <double>(<double>rand() / RAND_MAX < C_rate)
                    NK4Cr = int(round((self.population_size - 1) * (<double>rand() / RAND_MAX)))
                    for n in range(self.dim):
                        if Cr[n] == 0.0:
                            self.krill_positions[n, i] = self.krill_positions[n, NK4Cr]

                # Update position
                for n in range(self.dim):
                    self.krill_positions[n, i] += DX[n, 0]
                pos_2d = np.array([self.krill_positions[:, i]], dtype=np.double).T
                pos_2d = self.find_limits(pos_2d, self.best_position)
                for n in range(self.dim):
                    self.krill_positions[n, i] = pos_2d[n, 0]

                # Evaluate new position
                K[i] = self.objective_function(self.krill_positions[:, i])
                if K[i] < Kib[i]:
                    Kib[i] = K[i]
                    for n in range(self.dim):
                        Xib[n, i] = self.krill_positions[n, i]

            # Update global best
            best_idx = np.argmin(K)
            if K[best_idx] < self.best_value:
                for n in range(self.dim):
                    self.best_position[n] = self.krill_positions[n, best_idx]
                self.best_value = K[best_idx]

            Kgb.append(self.best_value)
            self.history.append((iteration, self.best_position.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_position, self.best_value, self.history
