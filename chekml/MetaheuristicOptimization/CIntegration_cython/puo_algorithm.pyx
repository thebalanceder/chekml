# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport exp, cos, fabs
from libc.stdlib cimport rand, RAND_MAX

# Ensure NumPy arrays are C-contiguous
cnp.import_array()

cdef class PumaOptimizer:
    cdef public:
        object objective_function
        int dim
        cnp.ndarray bounds
        int population_size
        int max_iter
        double Q
        double beta
        list PF
        double mega_explore
        double mega_exploit
        cnp.ndarray pumas
        cnp.ndarray best_solution
        double best_value
        list history
        cnp.ndarray unselected
        double f3_explore
        double f3_exploit
        cnp.ndarray seq_time_explore
        cnp.ndarray seq_time_exploit
        cnp.ndarray seq_cost_explore
        cnp.ndarray seq_cost_exploit
        double score_explore
        double score_exploit
        list pf_f3
        int flag_change

    def __init__(self, objective_function, int dim, bounds, int population_size=30, int max_iter=500, 
                 double Q=0.67, double beta=2, list PF=[0.5, 0.5, 0.3], double mega_explore=0.99, 
                 double mega_exploit=0.99):
        """
        Initialize the Puma Optimizer (PO).

        Parameters:
        - objective_function: Function to optimize (minimization).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of pumas (solutions).
        - max_iter: Maximum number of iterations.
        - Q: Probability threshold for exploitation phase.
        - beta: Scaling factor for exploitation phase.
        - PF: Weighting factors for F1, F2, and F3 calculations.
        - mega_explore: Exploration adaptation parameter.
        - mega_exploit: Exploitation adaptation parameter.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.float64)
        self.population_size = population_size
        self.max_iter = max_iter
        self.Q = Q
        self.beta = beta
        self.PF = PF
        self.mega_explore = mega_explore
        self.mega_exploit = mega_exploit
        self.pumas = None
        self.best_solution = None
        self.best_value = np.inf
        self.history = []
        self.unselected = np.ones(2, dtype=np.float64)
        self.f3_explore = 0.0
        self.f3_exploit = 0.0
        self.seq_time_explore = np.ones(3, dtype=np.float64)
        self.seq_time_exploit = np.ones(3, dtype=np.float64)
        self.seq_cost_explore = np.ones(3, dtype=np.float64)
        self.seq_cost_exploit = np.ones(3, dtype=np.float64)
        self.score_explore = 0.0
        self.score_exploit = 0.0
        self.pf_f3 = []
        self.flag_change = 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_pumas(self):
        """Generate initial puma population randomly within bounds."""
        cdef cnp.ndarray[double, ndim=2] bounds = self.bounds
        self.pumas = np.random.uniform(bounds[:, 0], bounds[:, 1], 
                                       (self.population_size, self.dim))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[double, ndim=1] evaluate_pumas(self):
        """Compute fitness values for the puma population."""
        cdef cnp.ndarray[double, ndim=2] pumas = self.pumas
        cdef int i
        cdef cnp.ndarray[double, ndim=1] fitness = np.empty(self.population_size, dtype=np.float64)
        for i in range(self.population_size):
            fitness[i] = self.objective_function(pumas[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[double, ndim=2] boundary_check(self, cnp.ndarray[double, ndim=2] X):
        """Ensure solutions stay within bounds."""
        cdef cnp.ndarray[double, ndim=2] bounds = self.bounds
        cdef cnp.ndarray[cnp.uint8_t, ndim=2, cast=True] FU = X > bounds[:, 1]
        cdef cnp.ndarray[cnp.uint8_t, ndim=2, cast=True] FL = X < bounds[:, 0]
        return X * (~(FU | FL)) + bounds[:, 1] * FU + bounds[:, 0] * FL

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[double, ndim=2] exploration_phase(self):
        """Simulate exploration phase (global search)."""
        cdef double pCR = 0.20
        cdef double PCR = 1.0 - pCR
        cdef double p = PCR / self.population_size
        cdef cnp.ndarray[double, ndim=2] pumas = self.pumas
        cdef cnp.ndarray[double, ndim=1] fitness = self.evaluate_pumas()
        cdef cnp.ndarray[cnp.int64_t, ndim=1] sorted_indices = np.argsort(fitness)
        pumas = pumas[sorted_indices]
        cdef int i, j, j0
        cdef cnp.ndarray[cnp.int64_t, ndim=1] A
        cdef int a, b, c, d, e, f
        cdef double G
        cdef cnp.ndarray[double, ndim=1] x, y, z
        cdef double new_cost

        for i in range(self.population_size):
            x = pumas[i].copy()
            A = np.random.permutation(self.population_size)
            A = A[A != i]
            a, b, c, d, e, f = A[:6]
            G = 2.0 * (<double>rand() / RAND_MAX) - 1.0  # Eq 26

            if (<double>rand() / RAND_MAX) < 0.5:
                y = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)  # Eq 25
            else:
                y = pumas[a] + G * (pumas[a] - pumas[b]) + G * (
                    (pumas[a] - pumas[b]) - (pumas[c] - pumas[d]) +
                    (pumas[c] - pumas[d]) - (pumas[e] - pumas[f])
                )  # Eq 25

            y = self.boundary_check(y[np.newaxis, :])[0]
            z = x.copy()
            j0 = rand() % self.dim
            for j in range(self.dim):
                if j == j0 or (<double>rand() / RAND_MAX) <= pCR:
                    z[j] = y[j]
                else:
                    z[j] = x[j]

            new_cost = self.objective_function(z)
            if new_cost < fitness[sorted_indices[i]]:
                pumas[i] = z
            else:
                pCR += p  # Eq 30

        self.pumas = pumas
        return self.pumas

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[double, ndim=2] exploitation_phase(self, int iter_count):
        """Simulate exploitation phase (local search)."""
        cdef cnp.ndarray[double, ndim=2] pumas = self.pumas
        cdef cnp.ndarray[double, ndim=1] best_solution = self.best_solution
        cdef cnp.ndarray[double, ndim=1] fitness = self.evaluate_pumas()
        cdef int i, j, r1
        cdef double beta1, R_1, new_cost
        cdef cnp.ndarray[double, ndim=1] beta2, w, v, F1, F2, S1, S2, VEC, Xatack, mbest
        cdef double sign
        cdef double rand_val

        for i in range(self.population_size):
            beta1 = 2.0 * (<double>rand() / RAND_MAX)
            beta2 = np.random.randn(self.dim)
            w = np.random.randn(self.dim)  # Eq 37
            v = np.random.randn(self.dim)  # Eq 38
            F1 = np.random.randn(self.dim) * exp(2.0 - iter_count * (2.0 / self.max_iter))  # Eq 35
            rand_val = <double>rand() / RAND_MAX
            F2 = w * v**2 * np.cos(2.0 * rand_val * w)  # Eq 36
            mbest = np.mean(pumas, axis=0) / self.population_size
            R_1 = 2.0 * (<double>rand() / RAND_MAX) - 1.0  # Eq 34
            S1 = 2.0 * (<double>rand() / RAND_MAX) - 1.0 + np.random.randn(self.dim)
            S2 = F1 * R_1 * pumas[i] + F2 * (1.0 - R_1) * best_solution
            VEC = S2 / S1

            rand_val = <double>rand() / RAND_MAX
            if rand_val <= 0.5:
                Xatack = VEC
                if (<double>rand() / RAND_MAX) > self.Q:
                    r1 = rand() % self.population_size
                    pumas[i] = best_solution + beta1 * np.exp(beta2) * (pumas[r1] - pumas[i])  # Eq 32
                else:
                    pumas[i] = beta1 * Xatack - best_solution  # Eq 32
            else:
                r1 = 1 + (rand() % (self.population_size - 1))
                sign = -1.0 if rand() % 2 else 1.0
                pumas[i] = (mbest * pumas[r1] - sign * pumas[i]) / (
                    1.0 + (self.beta * (<double>rand() / RAND_MAX))
                )  # Eq 32

            pumas[i] = self.boundary_check(pumas[i][np.newaxis, :])[0]
            new_cost = self.objective_function(pumas[i])
            if new_cost < fitness[i]:
                pumas[i] = pumas[i]  # Already updated

        self.pumas = pumas
        return self.pumas

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """Run the Puma Optimization Algorithm."""
        self.history = []
        self.initialize_pumas()
        cdef cnp.ndarray[double, ndim=1] initial_fitness = self.evaluate_pumas()
        cdef cnp.int64_t min_idx = np.argmin(initial_fitness)
        self.best_solution = self.pumas[min_idx].copy()
        self.best_value = initial_fitness[min_idx]
        cdef cnp.ndarray[double, ndim=1] initial_best = self.best_solution.copy()
        cdef double initial_best_cost = self.best_value
        cdef int iter_count, i, select_flag
        cdef list costs_explore = []
        cdef list costs_exploit = []
        cdef cnp.ndarray[double, ndim=2] pumas_explore, pumas_exploit, combined_pumas
        cdef cnp.ndarray[double, ndim=1] fitness_explore, fitness_exploit, combined_fitness
        cdef cnp.ndarray[cnp.int64_t, ndim=1] sorted_indices
        cdef cnp.ndarray[double, ndim=1] t_best
        cdef double t_best_cost
        cdef cnp.ndarray[double, ndim=1] count_select
        cdef double f1_explore, f1_exploit, f2_explore, f2_exploit
        cdef double lmn_explore, lmn_exploit
        cdef cnp.ndarray[double, ndim=1] fitness

        # Unexperienced Phase (first 3 iterations)
        for iter_count in range(3):
            pumas_explore = self.exploration_phase()
            fitness_explore = self.evaluate_pumas()
            costs_explore.append(np.min(fitness_explore))
            pumas_exploit = self.exploitation_phase(iter_count + 1)
            fitness_exploit = self.evaluate_pumas()
            costs_exploit.append(np.min(fitness_exploit))
            combined_pumas = np.vstack((self.pumas, pumas_explore, pumas_exploit))
            combined_fitness = np.concatenate((self.evaluate_pumas(), fitness_explore, fitness_exploit))
            sorted_indices = np.argsort(combined_fitness)[:self.population_size]
            self.pumas = combined_pumas[sorted_indices]
            self.best_solution = self.pumas[0].copy()
            self.best_value = self.objective_function(self.best_solution)
            self.history.append((iter_count, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iter_count + 1}: Best Value = {self.best_value}")

        # Hyper Initialization
        self.seq_cost_explore[0] = fabs(initial_best_cost - costs_explore[0])  # Eq 5
        self.seq_cost_exploit[0] = fabs(initial_best_cost - costs_exploit[0])  # Eq 8
        self.seq_cost_explore[1] = fabs(costs_explore[1] - costs_explore[0])  # Eq 6
        self.seq_cost_exploit[1] = fabs(costs_exploit[1] - costs_explore[0])  # Eq 9
        self.seq_cost_explore[2] = fabs(costs_explore[2] - costs_explore[1])  # Eq 7
        self.seq_cost_exploit[2] = fabs(costs_exploit[2] - costs_exploit[1])  # Eq 10

        for i in range(3):
            if self.seq_cost_explore[i] != 0:
                self.pf_f3.append(self.seq_cost_explore[i])
            if self.seq_cost_exploit[i] != 0:
                self.pf_f3.append(self.seq_cost_exploit[i])

        # Handle case where pf_f3 might be empty
        if not self.pf_f3:
            self.pf_f3.append(1e-10)

        # Calculate initial scores
        f1_explore = self.PF[0] * (self.seq_cost_explore[0] / self.seq_time_explore[0])  # Eq 1
        f1_exploit = self.PF[0] * (self.seq_cost_exploit[0] / self.seq_time_exploit[0])  # Eq 2
        f2_explore = self.PF[1] * (sum(self.seq_cost_explore) / sum(self.seq_time_explore))  # Eq 3
        f2_exploit = self.PF[1] * (sum(self.seq_cost_exploit) / sum(self.seq_time_exploit))  # Eq 4
        self.score_explore = (self.PF[0] * f1_explore) + (self.PF[1] * f2_explore)  # Eq 11
        self.score_exploit = (self.PF[0] * f1_exploit) + (self.PF[1] * f2_exploit)  # Eq 12

        # Experienced Phase
        for iter_count in range(3, self.max_iter):
            if self.score_explore > self.score_exploit:
                # Exploration
                select_flag = 1
                self.pumas = self.exploration_phase()
                count_select = self.unselected.copy()
                self.unselected[1] += 1
                self.unselected[0] = 1
                self.f3_explore = self.PF[2]
                self.f3_exploit += self.PF[2]
                fitness = self.evaluate_pumas()
                min_idx = np.argmin(fitness)
                t_best = self.pumas[min_idx].copy()
                t_best_cost = fitness[min_idx]
                self.seq_cost_explore[2] = self.seq_cost_explore[1]
                self.seq_cost_explore[1] = self.seq_cost_explore[0]
                self.seq_cost_explore[0] = fabs(self.best_value - t_best_cost)
                if self.seq_cost_explore[0] != 0:
                    self.pf_f3.append(self.seq_cost_explore[0])
                if t_best_cost < self.best_value:
                    self.best_solution = t_best.copy()
                    self.best_value = t_best_cost
            else:
                # Exploitation
                select_flag = 2
                self.pumas = self.exploitation_phase(iter_count + 1)
                count_select = self.unselected.copy()
                self.unselected[0] += 1
                self.unselected[1] = 1
                self.f3_explore += self.PF[2]
                self.f3_exploit = self.PF[2]
                fitness = self.evaluate_pumas()
                min_idx = np.argmin(fitness)
                t_best = self.pumas[min_idx].copy()
                t_best_cost = fitness[min_idx]
                self.seq_cost_exploit[2] = self.seq_cost_exploit[1]
                self.seq_cost_exploit[1] = self.seq_cost_exploit[0]
                self.seq_cost_exploit[0] = fabs(self.best_value - t_best_cost)
                if self.seq_cost_exploit[0] != 0:
                    self.pf_f3.append(self.seq_cost_exploit[0])
                if t_best_cost < self.best_value:
                    self.best_solution = t_best.copy()
                    self.best_value = t_best_cost

            if self.flag_change != select_flag:
                self.flag_change = select_flag
                self.seq_time_explore[2] = self.seq_time_explore[1]
                self.seq_time_explore[1] = self.seq_time_explore[0]
                self.seq_time_explore[0] = count_select[0]
                self.seq_time_exploit[2] = self.seq_time_exploit[1]
                self.seq_time_exploit[1] = self.seq_time_exploit[0]
                self.seq_time_exploit[0] = count_select[1]

            # Update scores
            f1_explore = self.PF[0] * (self.seq_cost_explore[0] / self.seq_time_explore[0])  # Eq 14
            f1_exploit = self.PF[0] * (self.seq_cost_exploit[0] / self.seq_time_exploit[0])  # Eq 13
            f2_explore = self.PF[1] * (sum(self.seq_cost_explore) / sum(self.seq_time_explore))  # Eq 16
            f2_exploit = self.PF[1] * (sum(self.seq_cost_exploit) / sum(self.seq_time_exploit))  # Eq 15

            if self.score_explore < self.score_exploit:
                self.mega_explore = max(self.mega_explore - 0.01, 0.01)
                self.mega_exploit = 0.99
            elif self.score_explore > self.score_exploit:
                self.mega_explore = 0.99
                self.mega_exploit = max(self.mega_exploit - 0.01, 0.01)

            lmn_explore = 1.0 - self.mega_explore  # Eq 24
            lmn_exploit = 1.0 - self.mega_exploit  # Eq 22

            self.score_explore = (
                self.mega_explore * f1_explore +
                self.mega_explore * f2_explore +
                lmn_explore * (min(self.pf_f3) * self.f3_explore)
            )  # Eq 20
            self.score_exploit = (
                self.mega_exploit * f1_exploit +
                self.mega_exploit * f2_exploit +
                lmn_exploit * (min(self.pf_f3) * self.f3_exploit)
            )  # Eq 19

            self.history.append((iter_count, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iter_count + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
