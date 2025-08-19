# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sin, cos, sqrt, pow
from libc.stdlib cimport rand, RAND_MAX

# Define NumPy types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef class PopulationVortexSearch:
    cdef object objective_function
    cdef int dim
    cdef np.ndarray bounds
    cdef int population_size
    cdef int max_iter
    cdef double prob_mut
    cdef double prob_cross
    cdef int vortex_size
    cdef int distribution_index
    cdef np.ndarray center
    cdef double best_value
    cdef np.ndarray best_solution
    cdef np.ndarray candidates
    cdef list iter_results
    cdef int function_evals

    def __init__(self, objective_function, int dim, bounds, int population_size=50, int max_iter=1000, prob_mut=None, prob_cross=None):
        """
        Initialize the Population-based Vortex Search (PVS) optimizer.
        
        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Array of (lower, upper) bounds for each dimension.
        - population_size: Number of candidate solutions.
        - max_iter: Maximum number of iterations (function evaluations).
        - prob_mut: Probability of mutation (default: 1/dim).
        - prob_cross: Probability of crossover (default: 1/dim).
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)
        self.population_size = population_size
        self.max_iter = max_iter
        self.prob_mut = prob_mut if prob_mut is not None else 1.0 / dim
        self.prob_cross = prob_cross if prob_cross is not None else 1.0 / dim
        self.vortex_size = population_size // 2
        self.distribution_index = 20

        self.center = None
        self.best_value = float("inf")
        self.best_solution = None
        self.candidates = None
        self.iter_results = []
        self.function_evals = 0

    cdef void initialize_vortex(self):
        """ Initialize the vortex center and candidate solutions """
        cdef np.ndarray[DTYPE_t, ndim=1] LB = self.bounds[:, 0]
        cdef np.ndarray[DTYPE_t, ndim=1] UB = self.bounds[:, 1]
        cdef double x = 0.1
        cdef double a = 1.0
        cdef double ginv = (1.0 / x) * 1.0 / np.random.gamma(a, x)
        cdef np.ndarray[DTYPE_t, ndim=1] radius = ginv * ((UB - LB) / 2.0)

        self.center = 0.5 * (UB + LB)
        self.candidates = np.random.normal(loc=self.center, scale=radius, size=(self.population_size, self.dim))
        self._bound_solutions()

    cdef void _bound_solutions(self):
        """ Ensure candidate solutions are within bounds """
        cdef np.ndarray[DTYPE_t, ndim=1] LB = self.bounds[:, 0]
        cdef np.ndarray[DTYPE_t, ndim=1] UB = self.bounds[:, 1]
        cdef int i, j
        cdef double[:, :] candidates = self.candidates
        cdef double r

        for i in range(self.population_size):
            for j in range(self.dim):
                if candidates[i, j] < LB[j]:
                    r = <double>rand() / RAND_MAX
                    candidates[i, j] = LB[j] + r * (UB[j] - LB[j])
                elif candidates[i, j] > UB[j]:
                    r = <double>rand() / RAND_MAX
                    candidates[i, j] = LB[j] + r * (UB[j] - LB[j])

    cdef np.ndarray[DTYPE_t, ndim=1] evaluate_candidates(self, int[:] indices=None):
        """ Evaluate the objective function for candidate solutions """
        if indices is None:
            return np.array([self.objective_function(self.candidates[i]) for i in range(self.population_size)], dtype=DTYPE)
        return np.array([self.objective_function(self.candidates[i]) for i in indices], dtype=DTYPE)

    cdef int polynomial_mutation(self, np.ndarray[DTYPE_t, ndim=1] solution, np.ndarray[DTYPE_t, ndim=1] mutated):
        """ Apply polynomial mutation to a solution, storing result in mutated """
        cdef np.ndarray[DTYPE_t, ndim=1] LB = self.bounds[:, 0]
        cdef np.ndarray[DTYPE_t, ndim=1] UB = self.bounds[:, 1]
        cdef int state = 0
        cdef double mut_pow = 1.0 / (1.0 + self.distribution_index)
        cdef int i
        cdef double y, yL, yU, delta1, delta2, rnd, xy, val, deltaq

        for i in range(self.dim):
            mutated[i] = solution[i]  # Copy input to output
            if <double>rand() / RAND_MAX <= self.prob_mut:
                y = solution[i]
                yL, yU = LB[i], UB[i]
                delta1 = (y - yL) / (yU - yL)
                delta2 = (yU - y) / (yU - yL)
                rnd = <double>rand() / RAND_MAX

                if rnd <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rnd + (1.0 - 2.0 * rnd) * pow(xy, self.distribution_index + 1)
                    deltaq = pow(val, mut_pow) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * pow(xy, self.distribution_index + 1)
                    deltaq = 1.0 - pow(val, mut_pow)

                y = y + deltaq * (yU - yL)
                if y < yL:
                    y = yL
                elif y > yU:
                    y = yU
                mutated[i] = y
                state += 1

        return state

    cdef np.ndarray[DTYPE_t, ndim=1] first_phase(self, np.ndarray[DTYPE_t, ndim=1] radius, int iteration):
        """ Generate new candidate solutions around the vortex center """
        cdef np.ndarray[DTYPE_t, ndim=2] candidates
        cdef int[:] indices
        if iteration == 0:
            candidates = np.random.normal(loc=self.center, scale=radius, size=(self.population_size, self.dim))
            self.candidates = candidates
            self._bound_solutions()
            return self.evaluate_candidates()
        else:
            candidates = np.random.normal(loc=self.center, scale=radius, size=(self.vortex_size, self.dim))
            self.candidates[:self.vortex_size] = candidates
            self._bound_solutions()
            indices = np.arange(self.vortex_size, dtype=np.int32)
            return self.evaluate_candidates(indices)

    cdef np.ndarray[DTYPE_t, ndim=1] second_phase(self, np.ndarray[DTYPE_t, ndim=1] obj_vals, int iteration):
        """ Apply crossover and mutation to remaining candidates """
        cdef np.ndarray[DTYPE_t, ndim=1] LB = self.bounds[:, 0]
        cdef np.ndarray[DTYPE_t, ndim=1] UB = self.bounds[:, 1]
        cdef np.ndarray[DTYPE_t, ndim=1] prob, full_obj_vals
        cdef int i, d, neighbor, param2change
        cdef double obj_val_sol, obj_val_mut
        cdef np.ndarray[DTYPE_t, ndim=1] sol
        cdef np.ndarray[DTYPE_t, ndim=1] mutated = np.empty(self.dim, dtype=DTYPE)
        cdef int state

        # Ensure obj_vals is full size for all candidates
        if iteration > 0:
            full_obj_vals = self.evaluate_candidates()
            full_obj_vals[:self.vortex_size] = obj_vals
            obj_vals = full_obj_vals
            self.function_evals += self.population_size - self.vortex_size
        else:
            obj_vals = obj_vals.copy()

        prob = 0.9 * (np.max(obj_vals) - obj_vals) + 0.1
        prob = np.cumsum(prob / np.sum(prob))
        
        for i in range(self.vortex_size, self.population_size):
            neighbor = np.searchsorted(prob, <double>rand() / RAND_MAX)
            while i == neighbor:
                neighbor = np.searchsorted(prob, <double>rand() / RAND_MAX)

            sol = self.candidates[i].copy()
            param2change = rand() % self.dim
            for d in range(self.dim):
                if <double>rand() / RAND_MAX < self.prob_cross or d == param2change:
                    sol[d] += (self.candidates[i, d] - self.candidates[neighbor, d]) * (<double>rand() / RAND_MAX - 0.5) * 2.0

            for d in range(self.dim):
                if sol[d] < LB[d]:
                    sol[d] = LB[d]
                elif sol[d] > UB[d]:
                    sol[d] = UB[d]

            obj_val_sol = self.objective_function(sol)
            self.function_evals += 1

            if obj_val_sol < obj_vals[i]:
                self.candidates[i] = sol
                obj_vals[i] = obj_val_sol
            else:
                state = self.polynomial_mutation(self.candidates[i], mutated)
                if state > 0:
                    obj_val_mut = self.objective_function(mutated)
                    self.function_evals += 1
                    if obj_val_mut < obj_vals[i]:
                        self.candidates[i] = mutated.copy()
                        obj_vals[i] = obj_val_mut

        return obj_vals

    def optimize(self, double error=1e-6, double optimal_value=0):
        """ Run the Population-based Vortex Search optimization """
        self.initialize_vortex()
        cdef np.ndarray[DTYPE_t, ndim=1] LB = self.bounds[:, 0]
        cdef np.ndarray[DTYPE_t, ndim=1] UB = self.bounds[:, 1]
        cdef double x = 0.1
        cdef int iteration = 0
        cdef double a, ginv
        cdef np.ndarray[DTYPE_t, ndim=1] radius, obj_vals
        cdef int min_idx
        cdef double fmin
        cdef np.ndarray[DTYPE_t, ndim=1] itr_best

        while self.function_evals < self.max_iter:
            a = (self.max_iter - self.function_evals) / self.max_iter
            a = max(a, 0.1)
            ginv = (1.0 / x) * 1.0 / np.random.gamma(a, x)
            radius = ginv * ((UB - LB) / 2.0)

            obj_vals = self.first_phase(radius, iteration)
            self.function_evals += self.population_size if iteration == 0 else self.vortex_size

            min_idx = np.argmin(obj_vals)
            fmin = obj_vals[min_idx]
            itr_best = self.candidates[min_idx]

            if fmin < self.best_value:
                self.best_value = fmin
                self.best_solution = itr_best.copy()
                self.center = itr_best.copy()

            obj_vals = self.second_phase(obj_vals, iteration)

            min_idx = np.argmin(obj_vals)
            fmin = obj_vals[min_idx]
            itr_best = self.candidates[min_idx]

            if fmin < self.best_value:
                self.best_value = fmin
                self.best_solution = itr_best.copy()
                self.center = itr_best.copy()

            self.iter_results.append({"iteration": iteration, "best_value": self.best_value, "best_solution": self.best_solution.copy()})
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

            if abs(optimal_value - self.best_value) <= error:
                break

            iteration += 1

        return self.best_solution, self.best_value, self.iter_results
