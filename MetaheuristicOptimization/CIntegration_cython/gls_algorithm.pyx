# cython: language_level=3
import numpy as np
cimport numpy as np

cdef class GuidedLocalSearch:
    cdef:
        object objective_function
        object feature_indicators_func
        int dim
        object bounds
        double lambda_
        int max_iterations

        object feature_costs
        object lower_bounds
        object upper_bounds
        object penalties

        object best_solution
        double best_fitness
        list history

    def __init__(self, objective_function, feature_costs, feature_indicators,
                 int dim=2, bounds=None, double lambda_=0.1, int max_iterations=100):
        self.objective_function = objective_function
        self.feature_costs = np.array(feature_costs, dtype=np.float64)
        self.feature_indicators_func = feature_indicators
        self.lambda_ = lambda_
        self.dim = dim
        self.bounds = bounds if bounds is not None else [(-5, 5)] * dim
        self.max_iterations = max_iterations

        self.lower_bounds = np.array([b[0] for b in self.bounds], dtype=np.float64)
        self.upper_bounds = np.array([b[1] for b in self.bounds], dtype=np.float64)

        self.penalties = np.zeros(len(self.feature_costs), dtype=np.float64)
        self.best_solution = None
        self.best_fitness = float("inf")
        self.history = []

    cpdef optimize(self):
        cdef int k = 0, i
        cdef np.ndarray[np.float64_t, ndim=1] s = self._construct_initial_solution()
        cdef np.ndarray[np.float64_t, ndim=1] l, utilities
        cdef double current_fitness, max_utility
        cdef np.ndarray[np.intp_t, ndim=1] max_indices

        while k < self.max_iterations:
            s = self._improvement_method(s)

            l = np.array(self.feature_indicators_func(s), dtype=np.float64)
            utilities = l * self.feature_costs / (1.0 + self.penalties)
            max_utility = np.max(utilities)
            max_indices = np.where(utilities == max_utility)[0]

            for i in max_indices:
                self.penalties[i] += 1

            current_fitness = self.objective_function(s)
            if current_fitness < self.best_fitness:
                self.best_fitness = current_fitness
                self.best_solution = s.copy()

            self.history.append((k, self.best_solution.copy()))
            print(f"Iteration {k + 1}: Best Fitness = {self.best_fitness}")
            k += 1

        return self.best_solution, self.best_fitness, self.history

    cdef np.ndarray[np.float64_t, ndim=1] _construct_initial_solution(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds)

    cdef double _evaluate_augmented_objective(self, np.ndarray[np.float64_t, ndim=1] solution):
        cdef double g = self.objective_function(solution)
        cdef np.ndarray[np.float64_t, ndim=1] l = np.array(self.feature_indicators_func(solution), dtype=np.float64)
        cdef double penalty_term = np.dot(self.penalties, l)
        return g + self.lambda_ * penalty_term

    cdef np.ndarray[np.float64_t, ndim=1] _improvement_method(self, np.ndarray[np.float64_t, ndim=1] solution):
        cdef np.ndarray[np.float64_t, ndim=1] neighbor = solution + np.random.uniform(-0.1, 0.1, self.dim)
        neighbor = np.clip(neighbor, self.lower_bounds, self.upper_bounds)
        if self._evaluate_augmented_objective(neighbor) < self._evaluate_augmented_objective(solution):
            return neighbor
        return solution

