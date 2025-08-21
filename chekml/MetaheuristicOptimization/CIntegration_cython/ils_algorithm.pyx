# iterative_local_search.pyx
import numpy as np
cimport numpy as np
from libc.math cimport fabs
from libc.stdlib cimport rand, srand
from cython cimport boundscheck, wraparound, nonecheck

@boundscheck(False)
@wraparound(False)
@nonecheck(False)
cdef class IterativeLocalSearch:
    cdef object objective_function
    cdef np.ndarray bounds
    cdef int n_iterations, n_restarts, dim
    cdef double step_size, perturbation_size
    cdef object rng
    cdef np.ndarray best_solution
    cdef double best_fitness
    cdef list history

    def __init__(self, objective_function, bounds, int n_iterations=1000, double step_size=0.05,
                 int n_restarts=30, double perturbation_size=1.0, int seed_value=1):
        self.objective_function = objective_function
        self.bounds = np.array(bounds, dtype=np.float64)
        self.n_iterations = n_iterations
        self.step_size = step_size
        self.n_restarts = n_restarts
        self.perturbation_size = perturbation_size
        self.dim = len(bounds)
        self.rng = np.random.default_rng(seed_value)
        self.best_solution = np.zeros(self.dim, dtype=np.float64)
        self.best_fitness = float("inf")
        self.history = []

    cdef inline bint _in_bounds(self, np.ndarray point):
        cdef int i
        for i in range(self.dim):
            if point[i] < self.bounds[i, 0] or point[i] > self.bounds[i, 1]:
                return False
        return True

    cdef np.ndarray _random_point_within_bounds(self):
        return self.bounds[:, 0] + self.rng.random(self.dim) * (self.bounds[:, 1] - self.bounds[:, 0])

    cdef tuple _hill_climb(self, np.ndarray start_point):
        cdef np.ndarray solution = start_point.copy()
        cdef double solution_eval = self.objective_function(solution)
        cdef np.ndarray candidate
        cdef double candidate_eval
        cdef int _

        for _ in range(self.n_iterations):
            candidate = None
            while candidate is None or not self._in_bounds(candidate):
                candidate = solution + self.rng.normal(0, self.step_size, self.dim)

            candidate_eval = self.objective_function(candidate)

            if candidate_eval <= solution_eval:
                solution = candidate
                solution_eval = candidate_eval

        return solution, solution_eval

    def optimize(self):
        self.best_solution = self._random_point_within_bounds()
        self.best_fitness = self.objective_function(self.best_solution)
        cdef int restart
        cdef np.ndarray start_pt, solution
        cdef double solution_eval

        for restart in range(self.n_restarts):
            start_pt = None
            while start_pt is None or not self._in_bounds(start_pt):
                start_pt = self.best_solution + self.rng.normal(0, self.perturbation_size, self.dim)

            solution, solution_eval = self._hill_climb(start_pt)

            if solution_eval < self.best_fitness:
                self.best_solution = solution
                self.best_fitness = solution_eval
                print(f"Restart {restart}: Best Fitness = {self.best_fitness:.5f}")

            self.history.append((restart, self.best_solution.copy(), self.best_fitness))

        return self.best_solution, self.best_fitness, self.history

