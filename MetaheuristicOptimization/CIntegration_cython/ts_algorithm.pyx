# tabu_search.pyx

import numpy as np
cimport numpy as np
from libc.math cimport fabs
from random import uniform

cdef class TabuSearch:
    cdef:
        object obj_func
        int dim, max_iter, tabu_tenure, neighborhood_size
        list bounds

    def __init__(self, objective_function, int dim, list bounds, int max_iter=200, int tabu_tenure=10, int neighborhood_size=20):
        self.obj_func = objective_function
        self.dim = dim
        self.bounds = bounds
        self.max_iter = max_iter
        self.tabu_tenure = tabu_tenure
        self.neighborhood_size = neighborhood_size

    cdef np.ndarray _generate_neighbor(self, np.ndarray current_solution):
        cdef np.ndarray neighbor = current_solution + np.random.uniform(-0.1, 0.1, self.dim)
        for i in range(self.dim):
            neighbor[i] = min(max(neighbor[i], self.bounds[i][0]), self.bounds[i][1])
        return neighbor

    cdef tuple _evaluate_neighbors(self, np.ndarray current_solution, dict tabu_list):
        cdef np.ndarray best_candidate = None
        cdef float best_candidate_value = float('inf')
        cdef tuple candidate_move = None

        for _ in range(self.neighborhood_size):
            neighbor = self._generate_neighbor(current_solution)
            move = tuple(np.round(neighbor - current_solution, 4))

            if move in tabu_list:
                continue

            value = self.obj_func(neighbor)
            if value < best_candidate_value:
                best_candidate = neighbor
                best_candidate_value = value
                candidate_move = move

        return best_candidate, best_candidate_value, candidate_move

    cpdef tuple optimize(self):
        cdef np.ndarray current_solution = np.array([uniform(b[0], b[1]) for b in self.bounds], dtype=np.float64)
        cdef float current_value = self.obj_func(current_solution)

        cdef np.ndarray best_solution = current_solution.copy()
        cdef float best_value = current_value

        cdef dict tabu_list = {}
        cdef list history = [(current_value, current_solution.tolist())]

        cdef int iteration
        for iteration in range(self.max_iter):
            neighbor, neighbor_value, move = self._evaluate_neighbors(current_solution, tabu_list)

            if neighbor is None:
                break

            current_solution = neighbor
            current_value = neighbor_value
            history.append((current_value, current_solution.tolist()))

            if current_value < best_value:
                best_solution = current_solution.copy()
                best_value = current_value

            if move is not None:
                tabu_list[move] = self.tabu_tenure
            tabu_list = {k: v - 1 for k, v in tabu_list.items() if v > 1}

        return best_solution.tolist(), best_value, history

