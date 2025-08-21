# variable_neighborhood_search.pyx
import numpy as np
cimport numpy as np
import cython
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time

cdef class VariableNeighborhoodSearch:
    cdef object obj_func
    cdef int dim, max_iterations, num_neighborhoods
    cdef list bounds, neighborhood_sizes
    cdef double mutation_rate

    def __init__(self, object objective_function, int dim, list bounds,
                 int max_iterations=100, list neighborhood_sizes=[1, 2, 3],
                 double mutation_rate=0.1):
        self.obj_func = objective_function
        self.dim = dim
        self.bounds = bounds
        self.max_iterations = max_iterations
        self.neighborhood_sizes = neighborhood_sizes
        self.num_neighborhoods = len(neighborhood_sizes)
        self.mutation_rate = mutation_rate

    cdef np.ndarray _generate_neighbor(self, np.ndarray current_solution, int neighborhood_size):
        cdef np.ndarray[np.float64_t, ndim=1] mutation = self.mutation_rate * (np.random.rand(self.dim) - 0.5) * neighborhood_size
        cdef np.ndarray[np.float64_t, ndim=1] neighbor = current_solution + mutation
        lower_bounds = np.array([b[0] for b in self.bounds], dtype=np.float64)
        upper_bounds = np.array([b[1] for b in self.bounds], dtype=np.float64)
        neighbor = np.clip(neighbor, lower_bounds, upper_bounds)
        return neighbor

    def optimize(self):
        cdef int i, neighborhood_index, neighborhood_size
        cdef np.ndarray[np.float64_t, ndim=1] current_solution, neighbor
        cdef double current_value, neighbor_value, best_value
        cdef list best_solution, history

        current_solution = np.array([np.random.uniform(b[0], b[1]) for b in self.bounds], dtype=np.float64)
        current_value = self.obj_func(current_solution)
        best_solution = current_solution.tolist()
        best_value = current_value
        history = [(best_value, best_solution[:])]

        for i in range(self.max_iterations):
            neighborhood_index = np.random.randint(0, self.num_neighborhoods)
            neighborhood_size = self.neighborhood_sizes[neighborhood_index]

            neighbor = self._generate_neighbor(current_solution, neighborhood_size)
            neighbor_value = self.obj_func(neighbor)

            if neighbor_value < current_value:
                current_solution = neighbor
                current_value = neighbor_value

                if current_value < best_value:
                    best_solution = current_solution.tolist()
                    best_value = current_value

            history.append((best_value, best_solution[:]))

        return best_solution, best_value, history

