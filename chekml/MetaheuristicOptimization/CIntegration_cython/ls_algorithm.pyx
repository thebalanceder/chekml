import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport INFINITY

# Define types for NumPy arrays
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class LocalSearch:
    cdef object objective_function
    cdef int dim
    cdef np.ndarray bounds
    cdef int max_iter
    cdef double step_size
    cdef int neighbor_count
    cdef np.ndarray best_solution
    cdef double best_value
    cdef list history

    def __init__(self, object objective_function, int dim, list bounds, int max_iter=100, double step_size=0.1, int neighbor_count=10):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)  # Bounds as [(low, high), ...]
        self.max_iter = max_iter
        self.step_size = step_size
        self.neighbor_count = neighbor_count
        self.best_solution = np.zeros(dim, dtype=DTYPE)
        self.best_value = INFINITY
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray random_solution(self):
        """Generate a random solution within the given bounds"""
        cdef np.ndarray[DTYPE_t, ndim=1] solution = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], self.dim)
        return solution

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef list generate_neighbors(self, np.ndarray[DTYPE_t, ndim=1] current_solution):
        """Generate neighboring solutions by adding small perturbations"""
        cdef list neighbors = []
        cdef np.ndarray[DTYPE_t, ndim=1] perturbation
        cdef np.ndarray[DTYPE_t, ndim=1] new_solution
        cdef int i
        for i in range(self.neighbor_count):
            perturbation = np.random.uniform(-self.step_size, self.step_size, self.dim)
            new_solution = current_solution + perturbation
            new_solution = np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])
            neighbors.append(new_solution)
        return neighbors

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """Run the Local Search algorithm"""
        cdef np.ndarray[DTYPE_t, ndim=1] current_solution = self.random_solution()
        cdef double current_value = self.objective_function(current_solution)
        cdef np.ndarray[DTYPE_t, ndim=1] best_neighbor
        cdef double best_neighbor_value
        cdef list neighbors
        cdef np.ndarray[DTYPE_t, ndim=1] neighbor
        cdef double neighbor_value
        cdef int iteration, i

        self.best_solution = current_solution.copy()
        self.best_value = current_value

        for iteration in range(self.max_iter):
            neighbors = self.generate_neighbors(current_solution)

            # Evaluate all neighbors and select the best
            best_neighbor = current_solution
            best_neighbor_value = current_value

            for neighbor in neighbors:
                neighbor_value = self.objective_function(neighbor)
                if neighbor_value < best_neighbor_value:
                    best_neighbor = neighbor
                    best_neighbor_value = neighbor_value

            # If no better neighbor found, stop (local optimum reached)
            if best_neighbor_value >= current_value:
                print(f"Stopping at iteration {iteration + 1}: No better neighbor found.")
                break

            # Move to the best neighbor
            current_solution = best_neighbor
            current_value = best_neighbor_value

            # Store search history for visualization
            self.history.append((iteration, current_solution.copy(), current_value))

            # Update global best
            if best_neighbor_value < self.best_value:
                self.best_solution = best_neighbor
                self.best_value = best_neighbor_value

            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
