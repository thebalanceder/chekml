import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt

# Ensure NumPy C API is initialized
cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class ArtificialAlgaeAlgorithm:
    cdef object objective_function
    cdef int dim
    cdef cnp.ndarray bounds
    cdef int population_size
    cdef int max_iter
    cdef double step_size
    cdef cnp.ndarray population
    cdef cnp.ndarray best_solution
    cdef double best_value
    cdef list history

    def __init__(self, objective_function, int dim, bounds, int population_size=50, int max_iter=100, double step_size=0.1):
        """
        Initialize the Artificial Algae Algorithm (AAA) optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Array of (lower, upper) bounds for each dimension.
        - population_size: Number of algae (solutions).
        - max_iter: Maximum number of iterations.
        - step_size: Step size for movement towards best solution.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.double)
        self.population_size = population_size
        self.max_iter = max_iter
        self.step_size = step_size
        self.best_value = float("inf")
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def initialize_population(self):
        """ Generate initial algae population randomly """
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                           (self.population_size, self.dim))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def evaluate_population(self):
        """ Compute fitness values for the algae population """
        cdef cnp.ndarray[cnp.double_t, ndim=1] fitness = np.empty(self.population_size, dtype=np.double)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.population[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def movement_phase(self):
        """ Move each algae towards the best solution """
        cdef int i, j
        cdef cnp.ndarray[cnp.double_t, ndim=1] direction
        cdef cnp.ndarray[cnp.double_t, ndim=2] population = self.population
        cdef cnp.ndarray[cnp.double_t, ndim=1] best_solution = self.best_solution
        cdef cnp.ndarray[cnp.double_t, ndim=1] lower_bounds = self.bounds[:, 0]
        cdef cnp.ndarray[cnp.double_t, ndim=1] upper_bounds = self.bounds[:, 1]
        cdef double step_size = self.step_size

        for i in range(self.population_size):
            # Calculate direction towards best solution
            direction = best_solution - population[i]
            # Update position
            for j in range(self.dim):
                population[i, j] += step_size * direction[j]
                # Ensure new position is within bounds
                if population[i, j] < lower_bounds[j]:
                    population[i, j] = lower_bounds[j]
                elif population[i, j] > upper_bounds[j]:
                    population[i, j] = upper_bounds[j]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """ Run the Artificial Algae Algorithm """
        self.initialize_population()
        cdef cnp.ndarray[cnp.double_t, ndim=1] fitness
        cdef int iteration, min_idx, i
        cdef double current_value

        for iteration in range(self.max_iter):
            # Evaluate fitness for each individual
            fitness = self.evaluate_population()
            # Find the best individual
            min_idx = np.argmin(fitness)
            current_value = fitness[min_idx]
            if current_value < self.best_value:
                self.best_solution = self.population[min_idx].copy()
                self.best_value = current_value

            # Update population positions
            self.movement_phase()

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

# Example objective function (Sphere function)
def sphere_function(x):
    return np.sum(x ** 2)

# Example usage
if __name__ == "__main__":
    # Define problem parameters
    dim = 10
    bounds = [(-5, 5)] * dim  # Bounds for each dimension
    population_size = 50
    max_iter = 100
    step_size = 0.1

    # Initialize and run optimizer
    optimizer = ArtificialAlgaeAlgorithm(sphere_function, dim, bounds, population_size, max_iter, step_size)
    best_solution, best_value, history = optimizer.optimize()

    print("\nOptimization Complete!")
    print(f"Best Solution: {best_solution}")
    print(f"Best Fitness: {best_value}")
