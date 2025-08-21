import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt

# Define NumPy types for Cython
DTYPE = np.double
ctypedef np.double_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class JaguarAlgorithmOptimizer:
    cdef public:
        object objective_function
        int dim
        np.ndarray bounds
        int population_size
        int max_iter
        double p_cruise
        double cruising_distance
        double alpha
        np.ndarray population
        np.ndarray best_solution
        double best_value
        list history

    def __init__(self, object objective_function, int dim, bounds, int population_size=50, int max_iter=100,
                 double p_cruise=0.8, double cruising_distance=0.1, double alpha=0.1):
        """
        Initialize the Jaguar Algorithm Optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of jaguars (solutions).
        - max_iter: Maximum number of iterations.
        - p_cruise: Probability of cruising (controls exploration).
        - cruising_distance: Maximum cruising distance.
        - alpha: Learning rate for position updates.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.p_cruise = p_cruise
        self.cruising_distance = cruising_distance
        self.alpha = alpha
        self.best_value = float("inf")
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_population(self):
        """ Generate initial jaguar population randomly """
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                           (self.population_size, self.dim)).astype(DTYPE)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[DTYPE_t, ndim=1] evaluate_population(self):
        """ Compute fitness values for the jaguar population """
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = np.empty(self.population_size, dtype=DTYPE)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.population[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[DTYPE_t, ndim=1] cruising_phase(self, int index, int iteration):
        """ Simulate cruising behavior for top-performing jaguars """
        cdef np.ndarray[DTYPE_t, ndim=1] direction = np.random.uniform(-1, 1, self.dim).astype(DTYPE)
        cdef double norm = sqrt(np.sum(direction ** 2))
        cdef double current_cruising_distance = self.cruising_distance * (1 - iteration / self.max_iter)
        cdef np.ndarray[DTYPE_t, ndim=1] new_solution = np.empty(self.dim, dtype=DTYPE)
        cdef int j

        # Normalize direction
        for j in range(self.dim):
            direction[j] /= norm

        # Update position
        for j in range(self.dim):
            new_solution[j] = self.population[index, j] + self.alpha * current_cruising_distance * direction[j]

        # Clip to bounds
        for j in range(self.dim):
            if new_solution[j] < self.bounds[j, 0]:
                new_solution[j] = self.bounds[j, 0]
            elif new_solution[j] > self.bounds[j, 1]:
                new_solution[j] = self.bounds[j, 1]

        return new_solution

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[DTYPE_t, ndim=1] random_walk_phase(self, int index):
        """ Simulate random walk for non-cruising jaguars """
        cdef np.ndarray[DTYPE_t, ndim=1] direction = np.random.uniform(-1, 1, self.dim).astype(DTYPE)
        cdef double norm = sqrt(np.sum(direction ** 2))
        cdef np.ndarray[DTYPE_t, ndim=1] new_solution = np.empty(self.dim, dtype=DTYPE)
        cdef int j

        # Normalize direction
        for j in range(self.dim):
            direction[j] /= norm

        # Update position
        for j in range(self.dim):
            new_solution[j] = self.population[index, j] + self.alpha * direction[j]

        # Clip to bounds
        for j in range(self.dim):
            if new_solution[j] < self.bounds[j, 0]:
                new_solution[j] = self.bounds[j, 0]
            elif new_solution[j] > self.bounds[j, 1]:
                new_solution[j] = self.bounds[j, 1]

        return new_solution

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """ Run the Jaguar Algorithm Optimization """
        self.initialize_population()
        cdef np.ndarray[DTYPE_t, ndim=1] fitness
        cdef np.ndarray[np.intp_t, ndim=1] sorted_indices
        cdef int iteration, i, num_cruising
        cdef double best_fitness

        for iteration in range(self.max_iter):
            # Evaluate fitness
            fitness = self.evaluate_population()

            # Sort population based on fitness
            sorted_indices = np.argsort(fitness)
            fitness = fitness[sorted_indices]
            self.population = self.population[sorted_indices]

            # Update best solution
            if fitness[0] < self.best_value:
                self.best_value = fitness[0]
                self.best_solution = self.population[0].copy()

            # Determine number of cruising jaguars
            num_cruising = <int>(self.p_cruise * self.population_size)

            # Update cruising jaguars
            for i in range(num_cruising):
                self.population[i] = self.cruising_phase(i, iteration)

            # Update remaining jaguars with random walk
            for i in range(num_cruising, self.population_size):
                self.population[i] = self.random_walk_phase(i)

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

# Example objective function (Sphere function)
cpdef double sphere_function(np.ndarray[DTYPE_t, ndim=1] x):
    return np.sum(x ** 2)

# Example usage
if __name__ == "__main__":
    dim = 10
    bounds = [(-5, 5)] * dim
    optimizer = JaguarAlgorithmOptimizer(sphere_function, dim, bounds)
    best_solution, best_value, history = optimizer.optimize()
    print(f"Best Solution: {best_solution}")
    print(f"Best Value: {best_value}")
