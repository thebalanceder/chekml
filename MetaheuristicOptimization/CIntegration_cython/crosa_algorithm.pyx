import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport rand, RAND_MAX

# Ensure NumPy C API is initialized
cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:, :] initialize_crows(int population_size, int dim, double[:, :] bounds):
    """ Generate initial crow positions randomly """
    cdef double[:, :] crows = np.empty((population_size, dim), dtype=np.double)
    cdef double[:] lower_bound = bounds[:, 0]
    cdef double[:] upper_bound = bounds[:, 1]
    cdef int i, j
    cdef double r
    for i in range(population_size):
        for j in range(dim):
            r = <double>rand() / RAND_MAX
            crows[i, j] = lower_bound[j] + (upper_bound[j] - lower_bound[j]) * r
    return crows

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:] evaluate_crows(double[:, :] crows, object objective_function, int population_size):
    """ Compute fitness values for the crow positions """
    cdef double[:] fitness = np.empty(population_size, dtype=np.double)
    cdef int i
    for i in range(population_size):
        fitness[i] = objective_function(np.asarray(crows[i]))
    return fitness

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:, :] update_positions(double[:, :] crows, double[:, :] memory, int population_size, int dim, 
                                  double awareness_probability, double flight_length, double[:, :] bounds):
    """ Update crow positions based on CSA rules """
    cdef double[:, :] new_crows = crows.copy()
    cdef int[:] random_crows = np.random.choice(population_size, population_size).astype(np.intc)
    cdef double[:] lower_bound = bounds[:, 0]
    cdef double[:] upper_bound = bounds[:, 1]
    cdef int i, j
    cdef double r
    for i in range(population_size):
        r = <double>rand() / RAND_MAX
        if r > awareness_probability:
            # State 1: Follow another crow
            r = <double>rand() / RAND_MAX
            for j in range(dim):
                new_crows[i, j] = crows[i, j] + flight_length * r * (memory[random_crows[i], j] - crows[i, j])
        else:
            # State 2: Random position within bounds
            for j in range(dim):
                r = <double>rand() / RAND_MAX
                new_crows[i, j] = lower_bound[j] + (upper_bound[j] - lower_bound[j]) * r
        # Clip to bounds manually
        for j in range(dim):
            if new_crows[i, j] < lower_bound[j]:
                new_crows[i, j] = lower_bound[j]
            elif new_crows[i, j] > upper_bound[j]:
                new_crows[i, j] = upper_bound[j]
    return new_crows

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void update_memory(double[:, :] crows, double[:] fitness, double[:, :] memory, double[:] fitness_memory,
                        int population_size, int dim, double[:, :] bounds):
    """ Update crow positions and memory based on fitness """
    cdef int i, j
    cdef bint within_bounds
    for i in range(population_size):
        # Check if new position is within bounds
        within_bounds = True
        for j in range(dim):
            if crows[i, j] < bounds[j, 0] or crows[i, j] > bounds[j, 1]:
                within_bounds = False
                break
        if within_bounds:
            # Update position
            for j in range(dim):
                crows[i, j] = crows[i, j]
            # Update memory if new fitness is better
            if fitness[i] < fitness_memory[i]:
                for j in range(dim):
                    memory[i, j] = crows[i, j]
                fitness_memory[i] = fitness[i]

class CrowSearchAlgorithm:
    def __init__(self, objective_function, dim: int, bounds, population_size: int = 20, max_iter: int = 5000, 
                 awareness_probability: float = 0.1, flight_length: float = 2.0):
        """
        Initialize the Crow Search Algorithm (CSA) optimizer.

        Parameters:
        - objective_function: Function to optimize (minimization).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of crows (solutions).
        - max_iter: Maximum number of iterations.
        - awareness_probability: Probability of crow awareness (AP).
        - flight_length: Flight length parameter (fl).
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.double)
        self.population_size = population_size
        self.max_iter = max_iter
        self.awareness_probability = awareness_probability
        self.flight_length = flight_length

        self.crows = None
        self.memory = None
        self.fitness_memory = None
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_crows(self):
        """ Initialize crow positions and memory """
        self.crows = initialize_crows(self.population_size, self.dim, self.bounds)
        self.memory = self.crows.copy()
        self.fitness_memory = evaluate_crows(self.crows, self.objective_function, self.population_size)

    def optimize(self):
        """ Run the Crow Search Algorithm """
        self.initialize_crows()

        for iteration in range(self.max_iter):
            # Update positions
            new_crows = update_positions(self.crows, self.memory, self.population_size, self.dim,
                                         self.awareness_probability, self.flight_length, self.bounds)
            new_fitness = evaluate_crows(new_crows, self.objective_function, self.population_size)

            # Update positions and memory
            update_memory(new_crows, new_fitness, self.crows, self.fitness_memory,
                          self.population_size, self.dim, self.bounds)

            # Track best solution
            min_fitness_idx = np.argmin(self.fitness_memory)
            if self.fitness_memory[min_fitness_idx] < self.best_value:
                self.best_solution = np.asarray(self.memory[min_fitness_idx]).copy()
                self.best_value = self.fitness_memory[min_fitness_idx]

            self.history.append((iteration, np.asarray(self.best_solution).copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history


