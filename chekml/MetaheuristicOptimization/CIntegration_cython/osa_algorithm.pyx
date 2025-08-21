import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport fmin, fmax

# Define numpy types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef class OwlSearchOptimizer:
    cdef public:
        object objective_function
        int dim
        np.ndarray bounds
        int population_size
        int max_iter
        double step_size
        double p_explore
        np.ndarray population
        np.ndarray best_solution
        double best_value
        list history

    def __init__(self, objective_function, int dim, bounds, int population_size=50, 
                 int max_iter=100, double step_size=0.1, double p_explore=0.1):
        """
        Initialize the Owl Search Algorithm (OSA) optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Array of (lower, upper) bounds for each dimension.
        - population_size: Number of owls (solutions).
        - max_iter: Maximum number of iterations.
        - step_size: Step size for movement.
        - p_explore: Probability of exploration.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)
        self.population_size = population_size
        self.max_iter = max_iter
        self.step_size = step_size
        self.p_explore = p_explore
        self.best_value = float("inf")
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_population(self):
        """Generate initial owl population randomly within bounds"""
        cdef DTYPE_t[:, :] bounds_view = self.bounds
        cdef DTYPE_t[:, :] population = np.empty((self.population_size, self.dim), dtype=DTYPE)
        cdef int i, j
        cdef double lb, ub
        for i in range(self.population_size):
            for j in range(self.dim):
                lb = bounds_view[j, 0]
                ub = bounds_view[j, 1]
                population[i, j] = lb + (ub - lb) * (<double>rand() / RAND_MAX)
        self.population = np.asarray(population)  # Convert memoryview to NumPy array

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[DTYPE_t, ndim=1] evaluate_population(self):
        """Compute fitness values for the owl population"""
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = np.empty(self.population_size, dtype=DTYPE)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.population[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[DTYPE_t, ndim=1] exploration_phase(self, int index):
        """Simulate owl exploration with random movement"""
        cdef np.ndarray[DTYPE_t, ndim=1] random_move = np.empty(self.dim, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] new_solution = np.empty(self.dim, dtype=DTYPE)
        cdef DTYPE_t[:, :] bounds_view = self.bounds
        cdef int j
        for j in range(self.dim):
            random_move[j] = self.step_size * (2.0 * (<double>rand() / RAND_MAX) - 1.0)
            new_solution[j] = self.population[index, j] + random_move[j]
            new_solution[j] = fmax(fmin(new_solution[j], bounds_view[j, 1]), bounds_view[j, 0])
        return new_solution

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[DTYPE_t, ndim=1] exploitation_phase(self, int index):
        """Simulate owl exploitation by moving towards the best solution"""
        cdef np.ndarray[DTYPE_t, ndim=1] new_solution = np.empty(self.dim, dtype=DTYPE)
        cdef DTYPE_t[:, :] bounds_view = self.bounds
        cdef int j
        for j in range(self.dim):
            new_solution[j] = self.population[index, j] + self.step_size * (self.best_solution[j] - self.population[index, j])
            new_solution[j] = fmax(fmin(new_solution[j], bounds_view[j, 1]), bounds_view[j, 0])
        return new_solution

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """Run the Owl Search Algorithm optimization"""
        self.initialize_population()
        cdef np.ndarray[DTYPE_t, ndim=1] fitness
        cdef np.ndarray[DTYPE_t, ndim=1] new_solution
        cdef int iteration, i, min_idx
        cdef double min_fitness
        cdef double r

        for iteration in range(self.max_iter):
            # Evaluate fitness for each owl
            fitness = self.evaluate_population()
            min_idx = np.argmin(fitness)
            min_fitness = fitness[min_idx]
            
            # Update best solution if a better one is found
            if min_fitness < self.best_value:
                self.best_value = min_fitness
                self.best_solution = self.population[min_idx].copy()

            # Update each owl's position
            for i in range(self.population_size):
                r = <double>rand() / RAND_MAX
                if r < self.p_explore:
                    # Exploration: Random movement
                    new_solution = self.exploration_phase(i)
                else:
                    # Exploitation: Move towards best solution
                    new_solution = self.exploitation_phase(i)
                self.population[i] = new_solution

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

