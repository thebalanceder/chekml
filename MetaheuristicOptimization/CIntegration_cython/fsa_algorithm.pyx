import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport INFINITY

# Define numpy types for static typing
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef class FutureSearchOptimizer:
    cdef public:
        object objective_function
        int dim
        np.ndarray bounds
        int population_size
        int max_iter
        int num_runs
        np.ndarray population
        np.ndarray local_best_positions
        np.ndarray local_best_values
        np.ndarray global_best_position
        double global_best_value
        list global_best_history
        list best_positions

    def __init__(self, object objective_function, int dim, np.ndarray bounds, 
                 int population_size=30, int max_iter=100, int num_runs=30):
        """
        Initialize the Future Search Algorithm optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Array of (lower, upper) bounds for each dimension.
        - population_size: Number of solutions in the population.
        - max_iter: Maximum number of iterations.
        - num_runs: Number of independent runs.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.asarray(bounds, dtype=DTYPE)
        self.population_size = population_size
        self.max_iter = max_iter
        self.num_runs = num_runs
        self.global_best_value = INFINITY
        self.global_best_history = []
        self.best_positions = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void initialize_population(self):
        """ Generate initial population randomly within bounds """
        cdef np.ndarray[DTYPE_t, ndim=2] population
        cdef np.ndarray[DTYPE_t, ndim=2] local_best_positions
        cdef np.ndarray[DTYPE_t, ndim=1] local_best_values
        cdef int i
        cdef double fitness

        population = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], 
            (self.population_size, self.dim)
        ).astype(DTYPE)
        self.population = population
        self.local_best_positions = population.copy()
        
        local_best_values = np.empty(self.population_size, dtype=DTYPE)
        for i in range(self.population_size):
            local_best_values[i] = self.objective_function(population[i])
        
        self.local_best_values = local_best_values
        i = np.argmin(local_best_values)
        self.global_best_position = population[i].copy()
        self.global_best_value = local_best_values[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[DTYPE_t, ndim=1] clip_to_bounds(self, np.ndarray[DTYPE_t, ndim=1] solution):
        """ Ensure solution stays within bounds """
        return np.clip(solution, self.bounds[:, 0], self.bounds[:, 1])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update_population(self):
        """ Update population using global and local bests """
        cdef int i, j
        cdef double new_fitness
        cdef np.ndarray[DTYPE_t, ndim=1] new_solution
        cdef np.ndarray[DTYPE_t, ndim=2] population = self.population
        cdef np.ndarray[DTYPE_t, ndim=2] local_best_positions = self.local_best_positions
        cdef np.ndarray[DTYPE_t, ndim=1] local_best_values = self.local_best_values
        cdef np.ndarray[DTYPE_t, ndim=1] global_best_position = self.global_best_position

        for i in range(self.population_size):
            # Update rule: Move towards global and local bests
            new_solution = population[i].copy()
            for j in range(self.dim):
                new_solution[j] += (
                    (-population[i, j] + global_best_position[j]) * np.random.rand() +
                    (-population[i, j] + local_best_positions[i, j]) * np.random.rand()
                )
            new_solution = self.clip_to_bounds(new_solution)
            population[i] = new_solution

            # Evaluate new solution
            new_fitness = self.objective_function(new_solution)

            # Update local best
            if new_fitness <= local_best_values[i]:
                local_best_positions[i] = new_solution.copy()
                local_best_values[i] = new_fitness

            # Update global best
            if new_fitness <= self.global_best_value:
                self.global_best_position = new_solution.copy()
                self.global_best_value = new_fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update_with_initial_strategy(self):
        """ Update solutions using initial strategy based on global and local bests """
        cdef int i, j
        cdef double new_fitness
        cdef np.ndarray[DTYPE_t, ndim=2] temp_population
        cdef np.ndarray[DTYPE_t, ndim=1] temp_fitness
        cdef np.ndarray[DTYPE_t, ndim=2] population = self.population
        cdef np.ndarray[DTYPE_t, ndim=2] local_best_positions = self.local_best_positions
        cdef np.ndarray[DTYPE_t, ndim=1] local_best_values = self.local_best_values
        cdef np.ndarray[DTYPE_t, ndim=1] global_best_position = self.global_best_position

        temp_population = np.empty((self.population_size, self.dim), dtype=DTYPE)
        for i in range(self.population_size):
            for j in range(self.dim):
                temp_population[i, j] = (
                    global_best_position[j] + 
                    (global_best_position[j] - local_best_positions[i, j]) * np.random.rand()
                )
            temp_population[i] = self.clip_to_bounds(temp_population[i])

            # Evaluate new solution
            new_fitness = self.objective_function(temp_population[i])

            # Update if better than previous fitness
            if new_fitness <= local_best_values[i]:
                population[i] = temp_population[i].copy()
                local_best_positions[i] = temp_population[i].copy()
                local_best_values[i] = new_fitness

        # Update global best based on temporary population
        temp_fitness = np.empty(self.population_size, dtype=DTYPE)
        for i in range(self.population_size):
            temp_fitness[i] = self.objective_function(temp_population[i])
        
        i = np.argmin(temp_fitness)
        if temp_fitness[i] <= self.global_best_value:
            self.global_best_position = temp_population[i].copy()
            self.global_best_value = temp_fitness[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple optimize(self):
        """ Run the Future Search Algorithm for multiple runs """
        cdef int run, iteration
        cdef list best_values_iter

        for run in range(self.num_runs):
            self.initialize_population()
            best_values_iter = []

            for iteration in range(self.max_iter):
                self.update_population()
                self.update_with_initial_strategy()
                best_values_iter.append(self.global_best_value)

                # Display progress
                print(f"Run {run + 1}, Iteration {iteration + 1}: Best Value = {self.global_best_value}")

            # Store results for this run
            self.global_best_history.append(best_values_iter)
            self.best_positions.append(self.global_best_position.copy())

        # Find the best result across all runs
        best_run_idx = np.argmin([min(history) for history in self.global_best_history])
        best_score = min(self.global_best_history[best_run_idx])
        best_position = self.best_positions[best_run_idx]

        print(f"Best Score across all runs: {best_score}")
        print(f"Best Position: {best_position}")

        return best_position, best_score, self.global_best_history
