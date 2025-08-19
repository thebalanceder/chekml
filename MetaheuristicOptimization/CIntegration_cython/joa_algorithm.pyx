import numpy as np
cimport numpy as cnp
cimport cython
from cpython cimport PyObject

# Define types for NumPy arrays
ctypedef cnp.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class JointOperationsAlgorithm:
    cdef object objective_function
    cdef int num_variables
    cdef cnp.ndarray bounds
    cdef int num_subpopulations
    cdef int population_size_per_subpop
    cdef int max_iterations
    cdef double mutation_rate
    cdef list populations
    cdef cnp.ndarray best_solution
    cdef double best_value
    cdef list history

    def __init__(self, object objective_function, int num_variables, cnp.ndarray bounds, 
                 int num_subpopulations=5, int population_size_per_subpop=10, 
                 int max_iterations=100, double mutation_rate=0.1):
        """
        Initialize the Joint Operations Algorithm (JOA) optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - num_variables: Number of dimensions (variables).
        - bounds: Array of (lower, upper) bounds for each dimension.
        - num_subpopulations: Number of subpopulations.
        - population_size_per_subpop: Population size per subpopulation.
        - max_iterations: Maximum number of iterations.
        - mutation_rate: Rate of movement towards other individuals.
        """
        self.objective_function = objective_function
        self.num_variables = num_variables
        self.bounds = bounds
        self.num_subpopulations = num_subpopulations
        self.population_size_per_subpop = population_size_per_subpop
        self.max_iterations = max_iterations
        self.mutation_rate = mutation_rate
        self.populations = None
        self.best_solution = None
        self.best_value = np.inf
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_populations(self):
        """Generate random initial populations for each subpopulation."""
        cdef int i
        self.populations = []
        for i in range(self.num_subpopulations):
            self.populations.append(
                np.random.uniform(
                    self.bounds[:, 0], 
                    self.bounds[:, 1], 
                    (self.population_size_per_subpop, self.num_variables)
                )
            )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray evaluate_populations(self):
        """Compute fitness values for all individuals in each subpopulation."""
        cdef cnp.ndarray[DTYPE_t, ndim=2] fitness = np.zeros(
            (self.num_subpopulations, self.population_size_per_subpop), dtype=np.float64
        )
        cdef int i, j
        cdef cnp.ndarray[DTYPE_t, ndim=2] pop
        for i in range(self.num_subpopulations):
            pop = self.populations[i]
            for j in range(self.population_size_per_subpop):
                fitness[i, j] = self.objective_function(pop[j, :])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_populations(self):
        """Update each subpopulation's position based on random interactions."""
        cdef int i, j, other_subpop_index, other_individual_index
        cdef cnp.ndarray[DTYPE_t, ndim=2] current_pop, other_pop
        cdef cnp.ndarray[DTYPE_t, ndim=1] direction
        for i in range(self.num_subpopulations):
            current_pop = self.populations[i]
            for j in range(self.population_size_per_subpop):
                # Select a random subpopulation (excluding the current one)
                other_subpop_index = np.random.randint(0, self.num_subpopulations - 1)
                if other_subpop_index >= i:
                    other_subpop_index += 1

                # Select a random individual from the selected subpopulation
                other_individual_index = np.random.randint(0, self.population_size_per_subpop)

                # Move towards the selected individual
                other_pop = self.populations[other_subpop_index]
                direction = other_pop[other_individual_index, :] - current_pop[j, :]
                current_pop[j, :] += self.mutation_rate * direction

                # Ensure the new position is within bounds
                current_pop[j, :] = np.clip(
                    current_pop[j, :], 
                    self.bounds[:, 0], 
                    self.bounds[:, 1]
                )
            self.populations[i] = current_pop

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """Run the Joint Operations Algorithm optimization."""
        self.initialize_populations()
        cdef cnp.ndarray fitness
        cdef double min_fitness
        cdef int min_i, min_j, flat_idx
        cdef int iteration
        for iteration in range(self.max_iterations):
            # Evaluate fitness for all subpopulations
            fitness = self.evaluate_populations()

            # Find the best solution in this iteration
            min_fitness = np.min(fitness)
            flat_idx = np.argmin(fitness)
            min_i = flat_idx // self.population_size_per_subpop  # Compute row index
            min_j = flat_idx % self.population_size_per_subpop   # Compute column index
            if min_fitness < self.best_value:
                self.best_value = min_fitness
                self.best_solution = self.populations[min_i][min_j, :].copy()

            # Update populations
            self.update_populations()

            # Store history
            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

# Example objective function (Sphere function)
def sphere_function(x):
    return np.sum(x ** 2)
