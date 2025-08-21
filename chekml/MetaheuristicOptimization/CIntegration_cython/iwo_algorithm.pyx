# cython: language_level=3
import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport floor

# Ensure NumPy C API is initialized
cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class InvasiveWeedOptimization:
    cdef object objective_function
    cdef int dim
    cdef double[:, :] bounds
    cdef public int max_iter  # Public for Python access
    cdef int initial_pop_size
    cdef int max_pop_size
    cdef int min_seeds
    cdef int max_seeds
    cdef double exponent
    cdef double sigma_initial
    cdef double sigma_final
    cdef double[:, :] population
    cdef double[:] best_solution
    cdef double best_cost
    cdef public list history  # Public for Python access

    def __init__(self, objective_function, int dim, bounds, int max_iter=200, 
                 int initial_pop_size=10, int max_pop_size=25, int min_seeds=1, 
                 int max_seeds=5, double exponent=2.0, double sigma_initial=0.5, 
                 double sigma_final=0.001):
        """
        Initialize the Invasive Weed Optimization (IWO) algorithm.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: List of (lower, upper) bounds for each dimension.
        - max_iter: Maximum number of iterations.
        - initial_pop_size: Initial population size.
        - max_pop_size: Maximum population size.
        - min_seeds: Minimum number of seeds per plant.
        - max_seeds: Maximum number of seeds per plant.
        - exponent: Variance reduction exponent.
        - sigma_initial: Initial standard deviation.
        - sigma_final: Final standard deviation.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.double)
        self.max_iter = max_iter
        self.initial_pop_size = initial_pop_size
        self.max_pop_size = max_pop_size
        self.min_seeds = min_seeds
        self.max_seeds = max_seeds
        self.exponent = exponent
        self.sigma_initial = sigma_initial
        self.sigma_final = sigma_final
        self.population = None
        self.best_solution = None
        self.best_cost = float("inf")
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_population(self):
        """ Generate initial population of plants randomly """
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                           (self.initial_pop_size, self.dim)).astype(np.double)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double[:] evaluate_population(self):
        """ Compute fitness values for the population """
        cdef int i
        cdef double[:] costs = np.zeros(self.population.shape[0], dtype=np.double)
        for i in range(self.population.shape[0]):
            costs[i] = self.objective_function(self.population[i])
        return costs

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double update_standard_deviation(self, int iteration):
        """ Update standard deviation based on iteration """
        return ((self.max_iter - iteration) / (self.max_iter - 1.0)) ** self.exponent * \
               (self.sigma_initial - self.sigma_final) + self.sigma_final

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double[:, :] reproduction(self, double[:] costs, double sigma):
        """ Generate seeds based on fitness values """
        cdef double best_cost = np.min(costs)
        cdef double worst_cost = np.max(costs)
        cdef list new_population = []
        cdef int i, j, k
        cdef int num_seeds
        cdef double ratio
        cdef double[:] new_plant

        for i in range(self.population.shape[0]):
            # Calculate number of seeds based on fitness
            if best_cost == worst_cost:
                num_seeds = self.min_seeds
            else:
                ratio = (costs[i] - worst_cost) / (best_cost - worst_cost)
                num_seeds = <int>floor(self.min_seeds + (self.max_seeds - self.min_seeds) * ratio)

            # Generate seeds
            for j in range(num_seeds):
                new_plant = np.zeros(self.dim, dtype=np.double)
                for k in range(self.dim):
                    new_plant[k] = self.population[i, k] + sigma * np.random.randn()
                    new_plant[k] = max(new_plant[k], self.bounds[k, 0])
                    new_plant[k] = min(new_plant[k], self.bounds[k, 1])
                new_population.append(np.array(new_plant))

        if not new_population:
            return self.population.copy()
        return np.array(new_population, dtype=np.double)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple optimize(self):
        """ Run the Invasive Weed Optimization algorithm """
        cdef int iteration, min_idx
        cdef double[:] costs
        cdef double[:, :] new_population
        cdef double sigma
        cdef cnp.ndarray[cnp.intp_t, ndim=1] sort_order

        self.initialize_population()

        for iteration in range(self.max_iter):
            # Evaluate current population
            costs = self.evaluate_population()
            min_idx = np.argmin(costs)
            if costs[min_idx] < self.best_cost:
                self.best_solution = self.population[min_idx].copy()
                self.best_cost = costs[min_idx]

            # Update standard deviation
            sigma = self.update_standard_deviation(iteration)

            # Reproduction phase
            new_population = self.reproduction(costs, sigma)

            # Merge populations only if new_population is not empty
            if new_population.shape[0] > 0:
                self.population = np.vstack([self.population, new_population])
            else:
                print(f"Iteration {iteration + 1}: No new seeds generated, keeping current population")

            # Sort population by cost
            costs = self.evaluate_population()
            sort_order = np.argsort(costs)
            self.population = np.array(self.population)[sort_order]

            # Competitive exclusion
            if self.population.shape[0] > self.max_pop_size:
                self.population = self.population[:self.max_pop_size]

            # Store best cost for this iteration
            self.history.append((iteration, np.array(self.best_solution), self.best_cost))

            # Display iteration information
            print(f"Iteration {iteration + 1}: Best Cost = {self.best_cost}")

        return np.array(self.best_solution), self.best_cost, self.history
