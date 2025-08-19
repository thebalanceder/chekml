# cython: language_level=3
# distutils: language = c++

import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport exp

# Define NumPy array types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int roulette_wheel_selection(np.ndarray[DTYPE_t, ndim=1] probabilities):
    """
    Perform roulette wheel selection based on given probabilities.
    
    Args:
        probabilities: Array of selection probabilities
    
    Returns:
        Selected index
    """
    cdef double r = rand() / <double>RAND_MAX
    cdef np.ndarray[DTYPE_t, ndim=1] cumsum = np.cumsum(probabilities)
    cdef int i
    for i in range(probabilities.shape[0]):
        if r <= cumsum[i]:
            return i
    return 0  # Fallback

cdef class ArtificialBeeColony:
    """
    Implementation of Artificial Bee Colony (ABC) optimization algorithm.
    
    Parameters:
    - objective_function: Function to optimize
    - dim: Number of decision variables
    - bounds: Tuple of (lower, upper) bounds for each dimension
    - population_size: Size of bee colony
    - max_iter: Maximum number of iterations
    - onlooker_ratio: Ratio of onlooker bees to population size
    - trial_limit_factor: Factor for abandonment limit
    - acceleration_bound: Upper bound for acceleration coefficient
    """
    
    cdef object objective_function
    cdef int dim
    cdef np.ndarray bounds
    cdef int population_size
    cdef int max_iter
    cdef int n_onlookers
    cdef int trial_limit
    cdef double acceleration_bound
    cdef np.ndarray population
    cdef np.ndarray costs
    cdef np.ndarray best_solution
    cdef double best_cost
    cdef np.ndarray trial_counters
    cdef list history
    
    def __init__(self, objective_function, int dim, bounds, int population_size=100, 
                 int max_iter=200, double onlooker_ratio=1.0, double trial_limit_factor=0.6, 
                 double acceleration_bound=1.0):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.n_onlookers = int(population_size * onlooker_ratio)
        self.trial_limit = int(trial_limit_factor * dim * population_size)
        self.acceleration_bound = acceleration_bound
        self.history = []
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void initialize_population(self):
        """Initialize population with random solutions within bounds."""
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                          (self.population_size, self.dim)).astype(DTYPE)
        self.costs = np.zeros(self.population_size, dtype=DTYPE)
        for i in range(self.population_size):
            self.costs[i] = self.objective_function(self.population[i])
        self.trial_counters = np.zeros(self.population_size, dtype=np.int32)
        
        # Initialize best solution
        cdef int min_idx = np.argmin(self.costs)
        self.best_solution = self.population[min_idx].copy()
        self.best_cost = self.costs[min_idx]
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void employed_bee_phase(self):
        """Perform employed bee phase (exploration)."""
        cdef int i, j, k
        cdef np.ndarray[DTYPE_t, ndim=1] new_position = np.zeros(self.dim, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] phi
        cdef double new_cost
        for i in range(self.population_size):
            # Select random bee (k != i)
            k = i
            while k == i:
                k = int(rand() % self.population_size)
            
            # Generate acceleration coefficient
            phi = self.acceleration_bound * np.random.uniform(-1, 1, self.dim).astype(DTYPE)
            
            # Create new solution
            for j in range(self.dim):
                new_position[j] = self.population[i, j] + phi[j] * (self.population[i, j] - self.population[k, j])
                new_position[j] = min(max(new_position[j], self.bounds[j, 0]), self.bounds[j, 1])
            
            # Evaluate new solution
            new_cost = self.objective_function(new_position)
            
            # Greedy selection
            if new_cost <= self.costs[i]:
                for j in range(self.dim):
                    self.population[i, j] = new_position[j]
                self.costs[i] = new_cost
                self.trial_counters[i] = 0
            else:
                self.trial_counters[i] += 1
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[DTYPE_t, ndim=1] calculate_selection_probabilities(self):
        """Calculate fitness values and selection probabilities."""
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = np.zeros(self.population_size, dtype=DTYPE)
        cdef double mean_cost = np.mean(self.costs)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = exp(-self.costs[i] / mean_cost)  # Convert cost to fitness
        return fitness / np.sum(fitness)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void onlooker_bee_phase(self):
        """Perform onlooker bee phase (exploitation)."""
        cdef np.ndarray[DTYPE_t, ndim=1] probabilities = self.calculate_selection_probabilities()
        cdef int m, i, j, k
        cdef np.ndarray[DTYPE_t, ndim=1] new_position = np.zeros(self.dim, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] phi
        cdef double new_cost
        for m in range(self.n_onlookers):
            # Select food source using roulette wheel
            i = roulette_wheel_selection(probabilities)
            
            # Select random bee (k != i)
            k = i
            while k == i:
                k = int(rand() % self.population_size)
            
            # Generate acceleration coefficient
            phi = self.acceleration_bound * np.random.uniform(-1, 1, self.dim).astype(DTYPE)
            
            # Create new solution
            for j in range(self.dim):
                new_position[j] = self.population[i, j] + phi[j] * (self.population[i, j] - self.population[k, j])
                new_position[j] = min(max(new_position[j], self.bounds[j, 0]), self.bounds[j, 1])
            
            # Evaluate new solution
            new_cost = self.objective_function(new_position)
            
            # Greedy selection
            if new_cost <= self.costs[i]:
                for j in range(self.dim):
                    self.population[i, j] = new_position[j]
                self.costs[i] = new_cost
                self.trial_counters[i] = 0
            else:
                self.trial_counters[i] += 1
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void scout_bee_phase(self):
        """Perform scout bee phase (abandonment and reinitialization)."""
        cdef int i, j
        for i in range(self.population_size):
            if self.trial_counters[i] >= self.trial_limit:
                for j in range(self.dim):
                    self.population[i, j] = self.bounds[j, 0] + (self.bounds[j, 1] - self.bounds[j, 0]) * (rand() / <double>RAND_MAX)
                self.costs[i] = self.objective_function(self.population[i])
                self.trial_counters[i] = 0
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update_best_solution(self):
        """Update the best solution found so far."""
        cdef int min_idx = np.argmin(self.costs)
        if self.costs[min_idx] < self.best_cost:
            self.best_solution = self.population[min_idx].copy()
            self.best_cost = self.costs[min_idx]
    
    cpdef tuple optimize(self):
        """Run the Artificial Bee Colony optimization."""
        self.initialize_population()
        
        cdef int iteration
        for iteration in range(self.max_iter):
            # Employed bee phase
            self.employed_bee_phase()
            
            # Onlooker bee phase
            self.onlooker_bee_phase()
            
            # Scout bee phase
            self.scout_bee_phase()
            
            # Update best solution
            self.update_best_solution()
            
            # Store iteration, best solution, and best cost
            self.history.append((iteration, self.best_solution.copy(), self.best_cost))
            
            # Display iteration information
            print(f"Iteration {iteration + 1}: Best Cost = {self.best_cost}")
        
        return self.best_solution, self.best_cost, self.history
