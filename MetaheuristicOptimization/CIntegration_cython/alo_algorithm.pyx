import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport cos, sin, sqrt, pi, exp
from libc.stdlib cimport rand, RAND_MAX
from libc.string cimport memset

# Ensure NumPy C API is initialized
cnp.import_array()

# Define types for NumPy arrays
DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.intp_t INDEX_t  # Use intp_t for integer indices

@cython.boundscheck(False)  # Disable bounds checking for performance
@cython.wraparound(False)   # Disable negative indexing
@cython.cdivision(True)     # Disable division checks
cdef class AntLionOptimization:
    cdef object objective_function
    cdef int dim
    cdef int population_size
    cdef int max_iter
    cdef cnp.ndarray bounds
    cdef cnp.ndarray antlion_positions
    cdef cnp.ndarray ant_positions
    cdef cnp.ndarray elite_antlion_position
    cdef double elite_antlion_fitness
    cdef cnp.ndarray sorted_antlions
    cdef list history

    def __init__(self, objective_function, int dim, bounds, int population_size=40, int max_iter=500):
        """
        Initialize the Ant Lion Optimizer (ALO).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: List of tuples [(lower, upper), ...] for each dimension or single values if same for all.
        - population_size: Number of search agents (antlions and ants).
        - max_iter: Maximum number of iterations.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter
        
        # Handle bounds
        if isinstance(bounds, (list, tuple, np.ndarray)) and len(bounds) == 2 and not isinstance(bounds[0], (list, tuple, np.ndarray)):
            self.bounds = np.array([[bounds[0], bounds[1]] for _ in range(dim)], dtype=DTYPE)
        else:
            self.bounds = np.array(bounds, dtype=DTYPE)
        
        self.antlion_positions = None
        self.ant_positions = None
        self.elite_antlion_position = None
        self.elite_antlion_fitness = float("inf")
        self.sorted_antlions = None
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_positions(self) except *:
        """ Generate initial random positions for antlions and ants """
        cdef cnp.ndarray[DTYPE_t, ndim=1] lb = self.bounds[:, 0]
        cdef cnp.ndarray[DTYPE_t, ndim=1] ub = self.bounds[:, 1]
        cdef int i, j
        cdef double r
        
        self.antlion_positions = np.zeros((self.population_size, self.dim), dtype=DTYPE)
        self.ant_positions = np.zeros((self.population_size, self.dim), dtype=DTYPE)
        
        if len(lb) == 1:
            for i in range(self.population_size):
                for j in range(self.dim):
                    r = <double>rand() / RAND_MAX
                    self.antlion_positions[i, j] = lb[0] + (ub[0] - lb[0]) * r
                    r = <double>rand() / RAND_MAX
                    self.ant_positions[i, j] = lb[0] + (ub[0] - lb[0]) * r
        else:
            for i in range(self.population_size):
                for j in range(self.dim):
                    r = <double>rand() / RAND_MAX
                    self.antlion_positions[i, j] = lb[j] + (ub[j] - lb[j]) * r
                    r = <double>rand() / RAND_MAX
                    self.ant_positions[i, j] = lb[j] + (ub[j] - lb[j]) * r

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[DTYPE_t, ndim=1] evaluate_fitness(self, cnp.ndarray[DTYPE_t, ndim=2] positions):
        """ Compute fitness values for given positions """
        cdef int i
        cdef cnp.ndarray[DTYPE_t, ndim=1] fitness = np.zeros(positions.shape[0], dtype=DTYPE)
        for i in range(positions.shape[0]):
            fitness[i] = self.objective_function(positions[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int roulette_wheel_selection(self, cnp.ndarray[DTYPE_t, ndim=1] weights):
        """ Perform roulette wheel selection based on weights """
        cdef int i
        cdef double accumulation = 0.0
        cdef double p
        cdef cnp.ndarray[DTYPE_t, ndim=1] cumsum = np.zeros(len(weights), dtype=DTYPE)
        
        # Compute cumulative sum
        for i in range(len(weights)):
            accumulation += weights[i]
            cumsum[i] = accumulation
        
        p = (<double>rand() / RAND_MAX) * accumulation
        
        for i in range(len(cumsum)):
            if cumsum[i] > p:
                return i
        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[DTYPE_t, ndim=2] random_walk_around_antlion(self, cnp.ndarray[DTYPE_t, ndim=1] antlion, int current_iter):
        """ Create random walks around a given antlion """
        cdef cnp.ndarray[DTYPE_t, ndim=1] lb = self.bounds[:, 0].copy()
        cdef cnp.ndarray[DTYPE_t, ndim=1] ub = self.bounds[:, 1].copy()
        cdef double I = 1.0
        cdef int i, t
        cdef double r
        cdef cnp.ndarray[DTYPE_t, ndim=2] walks = np.zeros((self.max_iter + 1, self.dim), dtype=DTYPE)
        cdef cnp.ndarray[DTYPE_t, ndim=1] X
        cdef double a, b, c, d
        
        # Adjust ratio I based on iteration
        if current_iter > self.max_iter / 10:
            I = 1 + 100 * (current_iter / <double>self.max_iter)
        if current_iter > self.max_iter / 2:
            I = 1 + 1000 * (current_iter / <double>self.max_iter)
        if current_iter > self.max_iter * 3 / 4:
            I = 1 + 10000 * (current_iter / <double>self.max_iter)
        if current_iter > self.max_iter * 0.9:
            I = 1 + 100000 * (current_iter / <double>self.max_iter)
        if current_iter > self.max_iter * 0.95:
            I = 1 + 1000000 * (current_iter / <double>self.max_iter)

        # Decrease boundaries
        for i in range(self.dim):
            lb[i] /= I
            ub[i] /= I

        # Move interval around antlion
        r = <double>rand() / RAND_MAX
        if r < 0.5:
            for i in range(self.dim):
                lb[i] += antlion[i]
        else:
            for i in range(self.dim):
                lb[i] = -lb[i] + antlion[i]

        r = <double>rand() / RAND_MAX
        if r >= 0.5:
            for i in range(self.dim):
                ub[i] += antlion[i]
        else:
            for i in range(self.dim):
                ub[i] = -ub[i] + antlion[i]

        # Generate random walks
        for i in range(self.dim):
            X = np.zeros(self.max_iter + 1, dtype=DTYPE)
            for t in range(self.max_iter):
                r = <double>rand() / RAND_MAX
                X[t + 1] = X[t] + (2 * (r > 0.5) - 1)
            
            a = np.min(X)
            b = np.max(X)
            c = lb[i]
            d = ub[i]
            
            for t in range(self.max_iter + 1):
                walks[t, i] = ((X[t] - a) * (d - c)) / (b - a + 1e-10) + c

        return walks

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """ Run the Ant Lion Optimization algorithm """
        cdef int current_iter, i, j, roulette_idx
        cdef cnp.ndarray[DTYPE_t, ndim=1] antlions_fitness
        cdef cnp.ndarray[DTYPE_t, ndim=1] ants_fitness
        cdef cnp.ndarray[DTYPE_t, ndim=2] double_population
        cdef cnp.ndarray[DTYPE_t, ndim=1] double_fitness
        cdef cnp.ndarray[INDEX_t, ndim=1] sorted_indices
        cdef cnp.ndarray[DTYPE_t, ndim=2] RA, RE
        
        # Initialize positions
        self.initialize_positions()
        
        # Evaluate initial antlion fitness
        antlions_fitness = self.evaluate_fitness(self.antlion_positions)
        
        # Sort antlions
        sorted_indices = np.argsort(antlions_fitness).astype(np.intp)
        self.sorted_antlions = self.antlion_positions[sorted_indices].copy()
        antlions_fitness = antlions_fitness[sorted_indices]
        
        # Set elite antlion
        self.elite_antlion_position = self.sorted_antlions[0].copy()
        self.elite_antlion_fitness = antlions_fitness[0]
        
        # Store initial best solution in history
        self.history.append((0, self.elite_antlion_position.copy(), self.elite_antlion_fitness))
        
        # Main loop
        for current_iter in range(1, self.max_iter):
            # Simulate random walks for each ant
            for i in range(self.population_size):
                # Select antlion using roulette wheel
                roulette_idx = self.roulette_wheel_selection(1 / (antlions_fitness + 1e-10))
                if roulette_idx == -1:
                    roulette_idx = 0
                
                # Random walk around selected antlion
                RA = self.random_walk_around_antlion(self.sorted_antlions[roulette_idx], current_iter)
                
                # Random walk around elite antlion
                RE = self.random_walk_around_antlion(self.elite_antlion_position, current_iter)
                
                # Update ant position (Equation 2.13)
                for j in range(self.dim):
                    self.ant_positions[i, j] = (RA[current_iter, j] + RE[current_iter, j]) / 2
            
            # Boundary checking
            for i in range(self.population_size):
                for j in range(self.dim):
                    if self.ant_positions[i, j] < self.bounds[j, 0]:
                        self.ant_positions[i, j] = self.bounds[j, 0]
                    elif self.ant_positions[i, j] > self.bounds[j, 1]:
                        self.ant_positions[i, j] = self.bounds[j, 1]
            
            # Evaluate ant fitness
            ants_fitness = self.evaluate_fitness(self.ant_positions)
            
            # Combine populations and update antlions
            double_population = np.vstack((self.sorted_antlions, self.ant_positions)).astype(DTYPE)
            double_fitness = np.concatenate((antlions_fitness, ants_fitness)).astype(DTYPE)
            
            sorted_indices = np.argsort(double_fitness).astype(np.intp)
            double_population = double_population[sorted_indices]
            double_fitness = double_fitness[sorted_indices]
            
            antlions_fitness = double_fitness[:self.population_size]
            self.sorted_antlions = double_population[:self.population_size].copy()
            
            # Update elite if better solution found
            if antlions_fitness[0] < self.elite_antlion_fitness:
                self.elite_antlion_position = self.sorted_antlions[0].copy()
                self.elite_antlion_fitness = antlions_fitness[0]
            
            # Ensure elite is in population
            self.sorted_antlions[0] = self.elite_antlion_position
            antlions_fitness[0] = self.elite_antlion_fitness
            
            # Store history
            self.history.append((current_iter, self.elite_antlion_position.copy(), self.elite_antlion_fitness))
            
            # Display progress every 50 iterations
            if (current_iter + 1) % 50 == 0:
                print(f"At iteration {current_iter + 1}, the elite fitness is {self.elite_antlion_fitness}")

        return self.elite_antlion_position, self.elite_antlion_fitness, self.history
