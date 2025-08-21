import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport rand, RAND_MAX
from cpython cimport PyObject

# Define numpy types
ctypedef cnp.int32_t DTYPE_INT
ctypedef cnp.float64_t DTYPE_FLOAT
ctypedef cnp.int64_t DTYPE_INDEX

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class KeyCuttingAlgorithm:
    cdef public int num_buses, num_branches, key_length, population_size, max_iter
    cdef public double probability_threshold
    cdef public cnp.ndarray keys, best_solution
    cdef public double best_fitness
    cdef public list history
    cdef object objective_function

    def __init__(self, objective_function, int num_buses, int num_branches, 
                 int population_size=20, int max_iter=100, double probability_threshold=0.5):
        """
        Initialize the Key Cutting Algorithm (KCA) for optimization.

        Parameters:
        - objective_function: Function to evaluate fitness.
        - num_buses: Number of buses (or dimensions for general use).
        - num_branches: Number of branches (or additional bits).
        - population_size: Number of keys (solutions).
        - max_iter: Maximum number of iterations.
        - probability_threshold: Threshold for key tooth modification.
        """
        self.objective_function = objective_function
        self.num_buses = num_buses
        self.num_branches = num_branches
        self.key_length = num_buses + num_branches
        self.population_size = population_size
        self.max_iter = max_iter
        self.probability_threshold = probability_threshold
        self.best_fitness = float("inf")
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_keys(self):
        """ Generate initial key set randomly with binary values (0 or 1) """
        cdef cnp.ndarray[DTYPE_INT, ndim=2] keys = np.random.randint(0, 2, 
                                                                    (self.population_size, self.key_length), 
                                                                    dtype=np.int32)
        self.keys = keys

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[DTYPE_FLOAT, ndim=1] evaluate_keys(self):
        """ Compute fitness values for all keys """
        cdef int i
        cdef cnp.ndarray[DTYPE_FLOAT, ndim=1] fitness = np.empty(self.population_size, dtype=np.float64)
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.keys[i], self.num_buses)
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[DTYPE_FLOAT, ndim=2] calculate_probability_factor(self):
        """ Calculate probability factor for each key tooth """
        cdef int i, j
        cdef int half_pop = self.population_size // 2
        cdef cnp.ndarray[DTYPE_FLOAT, ndim=2] prob_factors = np.zeros((half_pop, self.key_length), dtype=np.float64)
        cdef cnp.ndarray[DTYPE_INT, ndim=2] selected_keys = self.keys[:half_pop]
        cdef double tooth_sum
        for j in range(self.key_length):
            tooth_sum = 0
            for i in range(half_pop):
                tooth_sum += selected_keys[i, j]
            for i in range(half_pop):
                prob_factors[i, j] = 1.0 - (tooth_sum / half_pop)
        return prob_factors

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void generate_new_keys(self, cnp.ndarray[DTYPE_FLOAT, ndim=2] prob_factors, bint use_kca1):
        """ Generate second half of keys based on probability factors """
        cdef int i, j
        cdef int half_pop = self.population_size // 2
        cdef cnp.ndarray[DTYPE_INT, ndim=2] new_keys = self.keys[:half_pop].copy()
        cdef double random_num
        for i in range(half_pop):
            for j in range(self.key_length):
                random_num = rand() / float(RAND_MAX)
                if (use_kca1 and random_num > prob_factors[i, j]) or (not use_kca1 and random_num < prob_factors[i, j]):
                    new_keys[i, j] = 1 - new_keys[i, j]
        self.keys[half_pop:] = new_keys

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self, bint use_kca1=True):
        """
        Run the Key Cutting Algorithm (KCA) optimization.

        Parameters:
        - use_kca1: If True, use KCA1; if False, use KCA2.

        Returns:
        - best_solution: Best key found.
        - best_fitness: Best fitness value.
        - history: List of (generation, best_solution, best_fitness).
        """
        self.initialize_keys()
        cdef int generation, min_idx, i
        cdef cnp.ndarray[DTYPE_FLOAT, ndim=1] fitness
        cdef cnp.ndarray[DTYPE_INDEX, ndim=1] sorted_indices
        cdef cnp.ndarray[DTYPE_FLOAT, ndim=2] prob_factors

        for generation in range(self.max_iter):
            # Evaluate fitness
            fitness = self.evaluate_keys()
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_fitness:
                self.best_solution = self.keys[min_idx].copy()
                self.best_fitness = fitness[min_idx]

            # Select top 50% keys
            sorted_indices = np.argsort(fitness)
            self.keys = self.keys[sorted_indices]

            # Calculate probability factors
            prob_factors = self.calculate_probability_factor()

            # Generate new keys
            self.generate_new_keys(prob_factors, use_kca1)

            # Log history
            self.history.append((generation, self.best_solution.copy(), self.best_fitness))
            print(f"Iteration {generation + 1}: Best Fitness = {self.best_fitness}")

        return self.best_solution, self.best_fitness, self.history

# Example generic fitness function
def binary_fitness(cnp.ndarray[DTYPE_INT, ndim=1] key, int num_buses):
    """
    Generic fitness function for binary optimization.

    Parameters:
    - key: Binary array.
    - num_buses: Number of dimensions or buses.

    Returns:
    - Fitness value.
    """
    # Example: Sum of bits (minimize number of 1s)
    return np.sum(key)
