import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport sqrt, log, cos

# Define pi for Box-Muller transform
cdef double pi = 3.141592653589793

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class BiogeographyBasedOptimizer:
    cdef public:
        object objective_function
        int dim
        cnp.ndarray bounds
        int population_size
        int max_iter
        double keep_rate
        int n_keep
        int n_new
        double alpha
        double mutation_prob
        double mutation_sigma
        cnp.ndarray mu
        cnp.ndarray lambda_
        cnp.ndarray habitats
        cnp.ndarray best_solution
        double best_value
        list history

    def __init__(self, objective_function, int dim, bounds, int population_size=50, int max_iter=100, 
                 double keep_rate=0.2, double alpha=0.9, double mutation_prob=0.1, double mutation_scale=0.02):
        """
        Initialize the Biogeography-Based Optimization (BBO) algorithm.

        Parameters:
        - objective_function: Function to optimize (minimize).
        - dim: Number of dimensions (variables).
        - bounds: List of (lower, upper) bounds for each dimension.
        - population_size: Number of habitats (solutions).
        - max_iter: Maximum number of iterations.
        - keep_rate: Proportion of best habitats to keep each iteration.
        - alpha: Migration step size control parameter.
        - mutation_prob: Probability of mutation for each variable.
        - mutation_scale: Scale of mutation (relative to bounds range).
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.double)
        self.population_size = population_size
        self.max_iter = max_iter
        self.keep_rate = keep_rate
        self.n_keep = <int>(keep_rate * population_size)
        self.n_new = population_size - self.n_keep
        self.alpha = alpha
        self.mutation_prob = mutation_prob
        self.mutation_sigma = mutation_scale * (self.bounds[0, 1] - self.bounds[0, 0])

        # Migration rates
        self.mu = np.linspace(1, 0, population_size, dtype=np.double)
        self.lambda_ = 1 - self.mu

        self.habitats = None
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_habitats(self):
        """Generate initial habitats randomly within bounds."""
        cdef cnp.ndarray[cnp.double_t, ndim=2] habitats
        habitats = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                     (self.population_size, self.dim))
        self.habitats = self.enforce_bounds(habitats)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.double_t, ndim=1] evaluate_habitats(self):
        """Compute fitness values for all habitats."""
        cdef cnp.ndarray[cnp.double_t, ndim=1] fitness = np.empty(self.population_size, dtype=np.double)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.habitats[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int roulette_wheel_selection(self, cnp.ndarray[cnp.double_t, ndim=1] probabilities):
        """Select an index using roulette wheel selection."""
        cdef double r = rand() / <double>RAND_MAX
        cdef cnp.ndarray[cnp.double_t, ndim=1] cumsum = np.cumsum(probabilities)
        cdef int i
        for i in range(self.population_size):
            if r <= cumsum[i]:
                return i
        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void migration(self, cnp.ndarray[cnp.double_t, ndim=2] new_habitats, int habitat_idx, int var_idx):
        """Perform migration for a single variable of a habitat."""
        cdef double r = rand() / <double>RAND_MAX
        if r <= self.lambda_[habitat_idx]:
            # Emigration probabilities
            ep = self.mu.copy()  # Python-level NumPy array
            ep[habitat_idx] = 0
            ep_sum = np.sum(ep)  # Python-level variable
            if ep_sum > 0:
                ep /= ep_sum
            
            # Select source habitat
            source_idx = self.roulette_wheel_selection(ep)  # Python-level int
            
            # Migration
            new_habitats[habitat_idx, var_idx] += (
                self.alpha * (self.habitats[source_idx, var_idx] - self.habitats[habitat_idx, var_idx])
            )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void mutation(self, cnp.ndarray[cnp.double_t, ndim=2] new_habitats, int habitat_idx, int var_idx):
        """Apply mutation to a single variable of a habitat."""
        cdef double r = rand() / <double>RAND_MAX
        cdef double u1, u2, z
        if r <= self.mutation_prob:
            # Box-Muller transform for Gaussian random number
            u1 = rand() / <double>RAND_MAX
            u2 = rand() / <double>RAND_MAX
            z = sqrt(-2.0 * log(u1)) * cos(2 * pi * u2)
            new_habitats[habitat_idx, var_idx] += self.mutation_sigma * z

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.double_t, ndim=2] enforce_bounds(self, cnp.ndarray[cnp.double_t, ndim=2] habitats):
        """Ensure all habitats are within the specified bounds."""
        return np.clip(habitats, self.bounds[:, 0], self.bounds[:, 1])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """Run the Biogeography-Based Optimization algorithm."""
        self.initialize_habitats()
        cdef cnp.ndarray[cnp.double_t, ndim=1] fitness
        cdef cnp.ndarray[cnp.double_t, ndim=2] new_habitats
        cdef cnp.ndarray[cnp.intp_t, ndim=1] sorted_indices
        cdef int iteration, i, k
        cdef double best_fitness
        
        for iteration in range(self.max_iter):
            # Evaluate current population
            fitness = self.evaluate_habitats()
            sorted_indices = np.argsort(fitness)
            self.habitats = self.habitats[sorted_indices]
            fitness = fitness[sorted_indices]
            
            # Update best solution, ensuring it's within bounds
            if fitness[0] < self.best_value:
                self.best_solution = self.enforce_bounds(self.habitats[0:1])[0]
                self.best_value = self.objective_function(self.best_solution)
            
            # Create new population
            new_habitats = self.habitats.copy()
            
            # Migration and mutation
            for i in range(self.population_size):
                for k in range(self.dim):
                    self.migration(new_habitats, i, k)
                    self.mutation(new_habitats, i, k)
                
                # Apply bounds after migration and mutation
                new_habitats[i] = self.enforce_bounds(new_habitats[i:i+1])[0]
                
                # Evaluate new habitat
                fitness[i] = self.objective_function(new_habitats[i])
            
            # Sort new population
            sorted_indices = np.argsort(fitness)
            new_habitats = new_habitats[sorted_indices]
            fitness = fitness[sorted_indices]
            
            # Select next iteration population
            self.habitats = np.vstack((self.habitats[:self.n_keep], new_habitats[:self.n_new]))
            
            # Ensure all habitats are within bounds
            self.habitats = self.enforce_bounds(self.habitats)
            
            # Re-evaluate to ensure consistency
            fitness = self.evaluate_habitats()
            sorted_indices = np.argsort(fitness)
            self.habitats = self.habitats[sorted_indices]
            
            # Store history with best solution
            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}, Best Solution = {self.best_solution}")
        
        return self.best_solution, self.best_value, self.history
