import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, tanh, cos, sqrt
from numpy cimport ndarray

# Define numpy types for Cython
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef class EmperorPenguinOptimizer:
    cdef object objective_function
    cdef int dim
    cdef ndarray bounds
    cdef int population_size
    cdef int max_iter
    cdef double f, l, M
    cdef int adaptation_interval
    cdef ndarray penguins
    cdef ndarray best_solution
    cdef double best_value
    cdef list history
    cdef list strategies
    cdef ndarray strategy_probs
    cdef ndarray strategy_success

    def __init__(self, objective_function, int dim, bounds, int population_size=50, 
                 int max_iter=100, double f=2.0, double l=1.5, double M=0.5, 
                 int adaptation_interval=10):
        """
        Initialize the Self-adaptive Emperor Penguin Optimizer (SA-EPO).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of penguins (solutions).
        - max_iter: Maximum number of iterations.
        - f: Temperature profile parameter (initial value).
        - l: Distance profile parameter (initial value).
        - M: Social forces parameter (initial value).
        - adaptation_interval: Interval for updating strategy probabilities.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.f = f
        self.l = l
        self.M = M
        self.adaptation_interval = adaptation_interval

        self.penguins = None
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []
        
        # Parameter adaptation strategies and their probabilities
        self.strategies = ['linear', 'exponential', 'chaotic']  # Python strings
        self.strategy_probs = np.array([1.0/3, 1.0/3, 1.0/3], dtype=DTYPE)
        self.strategy_success = np.zeros(3, dtype=DTYPE)
        
    cpdef void initialize_penguins(self):
        """ Generate initial penguin positions randomly """
        cdef int i
        self.penguins = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                        (self.population_size, self.dim)).astype(DTYPE)

    cpdef ndarray evaluate_penguins(self):
        """ Compute fitness values for the penguin positions """
        cdef int i
        cdef ndarray fitness = np.empty(self.population_size, dtype=DTYPE)
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.penguins[i])
        return fitness

    cdef (double, double) compute_huddle_boundary(self, int t):
        """ Compute huddle boundary and temperature profile """
        cdef double T = self.max_iter
        cdef double T_prime = 2.0 - (t / T)
        cdef double R = np.random.rand()
        return T_prime, R

    cdef (double, double, double) adapt_parameters(self, str strategy, int t):
        """ Adapt control parameters (f, l, M) based on selected strategy """
        cdef double T = self.max_iter
        cdef double x
        cdef int i
        if strategy == 'linear':
            self.f = 2.0 - (t / T) * 1.5
            self.l = 1.5 - (t / T) * 1.0
            self.M = 0.5 + (t / T) * 0.3
        elif strategy == 'exponential':
            self.f = 2.0 * exp(-t / (T / 2))
            self.l = 1.5 * exp(-t / (T / 3))
            self.M = 0.5 * (1 + tanh(t / (T / 4)))
        elif strategy == 'chaotic':
            x = 0.7
            for i in range(t % 10):
                x = 4 * x * (1 - x)
            self.f = 1.5 + x * 0.5
            self.l = 1.0 + x * 0.5
            self.M = 0.3 + x * 0.4
        return self.f, self.l, self.M

    cpdef void update_strategy_probabilities(self):
        """ Update strategy selection probabilities based on historical success """
        cdef double total_success = np.sum(self.strategy_success) + 1e-10
        self.strategy_probs = self.strategy_success / total_success
        self.strategy_probs = np.clip(self.strategy_probs, 0.1, 0.9)
        self.strategy_probs /= np.sum(self.strategy_probs)
        self.strategy_success *= 0.9

    cdef ndarray huddle_movement(self, int index, int t):
        """ Simulate penguin movement in huddle """
        cdef double T_prime, R
        cdef str strategy
        cdef double f, l, M
        cdef ndarray D, new_solution
        cdef double S  # Explicitly declare S as a scalar
        cdef double new_value, old_value
        cdef int strategy_idx
        cdef ndarray rand_vector

        T_prime, R = self.compute_huddle_boundary(t)
        
        # Select adaptation strategy, ensuring Python string compatibility
        strategy = str(np.random.choice(self.strategies, p=self.strategy_probs))
        f, l, M = self.adapt_parameters(strategy, t)
        
        # Compute distance to best solution
        if self.best_solution is not None:
            D = np.abs(f * np.random.rand() * self.best_solution - self.penguins[index])
        else:
            D = np.abs(f * np.random.rand() * self.penguins[index])
        
        # Compute social forces (scalar)
        S = M * exp(-t / l) - exp(-t)
        
        # Generate random vector for element-wise multiplication
        rand_vector = np.random.rand(self.dim)
        
        # Update position using scalar S with broadcasting
        new_solution = self.penguins[index] + S * D * rand_vector
        
        # Clip to bounds
        new_solution = np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])
        
        # Evaluate new solution
        new_value = self.objective_function(new_solution)
        old_value = self.objective_function(self.penguins[index])
        
        # Track strategy success
        if new_value < old_value:
            strategy_idx = self.strategies.index(strategy)
            self.strategy_success[strategy_idx] += 1
        
        return new_solution

    cpdef tuple optimize(self):
        """ Run the Self-adaptive Emperor Penguin Optimization """
        cdef int generation, i, min_idx
        cdef ndarray fitness
        self.initialize_penguins()
        for generation in range(self.max_iter):
            fitness = self.evaluate_penguins()
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.penguins[min_idx].copy()
                self.best_value = fitness[min_idx]

            # Update penguin positions
            for i in range(self.population_size):
                self.penguins[i] = self.huddle_movement(i, generation)

            # Update strategy probabilities periodically
            if generation % self.adaptation_interval == 0 and generation > 0:
                self.update_strategy_probabilities()

            self.history.append((generation, self.best_solution.copy(), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
