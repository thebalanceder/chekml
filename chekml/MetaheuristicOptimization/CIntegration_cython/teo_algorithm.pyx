# cython: language_level=3
# distutils: language=c++

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp
from libc.stdlib cimport rand, RAND_MAX

# Define types for NumPy arrays
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class ThermalExchangeOptimizer:
    cdef public:
        int dim, population_size, max_iter
        double step_size, initial_temperature, final_temperature, cooling_rate
        double[:, :] bounds
        double[:] current_solution, best_solution
        double current_fitness, best_fitness
        list history
        object objective_func  # Explicitly declare as Python object

    def __init__(self, object objective_func, int dim, bounds, int population_size=50, int max_iter=100,
                 double step_size=0.1, double initial_temperature=100, double final_temperature=0.01,
                 double cooling_rate=0.99):
        """
        Initialize the Thermal Exchange Optimizer (TEO).

        Parameters:
        - objective_func: Function to optimize (Python callable).
        - dim: Number of dimensions (variables).
        - bounds: List of (lower, upper) bounds for each dimension.
        - population_size: Number of solutions.
        - max_iter: Maximum number of iterations.
        - step_size: Step size for solution perturbation.
        - initial_temperature: Initial temperature for annealing process.
        - final_temperature: Final temperature for convergence check.
        - cooling_rate: Cooling rate for temperature reduction.
        """
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter
        self.step_size = step_size
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate
        self.bounds = np.array(bounds, dtype=DTYPE)
        self.current_solution = np.empty(dim, dtype=DTYPE)
        self.best_solution = np.empty(dim, dtype=DTYPE)
        self.current_fitness = 0.0
        self.best_fitness = float("inf")
        self.history = []
        self.objective_func = objective_func

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_solution(self):
        """ Generate initial random solution """
        cdef int i
        cdef double[:] lb = self.bounds[:, 0]
        cdef double[:] ub = self.bounds[:, 1]
        for i in range(self.dim):
            self.current_solution[i] = lb[i] + (ub[i] - lb[i]) * (<double>rand() / RAND_MAX)
        self.current_fitness = self.objective_func(self.current_solution)
        for i in range(self.dim):
            self.best_solution[i] = self.current_solution[i]
        self.best_fitness = self.current_fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void perturb_solution(self, double[:] new_solution):
        """ Generate a new solution by perturbing the current solution """
        cdef int i
        cdef double[:] lb = self.bounds[:, 0]
        cdef double[:] ub = self.bounds[:, 1]
        for i in range(self.dim):
            new_solution[i] = self.current_solution[i] + self.step_size * np.random.randn()
            if new_solution[i] < lb[i]:
                new_solution[i] = lb[i]
            elif new_solution[i] > ub[i]:
                new_solution[i] = ub[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void accept_solution(self, double[:] new_solution, double new_fitness, double temperature):
        """ Determine whether to accept the new solution based on fitness and temperature """
        cdef int i
        if new_fitness < self.current_fitness or (<double>rand() / RAND_MAX) < exp((self.current_fitness - new_fitness) / temperature):
            for i in range(self.dim):
                self.current_solution[i] = new_solution[i]
            self.current_fitness = new_fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_best_solution(self):
        """ Update the best solution if the current solution is better """
        cdef int i
        if self.current_fitness < self.best_fitness:
            for i in range(self.dim):
                self.best_solution[i] = self.current_solution[i]
            self.best_fitness = self.current_fitness

    def optimize(self):
        """ Run the Thermal Exchange Optimization algorithm """
        self.initialize_solution()
        cdef double temperature = self.initial_temperature
        cdef int iteration
        cdef double[:] new_solution = np.empty(self.dim, dtype=DTYPE)
        cdef double new_fitness
        cdef list history = []

        for iteration in range(self.max_iter):
            self.perturb_solution(new_solution)
            new_fitness = self.objective_func(new_solution)
            self.accept_solution(new_solution, new_fitness, temperature)
            self.update_best_solution()
            temperature *= self.cooling_rate
            history.append((iteration, np.array(self.best_solution, copy=True), self.best_fitness))
            print(f"Iteration {iteration + 1}: Best Fitness = {self.best_fitness}")
            if temperature < self.final_temperature:
                break

        self.history = history
        return np.array(self.best_solution), self.best_fitness, self.history
