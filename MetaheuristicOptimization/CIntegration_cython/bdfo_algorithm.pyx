# cython: language_level=3
import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time

# Set random seed for reproducibility
srand(<unsigned int>time(NULL))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double rand_uniform(double low, double high):
    """Generate a random double between low and high."""
    return low + (high - low) * (<double>rand() / <double>RAND_MAX)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class BottlenoseDolphinForagingOptimizer:
    cdef public:
        object objective_function
        int dim
        cnp.ndarray bounds
        int population_size
        int max_iter
        double exploration_factor
        double adjustment_rate
        double elimination_ratio
        cnp.ndarray population
        cnp.ndarray best_solution
        double best_value
        list history

    def __init__(self, objective_function, int dim, bounds, int population_size=50, 
                 int max_iter=100, double exploration_factor=0.5, double adjustment_rate=0.3, 
                 double elimination_ratio=0.2):
        """
        Initialize the Bottlenose Dolphin Foraging Optimizer for continuous optimization.

        Parameters:
        - objective_function: Function to optimize (minimize).
        - dim: Number of dimensions (variables).
        - bounds: List of (lower, upper) bounds for each dimension.
        - population_size: Number of solutions in the population.
        - max_iter: Maximum number of iterations.
        - exploration_factor: Controls the magnitude of random exploration.
        - adjustment_rate: Controls the step size for local adjustments.
        - elimination_ratio: Percentage of worst solutions replaced per iteration.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.float64)
        self.population_size = population_size
        self.max_iter = max_iter
        self.exploration_factor = exploration_factor
        self.adjustment_rate = adjustment_rate
        self.elimination_ratio = elimination_ratio
        self.best_value = float("inf")
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_population(self):
        """Generate initial population of random solutions."""
        cdef cnp.ndarray[cnp.float64_t, ndim=2] pop = np.empty((self.population_size, self.dim), dtype=np.float64)
        cdef int i, j
        cdef double low, high
        for i in range(self.population_size):
            for j in range(self.dim):
                low = self.bounds[j, 0]
                high = self.bounds[j, 1]
                pop[i, j] = rand_uniform(low, high)
        self.population = pop

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] evaluate_population(self):
        """Compute objective values for all solutions in the population."""
        cdef cnp.ndarray[cnp.float64_t, ndim=1] fitness = np.empty(self.population_size, dtype=np.float64)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.population[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] bottom_grubbing_phase(self, int index):
        """
        Perform bottom grubbing-inspired adjustment on a solution.
        Splits the solution into two segments, evaluates their impact, and adjusts toward better regions.
        """
        cdef cnp.ndarray[cnp.float64_t, ndim=1] solution = self.population[index].copy()
        cdef double current_fitness = self.objective_function(solution)
        cdef int half_dim = self.dim // 2
        cdef cnp.ndarray[cnp.float64_t, ndim=1] perturbed1 = solution.copy()
        cdef cnp.ndarray[cnp.float64_t, ndim=1] perturbed2 = solution.copy()
        cdef cnp.ndarray[cnp.float64_t, ndim=1] perturbation = np.empty(self.dim, dtype=np.float64)
        cdef int j
        cdef double fitness1, fitness2

        # Generate perturbation
        for j in range(self.dim):
            perturbation[j] = rand_uniform(-0.1, 0.1)

        # Perturb first segment
        for j in range(half_dim):
            perturbed1[j] += perturbation[j]
            if perturbed1[j] < self.bounds[j, 0]:
                perturbed1[j] = self.bounds[j, 0]
            elif perturbed1[j] > self.bounds[j, 1]:
                perturbed1[j] = self.bounds[j, 1]
        fitness1 = self.objective_function(perturbed1)

        # Perturb second segment
        for j in range(half_dim, self.dim):
            perturbed2[j] += perturbation[j]
            if perturbed2[j] < self.bounds[j, 0]:
                perturbed2[j] = self.bounds[j, 0]
            elif perturbed2[j] > self.bounds[j, 1]:
                perturbed2[j] = self.bounds[j, 1]
        fitness2 = self.objective_function(perturbed2)

        # Adjust solution
        if self.best_solution is not None:
            for j in range(half_dim):
                if fitness1 < current_fitness:
                    solution[j] += self.adjustment_rate * (self.best_solution[j] - solution[j])
            for j in range(half_dim, self.dim):
                if fitness2 < current_fitness:
                    solution[j] += self.adjustment_rate * (self.best_solution[j] - solution[j])
        else:
            j = <int>rand_uniform(0, self.population_size)
            for k in range(half_dim):
                solution[k] += self.adjustment_rate * (self.population[j, k] - solution[k])
            for k in range(half_dim, self.dim):
                solution[k] += self.adjustment_rate * (self.population[j, k] - solution[k])

        # Clip to bounds
        for j in range(self.dim):
            if solution[j] < self.bounds[j, 0]:
                solution[j] = self.bounds[j, 0]
            elif solution[j] > self.bounds[j, 1]:
                solution[j] = self.bounds[j, 1]

        return solution

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] exploration_phase(self, int index):
        """Simulate exploration by introducing random perturbations."""
        cdef cnp.ndarray[cnp.float64_t, ndim=1] solution = self.population[index].copy()
        cdef cnp.ndarray[cnp.float64_t, ndim=1] perturbation = np.empty(self.dim, dtype=np.float64)
        cdef int j
        for j in range(self.dim):
            perturbation[j] = self.exploration_factor * rand_uniform(-1, 1) * (self.bounds[j, 1] - self.bounds[j, 0])
            solution[j] += perturbation[j]
            if solution[j] < self.bounds[j, 0]:
                solution[j] = self.bounds[j, 0]
            elif solution[j] > self.bounds[j, 1]:
                solution[j] = self.bounds[j, 1]
        return solution

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void elimination_phase(self):
        """Replace worst solutions with new random ones."""
        cdef cnp.ndarray[cnp.float64_t, ndim=1] fitness = self.evaluate_population()
        cdef cnp.ndarray[cnp.int64_t, ndim=1] indices = np.argsort(fitness)[-int(self.elimination_ratio * self.population_size):]
        cdef int i, j
        cdef double low, high
        for i in indices:
            for j in range(self.dim):
                low = self.bounds[j, 0]
                high = self.bounds[j, 1]
                self.population[i, j] = rand_uniform(low, high)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """Run the Bottlenose Dolphin Foraging Optimization."""
        self.initialize_population()
        cdef cnp.ndarray[cnp.float64_t, ndim=1] fitness
        cdef int min_idx, i, iteration
        cdef double r
        for iteration in range(self.max_iter):
            fitness = self.evaluate_population()
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.population[min_idx].copy()
                self.best_value = fitness[min_idx]

            # Bottom grubbing phase
            for i in range(self.population_size):
                self.population[i] = self.bottom_grubbing_phase(i)

            # Exploration phase
            for i in range(self.population_size):
                r = rand_uniform(0, 1)
                if r < 0.3:  # Apply exploration probabilistically
                    self.population[i] = self.exploration_phase(i)

            # Elimination phase
            self.elimination_phase()

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
