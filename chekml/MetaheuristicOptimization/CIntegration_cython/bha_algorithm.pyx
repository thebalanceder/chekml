import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt

# Define NumPy types for Cython
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class BlackHoleOptimizer:
    cdef public:
        object objective_function
        int dim
        np.ndarray bounds
        int population_size
        int max_iter
        np.ndarray stars
        np.ndarray black_hole
        int black_hole_idx
        double best_value
        list fitness_history
        int num_evaluations

    def __init__(self, object objective_function, int dim, np.ndarray bounds, int population_size=50, int max_iter=100):
        """
        Initialize the Black Hole Algorithm optimizer.

        Parameters:
        - objective_function: Function to optimize (minimize).
        - dim: Number of dimensions (variables).
        - bounds: Array of (lower, upper) bounds for each dimension.
        - population_size: Number of stars (solutions).
        - max_iter: Maximum number of iterations.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.asarray(bounds, dtype=DTYPE)
        self.population_size = population_size
        self.max_iter = max_iter
        self.stars = None
        self.black_hole = None
        self.black_hole_idx = -1
        self.best_value = np.inf
        self.fitness_history = []
        self.num_evaluations = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void initialize_stars(self):
        """Generate initial star positions randomly within bounds."""
        cdef np.ndarray[DTYPE_t, ndim=2] stars = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], (self.population_size, self.dim)).astype(DTYPE)
        self.stars = stars
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = self.evaluate_stars()
        self.num_evaluations += self.population_size
        self.black_hole_idx = np.argmin(fitness)
        self.black_hole = stars[self.black_hole_idx].copy()
        self.best_value = fitness[self.black_hole_idx]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[DTYPE_t, ndim=1] evaluate_stars(self):
        """Compute fitness values for all stars."""
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = np.empty(self.population_size, dtype=DTYPE)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.stars[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void bound_stars(self):
        """Ensure stars stay within bounds."""
        cdef np.ndarray[DTYPE_t, ndim=2] stars = self.stars
        cdef np.ndarray[DTYPE_t, ndim=1] lb = self.bounds[:, 0]
        cdef np.ndarray[DTYPE_t, ndim=1] ub = self.bounds[:, 1]
        cdef int i, j
        for i in range(self.population_size):
            for j in range(self.dim):
                if stars[i, j] > ub[j]:
                    stars[i, j] = ub[j]
                elif stars[i, j] < lb[j]:
                    stars[i, j] = lb[j]
        self.stars = stars

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update_star_positions(self):
        """Update star positions based on black hole attraction."""
        cdef np.ndarray[DTYPE_t, ndim=2] stars = self.stars
        cdef np.ndarray[DTYPE_t, ndim=1] black_hole = self.black_hole
        cdef int i, j
        cdef double landa
        cdef np.ndarray[DTYPE_t, ndim=1] rand_vec = np.random.rand(self.dim).astype(DTYPE)
        for i in range(self.population_size):
            if i != self.black_hole_idx:
                landa = np.random.rand()
                for j in range(self.dim):
                    stars[i, j] += rand_vec[j] * (black_hole[j] - stars[i, j])
        self.stars = stars

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void replace_with_better_black_hole(self):
        """Replace black hole if a star has better fitness."""
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = self.evaluate_stars()
        self.num_evaluations += self.population_size
        cdef int min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.best_value:
            self.black_hole_idx = min_idx
            self.black_hole = self.stars[min_idx].copy()
            self.best_value = fitness[min_idx]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void new_star_generation(self):
        """Replace stars that cross the event horizon with new random stars."""
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = self.evaluate_stars()
        self.num_evaluations += self.population_size
        cdef double R = fitness[self.black_hole_idx] / np.sum(fitness)
        cdef np.ndarray[DTYPE_t, ndim=1] distances = np.empty(self.population_size, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2] stars = self.stars
        cdef np.ndarray[DTYPE_t, ndim=1] black_hole = self.black_hole
        cdef np.ndarray[DTYPE_t, ndim=1] lb = self.bounds[:, 0]
        cdef np.ndarray[DTYPE_t, ndim=1] ub = self.bounds[:, 1]
        cdef int i, j
        cdef double dist
        for i in range(self.population_size):
            dist = 0.0
            for j in range(self.dim):
                dist += (black_hole[j] - stars[i, j]) ** 2
            distances[i] = sqrt(dist)
        for i in range(self.population_size):
            if distances[i] < R and i != self.black_hole_idx:
                stars[i] = np.random.uniform(lb, ub, self.dim).astype(DTYPE)
                fitness[i] = self.objective_function(stars[i])
                self.num_evaluations += 1
        self.stars = stars

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple optimize(self, bint display_flag=True):
        """Run the Black Hole Algorithm optimization."""
        self.initialize_stars()
        cdef int iteration
        for iteration in range(self.max_iter):
            self.update_star_positions()
            self.bound_stars()
            self.replace_with_better_black_hole()
            self.new_star_generation()
            self.bound_stars()
            self.fitness_history.append(self.best_value)
            if display_flag:
                print(f"Iteration {iteration + 1}: Best Fitness = {self.best_value}")
        return self.black_hole, self.best_value, self.fitness_history, self.num_evaluations

@cython.boundscheck(False)
@cython.wraparound(False)
def run_multiple_trials(object objective_function, int dim, np.ndarray bounds, 
                       int population_size=50, int max_iter=500, int runs=10, 
                       bint display_flag=True):
    """Run the Black Hole Algorithm multiple times and collect statistics."""
    cdef np.ndarray[DTYPE_t, ndim=2] best_positions = np.zeros((runs, dim), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] best_fitnesses = np.zeros(runs, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] fitness_evolutions = np.zeros((runs, max_iter), dtype=DTYPE)
    cdef int run
    cdef BlackHoleOptimizer optimizer
    cdef np.ndarray best_x
    cdef double best_fitness
    cdef list fitness_history
    for run in range(runs):
        optimizer = BlackHoleOptimizer(objective_function, dim, bounds, population_size, max_iter)
        best_x, best_fitness, fitness_history, _ = optimizer.optimize(display_flag=display_flag)
        best_positions[run] = best_x
        best_fitnesses[run] = best_fitness
        fitness_evolutions[run] = np.array(fitness_history, dtype=DTYPE)
    
    # Calculate statistics
    cdef double min_fitness = np.min(best_fitnesses)
    cdef double mean_fitness = np.mean(best_fitnesses)
    cdef double median_fitness = np.median(best_fitnesses)
    cdef double max_fitness = np.max(best_fitnesses)
    cdef double std_fitness = np.std(best_fitnesses)
    
    print(f"MIN={min_fitness:.6f}  MEAN={mean_fitness:.6f}  MEDIAN={median_fitness:.6f} "
          f"MAX={max_fitness:.6f}  SD={std_fitness:.6f}")
    
    return best_positions, best_fitnesses, fitness_evolutions
