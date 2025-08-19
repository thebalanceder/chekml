# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, cos, abs, M_PI

# Define numpy types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class MothFlameOptimizer:
    cdef public:
        object objective_function
        int dim, population_size, max_iter
        np.ndarray bounds, lb, ub, moth_pos, best_flame_pos, convergence_curve
        double best_flame_score

    def __init__(self, object objective_function, int dim, np.ndarray bounds, 
                 int population_size=30, int max_iter=1000):
        """
        Initialize the Moth-Flame Optimization (MFO) algorithm.

        Parameters:
        - objective_function: Function to optimize (minimize).
        - dim: Number of dimensions (variables).
        - bounds: Array of (lower, upper) bounds for each dimension or single pair.
        - population_size: Number of moths (search agents).
        - max_iter: Maximum number of iterations.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter
        self.bounds = np.asarray(bounds, dtype=DTYPE)

        # Handle bounds: single pair or per dimension
        if self.bounds.ndim == 1:
            self.lb = np.full(dim, self.bounds[0], dtype=DTYPE)
            self.ub = np.full(dim, self.bounds[1], dtype=DTYPE)
        else:
            self.lb = self.bounds[:, 0]
            self.ub = self.bounds[:, 1]

        self.moth_pos = None
        self.best_flame_pos = None
        self.best_flame_score = np.inf
        self.convergence_curve = np.zeros(max_iter, dtype=DTYPE)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_moths(self):
        """Initialize the positions of moths randomly within bounds."""
        self.moth_pos = np.random.uniform(self.lb, self.ub, 
                                        (self.population_size, self.dim)).astype(DTYPE)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[DTYPE_t, ndim=1] evaluate_moths(self):
        """Compute fitness values for all moths."""
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = np.empty(self.population_size, dtype=DTYPE)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.moth_pos[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void clip_to_bounds(self):
        """Ensure moth positions stay within bounds."""
        self.moth_pos = np.clip(self.moth_pos, self.lb, self.ub)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple optimize(self):
        """Run the Moth-Flame Optimization algorithm."""
        print("MFO is optimizing your problem")
        self.initialize_moths()
        cdef np.ndarray[DTYPE_t, ndim=2] previous_population = None
        cdef np.ndarray[DTYPE_t, ndim=1] previous_fitness = None
        cdef np.ndarray[DTYPE_t, ndim=2] best_flames = None
        cdef np.ndarray[DTYPE_t, ndim=1] best_flame_fitness = None
        cdef int iteration, i, j, flame_no
        cdef double a, b, t, distance_to_flame
        cdef np.ndarray[DTYPE_t, ndim=1] fitness, fitness_sorted, double_fitness, double_fitness_sorted
        cdef np.ndarray[DTYPE_t, ndim=2] sorted_population, double_population, double_sorted_population
        cdef np.ndarray[np.intp_t, ndim=1] sorted_indices

        for iteration in range(self.max_iter):
            # Number of flames (Eq. 3.14)
            flame_no = int(round(self.population_size - 
                                iteration * ((self.population_size - 1) / self.max_iter)))

            # Clip moth positions to bounds
            self.clip_to_bounds()

            # Calculate fitness
            fitness = self.evaluate_moths()

            if iteration == 0:
                # Sort initial population
                sorted_indices = np.argsort(fitness)
                fitness_sorted = fitness[sorted_indices]
                sorted_population = self.moth_pos[sorted_indices]

                # Initialize flames
                best_flames = sorted_population.copy()
                best_flame_fitness = fitness_sorted.copy()
            else:
                # Combine previous and best flames
                double_population = np.vstack((previous_population, best_flames))
                double_fitness = np.concatenate((previous_fitness, best_flame_fitness))

                # Sort combined population
                sorted_indices = np.argsort(double_fitness)
                double_fitness_sorted = double_fitness[sorted_indices]
                double_sorted_population = double_population[sorted_indices]

                # Update flames
                fitness_sorted = double_fitness_sorted[:self.population_size]
                sorted_population = double_sorted_population[:self.population_size]
                best_flames = sorted_population.copy()
                best_flame_fitness = fitness_sorted.copy()

            # Update best flame
            self.best_flame_score = fitness_sorted[0]
            self.best_flame_pos = sorted_population[0].copy()

            # Store previous population
            previous_population = self.moth_pos.copy()
            previous_fitness = fitness.copy()

            # Linearly decrease 'a' from -1 to -2 (Eq. 3.12)
            a = -1.0 + iteration * (-1.0 / self.max_iter)

            # Update moth positions
            for i in range(self.population_size):
                for j in range(self.dim):
                    if i < flame_no:  # Update w.r.t. corresponding flame
                        distance_to_flame = abs(sorted_population[i, j] - self.moth_pos[i, j])
                        b = 1.0
                        t = (a - 1.0) * np.random.rand() + 1.0
                        self.moth_pos[i, j] = (distance_to_flame * exp(b * t) * 
                                              cos(t * 2.0 * M_PI) + sorted_population[i, j])
                    else:  # Update w.r.t. best flame
                        distance_to_flame = abs(sorted_population[flame_no - 1, j] - 
                                               self.moth_pos[i, j])
                        b = 1.0
                        t = (a - 1.0) * np.random.rand() + 1.0
                        self.moth_pos[i, j] = (distance_to_flame * exp(b * t) * 
                                              cos(t * 2.0 * M_PI) + 
                                              sorted_population[flame_no - 1, j])

            # Store convergence
            self.convergence_curve[iteration] = self.best_flame_score

            # Display progress every 50 iterations
            if iteration % 50 == 0:
                print(f"At iteration {iteration}, the best fitness is {self.best_flame_score}")

        return self.best_flame_pos, self.best_flame_score, self.convergence_curve

