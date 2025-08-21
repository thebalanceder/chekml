# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython

# Define numpy types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class SquirrelSearchAlgorithm:
    cdef public:
        object objective_function
        int dim, population_size, max_iter
        np.ndarray bounds, squirrels, best_solution, convergence_curve
        np.ndarray fitness, velocities, gliding_distances, pulse_flying_rate
        np.ndarray tree_types
        double Fmax, Fmin, Gc, best_value
        int nfs, hnt, ant, noft

    def __init__(self, objective_function, int dim, bounds, int population_size=50, int max_iter=200,
                 double max_gliding_distance=1.11, double min_gliding_distance=0.5, double gliding_constant=1.9,
                 int num_food_sources=4, int hickory_nut_tree=1, int acorn_nut_tree=3, int no_food_trees=46):
        """
        Initialize the Squirrel Search Algorithm (SSA) optimizer.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)
        self.population_size = population_size
        self.max_iter = max_iter
        self.Fmax = max_gliding_distance
        self.Fmin = min_gliding_distance
        self.Gc = gliding_constant
        self.nfs = num_food_sources
        self.hnt = hickory_nut_tree
        self.ant = acorn_nut_tree
        self.noft = no_food_trees

        self.squirrels = None
        self.best_solution = None
        self.best_value = np.inf
        self.convergence_curve = np.zeros(max_iter, dtype=DTYPE)
        self.fitness = np.zeros(population_size, dtype=DTYPE)
        self.velocities = np.zeros((population_size, dim), dtype=DTYPE)
        self.gliding_distances = np.zeros(population_size, dtype=DTYPE)
        self.pulse_flying_rate = np.random.rand(population_size).astype(DTYPE)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def initialize_squirrels(self):
        """ Generate initial squirrel positions randomly """
        cdef np.ndarray[DTYPE_t, ndim=2] squirrels
        cdef np.ndarray[DTYPE_t, ndim=1] lb, ub
        cdef int i

        if self.bounds.ndim == 1:
            squirrels = np.random.rand(self.population_size, self.dim) * \
                        (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        else:
            squirrels = np.zeros((self.population_size, self.dim), dtype=DTYPE)
            lb = self.bounds[:, 0]
            ub = self.bounds[:, 1]
            for i in range(self.dim):
                squirrels[:, i] = np.random.rand(self.population_size) * (ub[i] - lb[i]) + lb[i]

        # Enforce bounds
        squirrels = np.clip(squirrels, self.bounds[:, 0], self.bounds[:, 1])
        self.squirrels = squirrels

        # Evaluate initial fitness
        for i in range(self.population_size):
            self.fitness[i] = self.objective_function(self.squirrels[i, :])

        # Randomly assign tree types: 1 (acorn), 2 (normal), 3 (hickory)
        self.tree_types = np.random.randint(1, 4, size=self.population_size, dtype=np.int32)

        # Find initial best solution
        cdef int min_idx = np.argmin(self.fitness)
        self.best_value = self.fitness[min_idx]
        self.best_solution = self.squirrels[min_idx, :].copy()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[DTYPE_t, ndim=1] enforce_bounds(self, np.ndarray[DTYPE_t, ndim=1] position):
        """ Enforce boundary constraints on a single position """
        return np.clip(position, self.bounds[:, 0], self.bounds[:, 1])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def update_squirrel_position(self, int idx):
        """ Update squirrel position based on tree type """
        cdef double gliding_distance = self.Fmin + (self.Fmax - self.Fmin) * np.random.rand()
        cdef double eps
        cdef np.ndarray[DTYPE_t, ndim=2] A
        self.gliding_distances[idx] = gliding_distance

        # Update velocity based on tree type
        if self.tree_types[idx] == 1:  # Acorn tree
            self.velocities[idx, :] += gliding_distance * self.Gc * \
                                       (self.squirrels[idx, :] - self.best_solution) * 1
        elif self.tree_types[idx] == 2:  # Normal tree
            self.velocities[idx, :] += gliding_distance * self.Gc * \
                                       (self.squirrels[idx, :] - self.best_solution) * 2
        else:  # Hickory tree
            self.velocities[idx, :] += gliding_distance * self.Gc * \
                                       (self.squirrels[idx, :] - self.best_solution) * 3

        # Update position
        self.squirrels[idx, :] += self.velocities[idx, :]

        # Enforce bounds after velocity update
        self.squirrels[idx, :] = self.enforce_bounds(self.squirrels[idx, :])

        # Random flying condition
        if np.random.rand() > self.pulse_flying_rate[idx]:
            eps = -1 + (1 - (-1)) * np.random.rand()
            A = np.random.rand(self.population_size, 1).astype(DTYPE)
            self.squirrels[idx, :] = self.best_solution + eps * np.mean(A)
            # Enforce bounds after random flying
            self.squirrels[idx, :] = self.enforce_bounds(self.squirrels[idx, :])

        # Evaluate new fitness
        cdef double new_fitness = self.objective_function(self.squirrels[idx, :])
        if new_fitness <= self.best_value:
            self.best_solution = self.squirrels[idx, :].copy()
            self.best_value = new_fitness

        return new_fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """ Run the Squirrel Search Algorithm """
        self.initialize_squirrels()
        cdef int iteration, i
        for iteration in range(self.max_iter):
            for i in range(self.population_size):
                self.fitness[i] = self.update_squirrel_position(i)

            # Store convergence curve
            self.convergence_curve[iteration] = self.best_value

            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.convergence_curve
