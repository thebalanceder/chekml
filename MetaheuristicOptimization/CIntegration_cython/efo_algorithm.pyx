# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt

# Define numpy types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class ElectromagneticFieldOptimizer:
    cdef:
        object objective_function
        int dim
        np.ndarray bounds
        int population_size
        int max_iter
        double randomization_rate
        double positive_selection_rate
        double positive_field_ratio
        double negative_field_ratio
        double golden_ratio
        np.ndarray em_population
        np.ndarray best_solution
        double best_value
        list history
        np.ndarray r_index1
        np.ndarray r_index2
        np.ndarray r_index3
        np.ndarray ps
        np.ndarray r_force
        np.ndarray rp
        np.ndarray randomization

    def __init__(self, objective_function, int dim, bounds, int population_size=50, int max_iter=100,
                 double randomization_rate=0.3, double positive_selection_rate=0.2,
                 double positive_field_ratio=0.1, double negative_field_ratio=0.45):
        """
        Initialize the Electromagnetic Field Optimization (EFO) algorithm.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)
        self.population_size = population_size
        self.max_iter = max_iter
        self.randomization_rate = randomization_rate
        self.positive_selection_rate = positive_selection_rate
        self.positive_field_ratio = positive_field_ratio
        self.negative_field_ratio = negative_field_ratio
        self.golden_ratio = (1.0 + sqrt(5.0)) / 2.0
        self.best_value = np.inf

        # Precompute random indices and values
        self.r_index1 = np.random.randint(0, int(self.population_size * self.positive_field_ratio),
                                         (self.dim, self.max_iter), dtype=np.int32)
        self.r_index2 = np.random.randint(int(self.population_size * (1 - self.negative_field_ratio)),
                                         self.population_size, (self.dim, self.max_iter), dtype=np.int32)
        self.r_index3 = np.random.randint(int(self.population_size * self.positive_field_ratio),
                                         int(self.population_size * (1 - self.negative_field_ratio)),
                                         (self.dim, self.max_iter), dtype=np.int32)
        self.ps = np.random.rand(self.dim, self.max_iter).astype(DTYPE)
        self.r_force = np.random.rand(self.max_iter).astype(DTYPE)
        self.rp = np.random.rand(self.max_iter).astype(DTYPE)
        self.randomization = np.random.rand(self.max_iter).astype(DTYPE)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_population(self):
        """Generate initial electromagnetic particles randomly"""
        self.em_population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                              (self.population_size, self.dim)).astype(DTYPE)
        self.evaluate_and_sort_population()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void evaluate_and_sort_population(self):
        """Compute fitness values and sort population by fitness"""
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = np.zeros(self.population_size, dtype=DTYPE)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.em_population[i])
        
        # Sort population by fitness
        cdef np.ndarray[long] sorted_indices = np.argsort(fitness)
        self.em_population = self.em_population[sorted_indices]
        fitness = fitness[sorted_indices]
        
        # Update best solution
        if fitness[0] < self.best_value:
            self.best_solution = self.em_population[0].copy()
            self.best_value = fitness[0]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef DTYPE_t[::1] generate_new_particle(self, int generation):
        """Generate a new particle based on EFO rules"""
        cdef DTYPE_t[::1] new_particle = np.zeros(self.dim, dtype=DTYPE)
        cdef double r = self.r_force[generation]
        cdef int i, ri
        cdef double value

        for i in range(self.dim):
            if self.ps[i, generation] > self.positive_selection_rate:
                new_particle[i] = (self.em_population[self.r_index3[i, generation], i] +
                                  self.golden_ratio * r * (self.em_population[self.r_index1[i, generation], i] -
                                                          self.em_population[self.r_index3[i, generation], i]) +
                                  r * (self.em_population[self.r_index3[i, generation], i] -
                                       self.em_population[self.r_index2[i, generation], i]))
            else:
                new_particle[i] = self.em_population[self.r_index1[i, generation], i]

            # Check boundaries
            if new_particle[i] < self.bounds[i, 0] or new_particle[i] > self.bounds[i, 1]:
                new_particle[i] = self.bounds[i, 0] + (self.bounds[i, 1] - self.bounds[i, 0]) * \
                                  self.randomization[generation]

        # Randomize one dimension with probability randomization_rate
        if self.rp[generation] < self.randomization_rate:
            ri = np.random.randint(0, self.dim)
            new_particle[ri] = self.bounds[ri, 0] + (self.bounds[ri, 1] - self.bounds[ri, 0]) * \
                               self.randomization[generation]

        return new_particle

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void insert_particle(self, DTYPE_t[::1] new_particle):
        """Insert new particle into population if it improves fitness"""
        cdef double new_fitness = self.objective_function(np.asarray(new_particle))
        cdef int last_idx = self.population_size - 1
        cdef double worst_fitness = self.objective_function(np.asarray(self.em_population[last_idx]))
        
        if new_fitness < worst_fitness:
            # Find insertion position
            fitness = np.zeros(self.population_size, dtype=DTYPE)
            for i in range(self.population_size):
                fitness[i] = self.objective_function(np.asarray(self.em_population[i]))
            
            insert_pos = np.searchsorted(fitness, new_fitness)
            # Shift population and insert new particle
            self.em_population = np.vstack((self.em_population[:insert_pos],
                                            np.asarray(new_particle),
                                            self.em_population[insert_pos:last_idx]))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """Run the Electromagnetic Field Optimization algorithm"""
        self.initialize_population()
        cdef int generation = 0
        self.history = []
        cdef DTYPE_t[::1] new_particle

        while generation < self.max_iter:
            # Generate and evaluate new particle
            new_particle = self.generate_new_particle(generation)
            self.insert_particle(new_particle)
            self.evaluate_and_sort_population()

            self.history.append((generation, self.best_solution.copy(), self.best_value))
            if generation % 1000 == 0:
                print(f"Iteration {generation + 1}: Best Value = {self.best_value}")

            generation += 1

        return self.best_solution, self.best_value, self.history
