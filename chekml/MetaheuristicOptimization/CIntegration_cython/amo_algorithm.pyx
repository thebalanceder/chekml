# cython: language_level=3
import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport sqrt, cos, sin, fabs

# Declare NumPy types
cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class AnimalMigrationOptimizer:
    cdef:
        object objective_function
        int dim, population_size, max_fes, fes
        cnp.ndarray bounds, population, fitness, global_best_solution
        double global_best_fitness
        list history

    def __init__(self, objective_function, int dim, cnp.ndarray bounds, int population_size=50, int max_fes=150000):
        """
        Initialize the Animal Migration Optimization (AMO) algorithm.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Array of shape (dim, 2) with (lower, upper) bounds.
        - population_size: Number of animals (solutions).
        - max_fes: Maximum number of function evaluations.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.asarray(bounds, dtype=np.double)
        self.population_size = population_size
        self.max_fes = max_fes
        self.fes = 0
        self.global_best_fitness = float("inf")
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_population(self):
        """Initialize the population randomly within bounds."""
        cdef:
            cnp.ndarray[cnp.double_t, ndim=1] lower_bounds = self.bounds[:, 0]
            cnp.ndarray[cnp.double_t, ndim=1] upper_bounds = self.bounds[:, 1]
            cnp.ndarray[cnp.double_t, ndim=2] population
            int i, j
        population = np.empty((self.population_size, self.dim), dtype=np.double)
        for i in range(self.population_size):
            for j in range(self.dim):
                population[i, j] = lower_bounds[j] + (rand() / RAND_MAX) * (upper_bounds[j] - lower_bounds[j])
        self.population = population
        self.fitness = self.evaluate_population()
        self.fes = self.population_size
        self.update_global_best()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.double_t, ndim=1] evaluate_population(self):
        """Compute fitness values for the population."""
        cdef:
            cnp.ndarray[cnp.double_t, ndim=1] fitness = np.empty(self.population_size, dtype=np.double)
            int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.population[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_bounds(self):
        """Ensure population stays within bounds."""
        cdef:
            cnp.ndarray[cnp.double_t, ndim=2] population = self.population
            cnp.ndarray[cnp.double_t, ndim=1] lower_bounds = self.bounds[:, 0]
            cnp.ndarray[cnp.double_t, ndim=1] upper_bounds = self.bounds[:, 1]
            int i, j
        for i in range(self.population_size):
            for j in range(self.dim):
                if population[i, j] < lower_bounds[j] or population[i, j] > upper_bounds[j]:
                    population[i, j] = lower_bounds[j] + (rand() / RAND_MAX) * (upper_bounds[j] - lower_bounds[j])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_global_best(self):
        """Update the global best solution and fitness."""
        cdef:
            cnp.ndarray[cnp.double_t, ndim=1] fitness = self.fitness
            int min_idx = 0
            double min_fitness = fitness[0]
            int i
        for i in range(1, self.population_size):
            if fitness[i] < min_fitness:
                min_fitness = fitness[i]
                min_idx = i
        if min_fitness < self.global_best_fitness:
            self.global_best_fitness = min_fitness
            self.global_best_solution = self.population[min_idx].copy()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef tuple get_indices(self):
        """Generate random indices for each individual, excluding itself."""
        cdef:
            cnp.ndarray[cnp.int32_t, ndim=1] r1 = np.zeros(self.population_size, dtype=np.int32)
            cnp.ndarray[cnp.int32_t, ndim=1] r2 = np.zeros(self.population_size, dtype=np.int32)
            cnp.ndarray[cnp.int32_t, ndim=1] r3 = np.zeros(self.population_size, dtype=np.int32)
            cnp.ndarray[cnp.int32_t, ndim=1] r4 = np.zeros(self.population_size, dtype=np.int32)
            cnp.ndarray[cnp.int32_t, ndim=1] r5 = np.zeros(self.population_size, dtype=np.int32)
            cnp.ndarray[cnp.int32_t, ndim=1] sequence
            int i, temp, seq_len
        for i in range(self.population_size):
            sequence = np.arange(self.population_size, dtype=np.int32)
            sequence = np.delete(sequence, i)
            seq_len = self.population_size - 1

            temp = rand() % seq_len
            r1[i] = sequence[temp]
            sequence = np.delete(sequence, temp)
            seq_len -= 1

            temp = rand() % seq_len
            r2[i] = sequence[temp]
            sequence = np.delete(sequence, temp)
            seq_len -= 1

            temp = rand() % seq_len
            r3[i] = sequence[temp]
            sequence = np.delete(sequence, temp)
            seq_len -= 1

            temp = rand() % seq_len
            r4[i] = sequence[temp]
            sequence = np.delete(sequence, temp)
            seq_len -= 1

            temp = rand() % seq_len
            r5[i] = sequence[temp]
        return r1, r2, r3, r4, r5

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int neighborhood_learning(self, int i):
        """Select exemplar particle based on neighborhood."""
        cdef:
            int lseq[5]
            int j
        if i == 0:
            lseq[0], lseq[1], lseq[2], lseq[3], lseq[4] = self.population_size - 2, self.population_size - 1, i, i + 1, i + 2
        elif i == 1:
            lseq[0], lseq[1], lseq[2], lseq[3], lseq[4] = self.population_size - 1, i - 1, i, i + 1, i + 2
        elif i == self.population_size - 2:
            lseq[0], lseq[1], lseq[2], lseq[3], lseq[4] = i - 2, i - 1, i, self.population_size - 1, 0
        elif i == self.population_size - 1:
            lseq[0], lseq[1], lseq[2], lseq[3], lseq[4] = i - 2, i - 1, i, 0, 1
        else:
            lseq[0], lseq[1], lseq[2], lseq[3], lseq[4] = i - 2, i - 1, i, i + 1, i + 2
        # Random permutation
        for j in range(5):
            temp = lseq[j]
            idx = rand() % 5
            lseq[j] = lseq[idx]
            lseq[idx] = temp
        return lseq[1]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """Run the Animal Migration Optimization algorithm."""
        self.initialize_population()
        cdef:
            int iteration = 0, i, d, exemplar_idx
            cnp.ndarray[cnp.double_t, ndim=2] new_population
            cnp.ndarray[cnp.double_t, ndim=1] new_fitness
            cnp.ndarray[cnp.double_t, ndim=1] probabilities
            cnp.ndarray[cnp.int32_t, ndim=1] sort_indices, r1, r2, r3, r4, r5
            double FF
        while self.fes <= self.max_fes:
            iteration += 1
            self.update_bounds()

            # Neighborhood-based learning phase
            new_population = np.zeros((self.population_size, self.dim), dtype=np.double)
            for i in range(self.population_size):
                FF = np.random.normal(0, 1)
                for d in range(self.dim):
                    exemplar_idx = self.neighborhood_learning(i)
                    new_population[i, d] = self.population[i, d] + FF * (self.population[exemplar_idx, d] - self.population[i, d])

            new_population = self.update_bounds_new(new_population)
            new_fitness = np.zeros(self.population_size, dtype=np.double)
            for i in range(self.population_size):
                new_fitness[i] = self.objective_function(new_population[i])
            self.fes += self.population_size

            for i in range(self.population_size):
                if new_fitness[i] <= self.fitness[i]:
                    self.population[i] = new_population[i]
                    self.fitness[i] = new_fitness[i]

            # Update probabilities based on fitness ranking
            sort_indices = np.argsort(self.fitness).astype(np.int32)
            probabilities = np.zeros(self.population_size, dtype=np.double)
            for i in range(self.population_size):
                probabilities[sort_indices[i]] = (self.population_size - i) / self.population_size

            # Global migration phase
            r1, r2, r3, r4, r5 = self.get_indices()
            new_population = np.zeros((self.population_size, self.dim), dtype=np.double)
            for i in range(self.population_size):
                for d in range(self.dim):
                    if rand() / RAND_MAX > probabilities[i]:
                        new_population[i, d] = (
                            self.population[r1[i], d] +
                            (rand() / RAND_MAX) * (self.global_best_solution[d] - self.population[i, d]) +
                            (rand() / RAND_MAX) * (self.population[r3[i], d] - self.population[i, d])
                        )
                    else:
                        new_population[i, d] = self.population[i, d]

            new_population = self.update_bounds_new(new_population)
            new_fitness = np.zeros(self.population_size, dtype=np.double)
            for i in range(self.population_size):
                new_fitness[i] = self.objective_function(new_population[i])
            self.fes += self.population_size

            for i in range(self.population_size):
                if new_fitness[i] <= self.fitness[i]:
                    self.population[i] = new_population[i]
                    self.fitness[i] = new_fitness[i]

            self.update_global_best()
            self.history.append((iteration, self.global_best_solution.copy(), self.global_best_fitness))
            print(f"Iteration {iteration}: Best Fitness = {self.global_best_fitness}")

        return self.global_best_solution, self.global_best_fitness, self.history

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.double_t, ndim=2] update_bounds_new(self, cnp.ndarray[cnp.double_t, ndim=2] population):
        """Ensure new population stays within bounds."""
        cdef:
            cnp.ndarray[cnp.double_t, ndim=1] lower_bounds = self.bounds[:, 0]
            cnp.ndarray[cnp.double_t, ndim=1] upper_bounds = self.bounds[:, 1]
            int i, j
        for i in range(self.population_size):
            for j in range(self.dim):
                if population[i, j] < lower_bounds[j] or population[i, j] > upper_bounds[j]:
                    population[i, j] = lower_bounds[j] + (rand() / RAND_MAX) * (upper_bounds[j] - lower_bounds[j])
        return population
