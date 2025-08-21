# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport fabs

cdef class HarmonySearch:
    cdef object objective_function
    cdef int dim, memory_size, max_iterations
    cdef double HMCR, PAR, bw
    cdef bint minimize
    cdef np.ndarray lower_bounds, upper_bounds
    cdef np.ndarray harmony_memory
    cdef np.ndarray fitness
    cdef np.ndarray best_solution
    cdef double best_fitness
    cdef list history
    cdef int best_idx
    cdef list bounds

    def __init__(self, objective_function, int dim=2, bounds=None,
                 int memory_size=100, int max_iterations=100,
                 double harmony_memory_considering_rate=0.95,
                 double pitch_adjustment_rate=0.3,
                 double bandwidth=0.2, bint minimize=True):
        """
        Initialize the Harmony Search optimizer.
        """
        self.objective_function = objective_function
        self.dim = dim
        # Ensure bounds is initialized correctly
        if bounds is None:
            self.bounds = [(-10.0, 10.0)] * self.dim
        else:
            self.bounds = bounds

        self.memory_size = memory_size
        self.max_iterations = max_iterations
        self.HMCR = harmony_memory_considering_rate
        self.PAR = pitch_adjustment_rate
        self.bw = bandwidth
        self.minimize = minimize

        self.lower_bounds = np.array([b[0] for b in self.bounds], dtype=np.float64)
        self.upper_bounds = np.array([b[1] for b in self.bounds], dtype=np.float64)

        self.harmony_memory = np.random.uniform(
            self.lower_bounds, self.upper_bounds, (self.memory_size, self.dim)
        )
        self.fitness = np.array(
            [self.objective_function(h) for h in self.harmony_memory], dtype=np.float64
        )

        if self.minimize:
            self.best_idx = np.argmin(self.fitness)
        else:
            self.best_idx = np.argmax(self.fitness)

        self.best_solution = self.harmony_memory[self.best_idx].copy()
        self.best_fitness = self.fitness[self.best_idx]
        self.history = []

    def optimize(self):
        cdef int itr, i
        cdef np.ndarray indices, cm_mask, pa_mask, rand_mask
        cdef np.ndarray harmony, random_harmony, adjusted_harmony, new_harmony
        cdef np.ndarray out_of_bounds
        cdef double new_fitness
        cdef int worst_idx, current_best_idx

        for itr in range(self.max_iterations):
            indices = np.random.randint(0, self.memory_size, self.dim)
            harmony = self.harmony_memory[indices, np.arange(self.dim)]

            cm_mask = np.random.rand(self.dim) < self.HMCR
            pa_mask = (np.random.rand(self.dim) < self.PAR) & cm_mask
            rand_mask = ~cm_mask

            random_harmony = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
            adjusted_harmony = harmony + self.bw * (2 * np.random.rand(self.dim) - 1)

            new_harmony = np.where(cm_mask, harmony, random_harmony)
            new_harmony = np.where(pa_mask, adjusted_harmony, new_harmony)

            out_of_bounds = (new_harmony > self.upper_bounds) | (new_harmony < self.lower_bounds)
            new_harmony[out_of_bounds] = harmony[out_of_bounds]

            new_fitness = self.objective_function(new_harmony)

            if self.minimize:
                worst_idx = np.argmax(self.fitness)
                if new_fitness < self.fitness[worst_idx]:
                    self.harmony_memory[worst_idx] = new_harmony
                    self.fitness[worst_idx] = new_fitness
            else:
                worst_idx = np.argmin(self.fitness)
                if new_fitness > self.fitness[worst_idx]:
                    self.harmony_memory[worst_idx] = new_harmony
                    self.fitness[worst_idx] = new_fitness

            if self.minimize:
                current_best_idx = np.argmin(self.fitness)
            else:
                current_best_idx = np.argmax(self.fitness)

            if self.fitness[current_best_idx] != self.best_fitness:
                self.best_fitness = self.fitness[current_best_idx]
                self.best_solution = self.harmony_memory[current_best_idx].copy()

            self.history.append((itr, self.best_solution.copy()))
            print(f"Iteration {itr + 1}: Best Fitness = {self.best_fitness}")

        return self.best_solution, self.best_fitness, self.history

