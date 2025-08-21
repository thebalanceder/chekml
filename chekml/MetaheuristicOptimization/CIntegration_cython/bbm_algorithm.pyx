# bumble_bee_optimizer.pyx

import numpy as np
cimport numpy as np
from libc.math cimport pow, fabs
from cython cimport boundscheck, wraparound

@boundscheck(False)
@wraparound(False)
cdef class BumbleBeeMatingOptimizer:
    cdef public object objective_function
    cdef int dim, population_size, max_iter
    cdef double queen_factor, drone_selection, worker_improvement
    cdef double brood_distribution, mating_resistance, replacement_ratio
    cdef object bounds  # store as generic object
    cdef object bees    # store as generic object
    cdef object queen   # store as generic object
    cdef double queen_value
    cdef list history

    def __init__(self, objective_function, int dim, bounds,
                 int population_size=50, int max_iter=100,
                 double queen_factor=0.3, double drone_selection=0.2,
                 double worker_improvement=1.35, double brood_distribution=0.46,
                 double mating_resistance=1.2, double replacement_ratio=0.23):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.float64)
        self.population_size = population_size
        self.max_iter = max_iter
        self.queen_factor = queen_factor
        self.drone_selection = drone_selection
        self.worker_improvement = worker_improvement
        self.brood_distribution = brood_distribution
        self.mating_resistance = mating_resistance
        self.replacement_ratio = replacement_ratio
        self.queen_value = float("inf")
        self.history = []

    cpdef void initialize_bees(self):
        self.bees = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                      (self.population_size, self.dim)).astype(np.float64)

    cpdef object evaluate_bees(self):
        cdef np.ndarray[np.float64_t, ndim=1] fitness = np.zeros(self.population_size, dtype=np.float64)
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.bees[i])
        return fitness

    cpdef void queen_selection_phase(self):
        cdef object fitness = self.evaluate_bees()
        cdef int min_idx = int(np.argmin(fitness))
        if fitness[min_idx] < self.queen_value:
            self.queen = self.bees[min_idx].copy()
            self.queen_value = fitness[min_idx]

    cpdef object blend_alpha_crossover(self, queen, drone, double alpha=0.5):
        cdef object lower = np.minimum(queen, drone) - alpha * np.abs(queen - drone)
        cdef object upper = np.maximum(queen, drone) + alpha * np.abs(queen - drone)
        return np.random.uniform(lower, upper, self.dim).astype(np.float64)

    cpdef object mating_phase(self, int index):
        cdef double r1 = np.random.rand(), r2 = np.random.rand()
        cdef str crossover_type
        cdef object drone = self.bees[np.random.randint(0, self.population_size)]

        if self.dim >= 3:
            crossover_type = str(np.random.choice(np.array(['one_point', 'two_point', 'three_point', 'blend_alpha'], dtype=object)))
        elif self.dim == 2:
            crossover_type = str(np.random.choice(np.array(['one_point', 'blend_alpha'], dtype=object)))
        else:
            crossover_type = 'blend_alpha'

        if r1 < self.drone_selection:
            Vi = (self.queen_factor ** (2.0/3.0)) / self.mating_resistance * r1
        else:
            Vi = (self.queen_factor ** (2.0/3.0)) / self.mating_resistance * r2

        if crossover_type == 'one_point' and self.dim >= 2:
            cut = np.random.randint(1, self.dim)
            new_solution = np.concatenate((self.queen[:cut], drone[cut:]))
        elif crossover_type == 'two_point' and self.dim >= 3:
            cut1, cut2 = np.sort(np.random.choice(self.dim, 2, replace=False))
            new_solution = np.concatenate((self.queen[:cut1], drone[cut1:cut2], self.queen[cut2:]))
        elif crossover_type == 'three_point' and self.dim >= 4:
            cuts = np.sort(np.random.choice(self.dim, 3, replace=False))
            new_solution = np.concatenate((self.queen[:cuts[0]], drone[cuts[0]:cuts[1]],
                                           self.queen[cuts[1]:cuts[2]], drone[cuts[2]:]))
        else:
            new_solution = self.blend_alpha_crossover(self.queen, drone)

        new_solution = self.queen + (new_solution - self.queen) * Vi * np.random.rand(self.dim)
        return np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])

    cpdef object worker_phase(self, int index):
        cdef double r3 = np.random.rand(), r4 = np.random.rand()
        cdef double CFR = 9.435 * np.random.gamma(0.85, 2.5)

        if r3 < self.brood_distribution:
            Vi2 = (self.worker_improvement ** (2.0/3.0)) / (2 * CFR) * r3
        else:
            Vi2 = (self.worker_improvement ** (2.0/3.0)) / (2 * CFR) * r4

        Improve = np.sign(self.queen_value - self.objective_function(self.bees[index])) * \
                  (self.queen - self.bees[index]) * np.random.rand(self.dim)

        new_solution = self.queen + (self.queen - self.bees[index]) * Vi2 + Improve
        return np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])

    cpdef void replacement_phase(self):
        cdef object fitness = self.evaluate_bees()
        cdef object worst_indices = np.argsort(fitness)[-int(self.replacement_ratio * self.population_size):]
        for i in worst_indices:
            self.bees[i] = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)

    cpdef tuple optimize(self):
        self.initialize_bees()
        for generation in range(self.max_iter):
            self.queen_selection_phase()
            for i in range(self.population_size):
                self.bees[i] = self.mating_phase(i)
            for i in range(self.population_size):
                self.bees[i] = self.worker_phase(i)
            self.replacement_phase()
            self.history.append((generation, self.queen.copy(), self.queen_value))
            print(f"Iteration {generation + 1}: Best Value = {self.queen_value}")
        return self.queen, self.queen_value, self.history

