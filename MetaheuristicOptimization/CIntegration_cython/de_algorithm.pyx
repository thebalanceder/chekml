import numpy as np
cimport numpy as np
from libc.math cimport fabs, sin, sqrt, M_PI

cdef class DifferentialEvolution:
    cdef:
        object f
        np.ndarray bounds
        int pop_size, max_generations, dim
        double F, CR, penalty_factor
        str strategy
        object constraint_function
        bint quantize
        list history

    def __init__(self, objective_function, bounds, int pop_size=50, double F=0.5, double CR=0.9,
                 int max_generations=1000, strategy="rand/1/bin",
                 constraint_function=None, bint quantize=False, double penalty_factor=1e6):
        self.f = objective_function
        self.bounds = np.array(bounds, dtype=np.float64)
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.max_generations = max_generations
        self.strategy = strategy
        self.constraint_function = constraint_function
        self.penalty_factor = penalty_factor
        self.quantize = quantize
        self.dim = self.bounds.shape[0]
        self.history = []

    cdef np.ndarray _initialize_population(self):
        cdef np.ndarray min_b = self.bounds[:, 0]
        cdef np.ndarray max_b = self.bounds[:, 1]
        cdef np.ndarray pop = np.random.rand(self.pop_size, self.dim) * (max_b - min_b) + min_b
        if self.quantize:
            pop = np.round(pop)
        return pop

    cdef double _evaluate(self, np.ndarray x):
        cdef double penalty = 0.0
        if self.constraint_function:
            penalty = self.penalty_factor * self.constraint_function(x)
        return self.f(x) + penalty

    cdef np.ndarray _mutation(self, np.ndarray pop, np.ndarray best, int idx):
        cdef list ids = list(range(self.pop_size))
        ids.remove(idx)
        r1, r2, r3 = pop[np.random.choice(ids, 3, replace=False)]

        if self.strategy == "rand/1/bin":
            return r1 + self.F * (r2 - r3)
        elif self.strategy == "best/1/bin":
            return best + self.F * (r1 - r2)
        elif self.strategy == "rand-to-best/1":
            rand = pop[np.random.choice(ids)]
            return rand + 0.5 * (best - rand) + self.F * (r1 - r2)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    cdef np.ndarray _crossover(self, np.ndarray target, np.ndarray mutant):
        cdef np.ndarray cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        return np.where(cross_points, mutant, target)

    cpdef tuple optimize(self):
        cdef np.ndarray pop = self._initialize_population()
        cdef np.ndarray fitness = np.array([self._evaluate(ind) for ind in pop])
        cdef np.ndarray mutant, trial
        cdef int best_idx, i

        for gen in range(self.max_generations):
            best_idx = np.argmin(fitness)
            best = pop[best_idx]
            new_pop = np.empty_like(pop)

            for i in range(self.pop_size):
                mutant = self._mutation(pop, best, i)
                mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])
                trial = self._crossover(pop[i], mutant)
                if self.quantize:
                    trial = np.round(trial)

                trial_fit = self._evaluate(trial)
                if trial_fit < fitness[i]:
                    new_pop[i] = trial
                    fitness[i] = trial_fit
                else:
                    new_pop[i] = pop[i]

            pop = new_pop
            self.history.append((gen, best.copy(), fitness[best_idx]))
            print(f"Generation {gen+1}: Best Fitness = {fitness[best_idx]:.6f}")

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx], self.history

