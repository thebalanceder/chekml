import numpy as np

class DifferentialEvolution:
    def __init__(self, objective_function, bounds, pop_size=50, F=0.5, CR=0.9,
                 max_generations=1000, strategy="rand/1/bin",
                 constraint_function=None, quantize=False, penalty_factor=1e6):
        self.f = objective_function
        self.bounds = np.array(bounds)
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.max_generations = max_generations
        self.strategy = strategy
        self.constraint_function = constraint_function
        self.penalty_factor = penalty_factor
        self.quantize = quantize
        self.dim = len(bounds)
        self.history = []

    def _initialize_population(self):
        min_b, max_b = self.bounds[:, 0], self.bounds[:, 1]
        pop = np.random.rand(self.pop_size, self.dim) * (max_b - min_b) + min_b
        if self.quantize:
            pop = np.round(pop)
        return pop

    def _evaluate(self, x):
        penalty = 0.0
        if self.constraint_function:
            penalty = self.penalty_factor * self.constraint_function(x)
        return self.f(x) + penalty

    def _mutation(self, pop, best, idx):
        ids = list(range(self.pop_size))
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

    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def optimize(self):
        pop = self._initialize_population()
        fitness = np.array([self._evaluate(ind) for ind in pop])

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

