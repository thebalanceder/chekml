import numpy as np
import math

class EagleStrategyOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=300, max_iter=1000, 
                 c1=2.0, c2=2.0, w_max=0.9, w_min=0.4):
        self.obj_func = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)
        self.population_size = population_size
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min

        self.population = None
        self.velocity = None
        self.local_best = None
        self.local_best_cost = None
        self.global_best = None
        self.global_best_cost = float("inf")

        self.history = []

    def initialize(self):
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocity = np.random.rand(self.population_size, self.dim)

        self.local_best = self.population.copy()
        self.local_best_cost = np.array([self.obj_func(ind) for ind in self.population])

        best_idx = np.argmin(self.local_best_cost)
        self.global_best = self.local_best[best_idx]
        self.global_best_cost = self.local_best_cost[best_idx]

    def update_velocity_position(self, iter_):
        r1 = np.random.rand(self.population_size, self.dim)
        r2 = np.random.rand(self.population_size, self.dim)
        w = self.w_max - ((self.w_max - self.w_min) * iter_ / self.max_iter)

        cognitive = self.c1 * r1 * (self.local_best - self.population)
        social = self.c2 * r2 * (self.global_best - self.population)
        self.velocity = w * self.velocity + cognitive + social
        self.population += self.velocity

        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        self.population = np.clip(self.population, lb, ub)

    def levy_flight(self):
        beta = 1.5
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)

        steps = np.zeros_like(self.population)
        for i in range(self.population_size):
            u = np.random.randn(self.dim) * sigma
            v = np.random.rand(self.dim)
            step = u / np.power(np.abs(v), 1 / beta)
            steps[i] = 0.1 * step * (self.local_best[i] - self.global_best)

        s = self.local_best + steps

        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        s = np.clip(s, lb, ub)

        return s

    def optimize(self):
        self.initialize()
        for iter_ in range(self.max_iter):
            if np.random.rand() < 0.2:
                s = self.levy_flight()
                s_cost = np.array([self.obj_func(ind) for ind in s])
                new_best_idx = np.argmin(s_cost)
                if s_cost[new_best_idx] < self.global_best_cost:
                    self.global_best = s[new_best_idx]
                    self.global_best_cost = s_cost[new_best_idx]
                continue

            self.update_velocity_position(iter_)
            fitness = np.array([self.obj_func(ind) for ind in self.population])

            for i in range(self.population_size):
                if fitness[i] < self.local_best_cost[i]:
                    self.local_best[i] = self.population[i]
                    self.local_best_cost[i] = fitness[i]

            best_idx = np.argmin(self.local_best_cost)
            if self.local_best_cost[best_idx] < self.global_best_cost:
                self.global_best = self.local_best[best_idx]
                self.global_best_cost = self.local_best_cost[best_idx]

            self.history.append((iter_, self.global_best.copy(), self.global_best_cost))
            print(f"Iteration {iter_ + 1}: Best Cost = {self.global_best_cost:.6f}")

        return self.global_best, self.global_best_cost, self.history

