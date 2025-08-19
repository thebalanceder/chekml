# cython: language_level=3
import numpy as np
cimport numpy as np
from libc.math cimport cos, pi, sqrt, fabs, atan

cdef class SpecialForcesOptimizer:
    cdef public int dim, population_size, max_iter
    cdef public double tv1, tv2, p0, k
    cdef public object objective_function
    cdef np.ndarray bounds, members, best_solution
    cdef double best_value
    cdef list history
    def __init__(self, objective_function, int dim, bounds, int population_size=30, int max_iter=200,
                 double tv1=0.5, double tv2=0.3, double p0=0.25, double k=0.4):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.float64)
        self.population_size = population_size
        self.max_iter = max_iter
        self.tv1 = tv1
        self.tv2 = tv2
        self.p0 = p0
        self.k = k

        self.members = None
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_members(self):
        self.members = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                         (self.population_size, self.dim))

    def evaluate_members(self):
        return np.array([self.objective_function(member) for member in self.members])

    cpdef double calculate_instruction(self, int t):
        return (1 - 0.15 * np.random.rand()) * (1 - t / self.max_iter)

    cpdef double calculate_loss_probability(self, int t):
        return self.p0 * cos(pi * t / (2 * self.max_iter))

    cpdef double calculate_raid_coefficient(self, int t):
        return 0.75 - 0.55 * atan((t / self.max_iter) ** (2 * pi))

    cpdef np.ndarray large_scale_search(self, int index, double r1):
        cdef np.ndarray lb, ub, term1, term2, new_position
        if r1 >= 0.5:
            lb = self.bounds[:, 0]
            ub = self.bounds[:, 1]
            term1 = r1 * (self.best_solution - self.members[index])
            term2 = (1 - r1) * (ub - lb) * np.random.choice([-1, 1], size=self.dim)
            new_position = self.members[index] + term1 + term2
            return np.clip(new_position, lb, ub)
        return None

    cpdef np.ndarray raid(self, int index, double r1, double w):
        if r1 < 0.5:
            available_indices = [i for i in range(self.population_size) if i != index]
            if not available_indices:
                return None
            aim_idx = np.random.choice(available_indices)
            X_i = self.members[index]
            X_aim = self.members[aim_idx]
            f_i = self.objective_function(X_i)
            f_aim = self.objective_function(X_aim)

            if f_i + f_aim != 0:
                A_i = (f_i / (f_i + f_aim)) * (X_aim - X_i)
            else:
                A_i = np.zeros(self.dim)

            new_position = X_i + w * A_i
            return np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])
        return None

    cpdef np.ndarray transition_phase(self, int index, double r2, double w, double instruction):
        if r2 >= 0.5:
            A_i = self.raid(index, 0.4, w)
            if A_i is not None:
                return A_i
            return self.members[index]
        else:
            new_position = instruction * (self.best_solution - self.members[index]) + 0.1 * self.members[index]
            return np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])

    cpdef np.ndarray arrest_rescue(self, int t):
        X_ave = np.mean(self.members, axis=0)
        r = np.random.uniform(-1, 1, self.dim)
        new_positions = self.best_solution + r * np.abs(self.best_solution - X_ave)
        return np.clip(new_positions, self.bounds[:, 0], self.bounds[:, 1])

    cpdef np.ndarray unmanned_search(self, int t):
        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]
        c = self.k * (lb + (1 - t / self.max_iter) * (ub - lb))

        v = np.random.randn(self.dim)
        norm = np.sqrt(np.sum(v ** 2))
        if norm != 0:
            v = v / norm * c
        else:
            v = np.zeros(self.dim)

        base_idx = np.random.randint(self.population_size)
        X_u = self.members[base_idx] + v
        return np.clip(X_u, lb, ub)

    def optimize(self):
        self.initialize_members()
        for t in range(self.max_iter):
            fitness = self.evaluate_members()
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.members[min_idx].copy()
                self.best_value = fitness[min_idx]

            X_u = self.unmanned_search(t)
            fitness_u = self.objective_function(X_u)
            if fitness_u < self.best_value:
                self.best_solution = X_u.copy()
                self.best_value = fitness_u

            instruction = self.calculate_instruction(t)
            p = self.calculate_loss_probability(t)
            w = self.calculate_raid_coefficient(t)

            new_members = self.members.copy()
            for i in range(self.population_size):
                if np.random.rand() < p:
                    continue

                r1 = np.random.rand()
                r2 = np.random.rand()

                if instruction >= self.tv1:
                    new_pos = self.large_scale_search(i, r1)
                    if new_pos is None:
                        new_pos = self.raid(i, r1, w)
                    if new_pos is not None:
                        new_members[i] = new_pos
                elif self.tv2 < instruction < self.tv1:
                    new_members[i] = self.transition_phase(i, r2, w, instruction)
                else:
                    new_members[i] = self.arrest_rescue(t)

            self.members = new_members
            self.history.append((t, self.best_solution.copy(), self.best_value))
            print(f"Iteration {t + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

