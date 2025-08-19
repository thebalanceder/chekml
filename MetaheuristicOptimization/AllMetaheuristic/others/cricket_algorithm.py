import numpy as np
import random
import time
import math
from CoefCalculate import CoefCalculate  # Assuming CoefCalculate is available as a Python module

class CricketAlgorithm:
    def __init__(self, fun, dim, bounds, population_size=25, alpha=0.5, max_iter=1000, tol=1e-6):
        """
        Initialize the Cricket Algorithm optimizer.

        Parameters:
        - fun: Objective function to minimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of crickets (solutions).
        - alpha: Scaling factor for randomization.
        - max_iter: Maximum number of iterations.
        - tol: Stopping tolerance for convergence.
        """
        self.fun = fun
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.beta_min = 0.2
        self.Q_min = 0
        self.solutions = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.N_iter = 0

    def initialize_solutions(self):
        """Initialize the population of solutions (crickets)."""
        self.solutions = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                          (self.population_size, self.dim))
        self.fitness = np.array([self.fun(sol) for sol in self.solutions])
        min_idx = np.argmin(self.fitness)
        self.best_solution = self.solutions[min_idx].copy()
        self.best_fitness = self.fitness[min_idx]

    def simple_bounds(self, s):
        """Apply boundary constraints to a solution."""
        s = np.clip(s, self.bounds[:, 0], self.bounds[:, 1])
        return s

    def alpha_new(self, alpha):
        """Update alpha parameter."""
        delta = 0.97
        return delta * alpha

    def optimize(self):
        """Run the Cricket Algorithm optimization."""
        start_time = time.time()
        self.initialize_solutions()
        Q = np.zeros(self.population_size)
        v = np.zeros((self.population_size, self.dim))
        scale = self.bounds[:, 1] - self.bounds[:, 0]
        history = []  # Store history for plotting

        while self.best_fitness > self.tol and self.N_iter < self.max_iter * self.population_size:
            for i in range(self.population_size):
                # Simulate cricket parameters
                N = np.random.randint(0, 121, self.dim)
                T = 0.891797 * N + 40.0252
                T = np.clip(T, 55, 180)
                C = (5 / 9) * (T - 32)
                V = 20.1 * np.sqrt(273 + C)
                V = np.sqrt(V) / 1000
                Z = self.solutions[i] - self.best_solution

                # Calculate frequency
                F = np.zeros(self.dim)
                for j in range(self.dim):
                    if Z[j] != 0:
                        F[j] = V[j] / Z[j]

                # Compute Q[i] as a scalar (use mean of F to reduce to scalar)
                Q[i] = self.Q_min + np.mean(F - self.Q_min) * np.random.rand()
                v[i] = v[i] + (self.solutions[i] - self.best_solution) * Q[i] + V
                S = self.solutions[i] + v[i]

                # Calculate gamma using CoefCalculate
                SumF = np.mean(F) + 10000  # Ensure scalar
                SumT = np.mean(C)          # Ensure scalar
                gamma = CoefCalculate(SumF, SumT)

                # Update solution based on fitness comparison
                M = np.zeros(self.dim)
                for j in range(self.population_size):
                    if self.fitness[i] < self.fitness[j]:
                        distance = np.sqrt(np.sum((self.solutions[i] - self.solutions[j]) ** 2))
                        PS = self.fitness[i] * (4 * math.pi * (distance ** 2))
                        Lp = PS + 10 * np.log10(1 / (4 * math.pi * (distance ** 2)))
                        Aatm = (7.4 * (np.mean(F) ** 2 * distance) / (50 * (10 ** (-8))))
                        RLP = Lp - Aatm
                        K = RLP * np.exp(-gamma * distance ** 2)
                        beta = K + self.beta_min
                        tmpf = self.alpha * (np.random.rand(self.dim) - 0.5) * scale
                        M = self.solutions[i] * (1 - beta) + self.solutions[j] * beta + tmpf
                    else:
                        M = self.best_solution + 0.01 * np.random.randn(self.dim)

                # Select new solution
                new_solution = S if np.random.rand() > gamma else M
                new_solution = self.simple_bounds(new_solution)

                # Evaluate new solution
                Fnew = self.fun(new_solution)
                self.N_iter += 1

                if Fnew <= self.fitness[i]:
                    self.solutions[i] = new_solution
                    self.fitness[i] = Fnew

                    if Fnew <= self.best_fitness:
                        self.best_solution = new_solution
                        self.best_fitness = Fnew

                self.alpha = self.alpha_new(self.alpha)

            # Store history for this iteration
            history.append((self.N_iter // self.population_size, self.best_solution.copy(), self.best_fitness))
            print(f"Iteration {self.N_iter // self.population_size + 1}: Best Fitness = {self.best_fitness}")

        elapsed_time = time.time() - start_time
        print(f"Number of evaluations: {self.N_iter}")
        print(f"Best solution: {self.best_solution}")
        print(f"Best fitness: {self.best_fitness}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        return self.best_solution, self.best_fitness, history
