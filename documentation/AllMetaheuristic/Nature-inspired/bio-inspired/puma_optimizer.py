import numpy as np

class PumaOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=30, max_iter=100, 
                 Q=0.67, beta=2, PF=[0.5, 0.5, 0.3], mega_explore=0.99, mega_exploit=0.99):
        """
        Initialize the Puma Optimizer (PO).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of pumas (solutions).
        - max_iter: Maximum number of iterations.
        - Q: Probability threshold for exploitation phase.
        - beta: Scaling factor for exploitation phase.
        - PF: Weighting factors for F1, F2, and F3 calculations.
        - mega_explore: Exploration adaptation parameter.
        - mega_exploit: Exploitation adaptation parameter.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.Q = Q
        self.beta = beta
        self.PF = PF
        self.mega_explore = mega_explore
        self.mega_exploit = mega_exploit

        self.pumas = None  # Population of pumas (solutions)
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []
        
        # Initialize tracking variables
        self.unselected = np.ones(2)  # Tracks unselected phases: [Exploration, Exploitation]
        self.f3_explore = 0
        self.f3_exploit = 0
        self.seq_time_explore = np.ones(3)
        self.seq_time_exploit = np.ones(3)
        self.seq_cost_explore = np.ones(3)
        self.seq_cost_exploit = np.ones(3)
        self.score_explore = 0
        self.score_exploit = 0
        self.pf_f3 = []
        self.flag_change = 1

    def initialize_pumas(self):
        """ Generate initial puma population randomly """
        self.pumas = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                       (self.population_size, self.dim))

    def evaluate_pumas(self):
        """ Compute fitness values for the puma population """
        return np.array([self.objective_function(puma) for puma in self.pumas])

    def boundary_check(self, X):
        """ Ensure solutions stay within bounds """
        FU = X > self.bounds[:, 1]
        FL = X < self.bounds[:, 0]
        X = X * (~(FU + FL)) + self.bounds[:, 1] * FU + self.bounds[:, 0] * FL
        return X

    def exploration_phase(self):
        """ Simulate exploration phase (global search) """
        pCR = 0.20
        PCR = 1 - pCR
        p = PCR / self.population_size
        new_pumas = np.zeros_like(self.pumas)

        fitness = self.evaluate_pumas()
        sorted_indices = np.argsort(fitness)
        self.pumas = self.pumas[sorted_indices]

        for i in range(self.population_size):
            x = self.pumas[i]
            A = np.random.permutation(self.population_size)
            A = A[A != i]
            a, b, c, d, e, f = A[:6]
            G = 2 * np.random.rand() - 1  # Eq 26

            if np.random.rand() < 0.5:
                y = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)  # Eq 25
            else:
                y = self.pumas[a] + G * (self.pumas[a] - self.pumas[b]) + G * (
                    (self.pumas[a] - self.pumas[b]) - (self.pumas[c] - self.pumas[d]) +
                    (self.pumas[c] - self.pumas[d]) - (self.pumas[e] - self.pumas[f])
                )  # Eq 25

            y = self.boundary_check(y)
            z = np.copy(x)
            j0 = np.random.randint(0, self.dim)
            for j in range(self.dim):
                if j == j0 or np.random.rand() <= pCR:
                    z[j] = y[j]
                else:
                    z[j] = x[j]

            new_pumas[i] = z
            new_cost = self.objective_function(z)
            if new_cost < fitness[i]:
                self.pumas[i] = z
            else:
                pCR += p  # Eq 30

        return self.pumas

    def exploitation_phase(self, iter_count):
        """ Simulate exploitation phase (local search) """
        new_pumas = np.zeros_like(self.pumas)
        for i in range(self.population_size):
            beta1 = 2 * np.random.rand()
            beta2 = np.random.randn(self.dim)
            w = np.random.randn(self.dim)  # Eq 37
            v = np.random.randn(self.dim)  # Eq 38
            F1 = np.random.randn(self.dim) * np.exp(2 - iter_count * (2 / self.max_iter))  # Eq 35
            F2 = w * v**2 * np.cos((2 * np.random.rand()) * w)  # Eq 36
            mbest = np.mean(self.pumas, axis=0) / self.population_size
            R_1 = 2 * np.random.rand() - 1  # Eq 34
            S1 = (2 * np.random.rand() - 1 + np.random.randn(self.dim))
            S2 = (F1 * R_1 * self.pumas[i] + F2 * (1 - R_1) * self.best_solution)
            VEC = S2 / S1

            if np.random.rand() <= 0.5:
                Xatack = VEC
                if np.random.rand() > self.Q:
                    new_pumas[i] = self.best_solution + beta1 * (np.exp(beta2)) * (
                        self.pumas[np.random.randint(self.population_size)] - self.pumas[i]
                    )  # Eq 32
                else:
                    new_pumas[i] = beta1 * Xatack - self.best_solution  # Eq 32
            else:
                r1 = np.random.randint(1, self.population_size)
                sign = (-1) ** np.random.randint(0, 2)
                new_pumas[i] = (mbest * self.pumas[r1] - sign * self.pumas[i]) / (
                    1 + (self.beta * np.random.rand())
                )  # Eq 32

            new_pumas[i] = self.boundary_check(new_pumas[i])
            new_cost = self.objective_function(new_pumas[i])
            if new_cost < self.evaluate_pumas()[i]:
                self.pumas[i] = new_pumas[i]

        return self.pumas

    def optimize(self):
        """ Run the Puma Optimization Algorithm """
        self.history = []
        self.initialize_pumas()
        initial_fitness = self.evaluate_pumas()
        min_idx = np.argmin(initial_fitness)
        self.best_solution = self.pumas[min_idx].copy()
        self.best_value = initial_fitness[min_idx]
        initial_best = self.best_solution.copy()
        initial_best_cost = self.best_value

        # Unexperienced Phase (first 3 iterations)
        costs_explore = []
        costs_exploit = []
        for iter_count in range(3):
            # Exploration Phase
            pumas_explore = self.exploration_phase()
            fitness_explore = self.evaluate_pumas()
            costs_explore.append(np.min(fitness_explore))

            # Exploitation Phase
            pumas_exploit = self.exploitation_phase(iter_count + 1)
            fitness_exploit = self.evaluate_pumas()
            costs_exploit.append(np.min(fitness_exploit))

            # Combine and select best solutions
            combined_pumas = np.vstack((self.pumas, pumas_explore, pumas_exploit))
            combined_fitness = np.concatenate((self.evaluate_pumas(), fitness_explore, fitness_exploit))
            sorted_indices = np.argsort(combined_fitness)[:self.population_size]
            self.pumas = combined_pumas[sorted_indices]
            self.best_solution = self.pumas[0].copy()
            self.best_value = self.objective_function(self.best_solution)
            self.history.append((iter_count, self.best_solution.copy(), self.best_value))

            print(f"Iteration {iter_count + 1}: Best Value = {self.best_value}")

        # Hyper Initialization
        self.seq_cost_explore[0] = abs(initial_best_cost - costs_explore[0])  # Eq 5
        self.seq_cost_exploit[0] = abs(initial_best_cost - costs_exploit[0])  # Eq 8
        self.seq_cost_explore[1] = abs(costs_explore[1] - costs_explore[0])  # Eq 6
        self.seq_cost_exploit[1] = abs(costs_exploit[1] - costs_exploit[0])  # Eq 9
        self.seq_cost_explore[2] = abs(costs_explore[2] - costs_explore[1])  # Eq 7
        self.seq_cost_exploit[2] = abs(costs_exploit[2] - costs_exploit[1])  # Eq 10

        for i in range(3):
            if self.seq_cost_explore[i] != 0:
                self.pf_f3.append(self.seq_cost_explore[i])
            if self.seq_cost_exploit[i] != 0:
                self.pf_f3.append(self.seq_cost_exploit[i])

        # Calculate initial scores
        f1_explore = self.PF[0] * (self.seq_cost_explore[0] / self.seq_time_explore[0])  # Eq 1
        f1_exploit = self.PF[0] * (self.seq_cost_exploit[0] / self.seq_time_exploit[0])  # Eq 2
        f2_explore = self.PF[1] * (sum(self.seq_cost_explore) / sum(self.seq_time_explore))  # Eq 3
        f2_exploit = self.PF[1] * (sum(self.seq_cost_exploit) / sum(self.seq_time_exploit))  # Eq 4
        self.score_explore = (self.PF[0] * f1_explore) + (self.PF[1] * f2_explore)  # Eq 11
        self.score_exploit = (self.PF[0] * f1_exploit) + (self.PF[1] * f2_exploit)  # Eq 12

        # Experienced Phase
        for iter_count in range(3, self.max_iter):
            if self.score_explore > self.score_exploit:
                # ocurrance Exploration
                select_flag = 1
                self.pumas = self.exploration_phase()
                count_select = self.unselected.copy()
                self.unselected[1] += 1
                self.unselected[0] = 1
                self.f3_explore = self.PF[2]
                self.f3_exploit += self.PF[2]
                fitness = self.evaluate_pumas()
                min_idx = np.argmin(fitness)
                t_best = self.pumas[min_idx].copy()
                t_best_cost = fitness[min_idx]
                self.seq_cost_explore[2] = self.seq_cost_explore[1]
                self.seq_cost_explore[1] = self.seq_cost_explore[0]
                self.seq_cost_explore[0] = abs(self.best_value - t_best_cost)
                if self.seq_cost_explore[0] != 0:
                    self.pf_f3.append(self.seq_cost_explore[0])
                if t_best_cost < self.best_value:
                    self.best_solution = t_best.copy()
                    self.best_value = t_best_cost
            else:
                # Exploitation
                select_flag = 2
                self.pumas = self.exploitation_phase(iter_count + 1)
                count_select = self.unselected.copy()
                self.unselected[0] += 1
                self.unselected[1] = 1
                self.f3_explore += self.PF[2]
                self.f3_exploit = self.PF[2]
                fitness = self.evaluate_pumas()
                min_idx = np.argmin(fitness)
                t_best = self.pumas[min_idx].copy()
                t_best_cost = fitness[min_idx]
                self.seq_cost_exploit[2] = self.seq_cost_exploit[1]
                self.seq_cost_exploit[1] = self.seq_cost_exploit[0]
                self.seq_cost_exploit[0] = abs(self.best_value - t_best_cost)
                if self.seq_cost_exploit[0] != 0:
                    self.pf_f3.append(self.seq_cost_exploit[0])
                if t_best_cost < self.best_value:
                    self.best_solution = t_best.copy()
                    self.best_value = t_best_cost

            if self.flag_change != select_flag:
                self.flag_change = select_flag
                self.seq_time_explore[2] = self.seq_time_explore[1]
                self.seq_time_explore[1] = self.seq_time_explore[0]
                self.seq_time_explore[0] = count_select[0]
                self.seq_time_exploit[2] = self.seq_time_exploit[1]
                self.seq_time_exploit[1] = self.seq_time_exploit[0]
                self.seq_time_exploit[0] = count_select[1]

            # Update scores
            f1_explore = self.PF[0] * (self.seq_cost_explore[0] / self.seq_time_explore[0])  # Eq 14
            f1_exploit = self.PF[0] * (self.seq_cost_exploit[0] / self.seq_time_exploit[0])  # Eq 13
            f2_explore = self.PF[1] * (sum(self.seq_cost_explore) / sum(self.seq_time_explore))  # Eq 16
            f2_exploit = self.PF[1] * (sum(self.seq_cost_exploit) / sum(self.seq_time_exploit))  # Eq 15

            if self.score_explore < self.score_exploit:
                self.mega_explore = max(self.mega_explore - 0.01, 0.01)
                self.mega_exploit = 0.99
            elif self.score_explore > self.score_exploit:
                self.mega_explore = 0.99
                self.mega_exploit = max(self.mega_exploit - 0.01, 0.01)

            lmn_explore = 1 - self.mega_explore  # Eq 24
            lmn_exploit = 1 - self.mega_exploit  # Eq 22

            self.score_explore = (
                self.mega_explore * f1_explore +
                self.mega_explore * f2_explore +
                lmn_explore * (min(self.pf_f3) * self.f3_explore)
            )  # Eq 20
            self.score_exploit = (
                self.mega_exploit * f1_exploit +
                self.mega_exploit * f2_exploit +
                lmn_exploit * (min(self.pf_f3) * self.f3_exploit)
            )  # Eq 19

            self.history.append((iter_count, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iter_count + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
