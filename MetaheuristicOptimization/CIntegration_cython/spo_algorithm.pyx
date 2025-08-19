import numpy as np
cimport numpy as np

# Define the type of the input and output arrays
ctypedef np.float64_t DTYPE_t

class StochasticPaintOptimizer:
    def __init__(self, fun, lb, ub, problem_size=30, batch_size=50, epoch=100):
        self.fun = fun
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.problem_size = problem_size
        self.batch_size = batch_size
        self.epoch = epoch

        self.lb_vec = np.full(self.problem_size, self.lb)
        self.ub_vec = np.full(self.problem_size, self.ub)

        self.N1 = self.batch_size // 3
        self.N2 = self.batch_size // 3
        self.N3 = self.batch_size - self.N1 - self.N2

        self.colors = self._initialize_population()
        self.fitness = np.array([self.fun(ind) for ind in self.colors])

        self.best_pos = None
        self.best_fit = float("inf")
        self.history = []  # To store the history of the optimization process

    def _initialize_population(self):
        return np.random.uniform(self.lb, self.ub, (self.batch_size, self.problem_size))

    def _bound(self, np.ndarray[DTYPE_t, ndim=1] x):
        return np.clip(x, self.lb_vec, self.ub_vec)

    def optimize(self):
        for epoch in range(self.epoch):
            # Sort and regroup
            sort_idx = np.argsort(self.fitness)
            self.colors = self.colors[sort_idx]
            self.fitness = self.fitness[sort_idx]

            group1 = self.colors[:self.N1]
            group2 = self.colors[self.N1:self.N1 + self.N2]
            group3 = self.colors[self.N1 + self.N2:]

            new_colors = []
            new_fitness = []

            for i in range(self.batch_size):
                xi = self.colors[i]

                # Complement Combination
                id1 = np.random.randint(0, self.N1)
                id2 = np.random.randint(0, self.N3)
                complement = xi + np.random.rand(self.problem_size) * (group1[id1] - group3[id2])

                # Analog Combination
                if i < self.N1:
                    analog_group = group1
                    ids = np.random.randint(0, self.N1, 2)
                elif i < self.N1 + self.N2:
                    analog_group = group2
                    ids = np.random.randint(0, self.N2, 2)
                else:
                    analog_group = group3
                    ids = np.random.randint(0, self.N3, 2)
                analog = xi + np.random.rand(self.problem_size) * (analog_group[ids[1]] - analog_group[ids[0]])

                # Triangle Combination
                tid1 = np.random.randint(0, self.N1)
                tid2 = np.random.randint(0, self.N2)
                tid3 = np.random.randint(0, self.N3)
                triangle = xi + np.random.rand(self.problem_size) * ((group1[tid1] + group2[tid2] + group3[tid3]) / 3)

                # Rectangle Combination
                rid1 = np.random.randint(0, self.N1)
                rid2 = np.random.randint(0, self.N2)
                rid3 = np.random.randint(0, self.N3)
                rid4 = np.random.randint(0, self.batch_size)
                rectangle = xi + (
                    np.random.rand(self.problem_size) * group1[rid1] +
                    np.random.rand(self.problem_size) * group2[rid2] +
                    np.random.rand(self.problem_size) * group3[rid3] +
                    np.random.rand(self.problem_size) * self.colors[rid4]
                ) / 4

                # Evaluate all candidates
                for candidate in [complement, analog, triangle, rectangle]:
                    candidate = self._bound(candidate)
                    fit = self.fun(candidate)
                    new_colors.append(candidate)
                    new_fitness.append(fit)

            # Combine and select best
            self.colors = np.vstack((self.colors, new_colors))
            self.fitness = np.hstack((self.fitness, new_fitness))

            best_idx = np.argsort(self.fitness)[:self.batch_size]
            self.colors = self.colors[best_idx]
            self.fitness = self.fitness[best_idx]

            # Track the best solution and its fitness during the process
            self.best_pos = self.colors[0]
            self.best_fit = self.fitness[0]
            self.history.append((epoch, self.best_pos, self.best_fit))  # Save history
            print(f"Iteration {epoch + 1}: Best Value = {self.best_fit}")
            
        return self.best_pos, self.best_fit, self.history  # Return history alongside best_pos and best_fit

