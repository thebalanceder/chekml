import numpy as np

class HarmonySearch:
    def __init__(self, objective_function, dim=2, bounds=None,
                 memory_size=100, max_iterations=100,
                 harmony_memory_considering_rate=0.95, pitch_adjustment_rate=0.3,
                 bandwidth=0.2, minimize=True):
        """
        Initialize the Harmony Search optimizer.

        Parameters:
        - objective_function: Function to be minimized or maximized.
        - dim: Dimensionality of the problem.
        - bounds: List of (lower, upper) tuples per dimension.
        - memory_size: Harmony Memory Size (HMS).
        - max_iterations: Maximum number of iterations.
        - harmony_memory_considering_rate: HMCR.
        - pitch_adjustment_rate: PAR.
        - bandwidth: Bandwidth for pitch adjustment.
        - minimize: True to minimize the objective function, False to maximize.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = bounds if bounds else [(-10, 10)] * dim
        self.memory_size = memory_size
        self.max_iterations = max_iterations
        self.HMCR = harmony_memory_considering_rate
        self.PAR = pitch_adjustment_rate
        self.bw = bandwidth
        self.minimize = minimize

        self.lower_bounds = np.array([b[0] for b in self.bounds])
        self.upper_bounds = np.array([b[1] for b in self.bounds])

        self.harmony_memory = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.memory_size, self.dim))
        self.fitness = np.array([self.objective_function(harmony) for harmony in self.harmony_memory])

        if self.minimize:
            self.best_idx = np.argmin(self.fitness)
        else:
            self.best_idx = np.argmax(self.fitness)

        self.best_solution = self.harmony_memory[self.best_idx].copy()
        self.best_fitness = self.fitness[self.best_idx]
        self.history = []

    def optimize(self):
        for itr in range(self.max_iterations):
            # === Harmony improvisation ===
            indices = np.random.randint(0, self.memory_size, self.dim)
            harmony = self.harmony_memory[indices, np.arange(self.dim)]

            cm_mask = np.random.rand(self.dim) < self.HMCR
            pa_mask = (np.random.rand(self.dim) < self.PAR) & cm_mask
            rand_mask = ~cm_mask

            random_harmony = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
            adjusted_harmony = harmony + self.bw * (2 * np.random.rand(self.dim) - 1)

            new_harmony = np.where(cm_mask, harmony, random_harmony)
            new_harmony = np.where(pa_mask, adjusted_harmony, new_harmony)

            # Clamp to bounds
            out_of_bounds = (new_harmony > self.upper_bounds) | (new_harmony < self.lower_bounds)
            new_harmony[out_of_bounds] = harmony[out_of_bounds]

            # Evaluate new harmony
            new_fitness = self.objective_function(new_harmony)

            # Replace the worst harmony if better
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

            # Update best
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

