import numpy as np

class ClonalSelectionAlgorithm:
    def __init__(self, objective_function, bounds, population_size=100, generations=50, 
                 mutation_prob=0.1, clone_rate=0.1, replace_rate=0.1, dim=2):
        self.objective_function = objective_function
        self.bounds = bounds
        self.dim = dim
        self.population_size = population_size
        self.generations = generations
        self.mutation_prob = mutation_prob
        self.clone_rate = clone_rate
        self.replace_rate = replace_rate
        self.bits_per_dim = 22  # bit precision
        self.bit_length = self.bits_per_dim * self.dim

        self.population = self._initialize_population()
        self.history = []

    def _initialize_population(self):
        return np.random.randint(0, 2, (self.population_size, self.bit_length))

    def _decode(self, binary):
        decoded = []
        for d in range(self.dim):
            start = d * self.bits_per_dim
            end = start + self.bits_per_dim
            gene_strs = ["".join(map(str, row[start:end])) for row in binary]
            gene_vals = [int(s, 2) / (2**self.bits_per_dim - 1) for s in gene_strs]
            scaled_vals = [self.bounds[d][0] + val * (self.bounds[d][1] - self.bounds[d][0]) for val in gene_vals]
            decoded.append(scaled_vals)
        return np.column_stack(decoded)

    def _evaluate(self, decoded_population):
        return np.array([self.objective_function(ind) for ind in decoded_population])

    def _reproduce(self, sorted_indices):
        clone_sizes = np.round(self.clone_rate * self.population_size * 
                               (np.linspace(1, 0.1, self.population_size))).astype(int)
        clones = [np.tile(self.population[i], (clone_sizes[i], 1)) for i in sorted_indices]
        return np.vstack(clones), np.cumsum(clone_sizes)

    def _mutate(self, clones):
        mutation_mask = np.random.rand(*clones.shape) < self.mutation_prob
        return np.bitwise_xor(clones, mutation_mask.astype(int))

    def _pm_cont(self, current_pm, initial_pm, decay_rate, current_iter, max_iter):
        return initial_pm * ((1 - decay_rate * current_iter / max_iter))

    def optimize(self):
        pm = self.mutation_prob
        for gen in range(self.generations):
            decoded = self._decode(self.population)
            fitness = self._evaluate(decoded)
            sorted_indices = np.argsort(fitness)
            best_index = sorted_indices[0]
            best_fitness = fitness[best_index]
            best_solution = self.population[best_index]
            decoded_position = self._decode(np.array([best_solution]))[0]

            # ✅ Only one history entry per generation
            self.history.append((gen, decoded_position.tolist(), best_fitness))

            # Reproduction & mutation
            clones, pcs = self._reproduce(sorted_indices)
            mutated_clones = self._mutate(clones)

            decoded_clones = self._decode(mutated_clones)
            fitness_clones = self._evaluate(decoded_clones)

            pcs = np.insert(pcs, 0, 0)
            best_clones = [
                mutated_clones[pcs[i] + np.argmin(fitness_clones[pcs[i]:pcs[i+1]])]
                for i in range(self.population_size)
            ]

            self.population[sorted_indices[::-1]] = best_clones

            # Replace worst
            num_replace = int(self.replace_rate * self.population_size)
            self.population[sorted_indices[-num_replace:]] = self._initialize_population()[:num_replace]

            # Decay mutation
            pm = self._pm_cont(pm, self.mutation_prob, 0.8, gen, self.generations)

            print(f"Generation {gen + 1}: Best Fitness = {best_fitness:.5f}")

        final_decoded = self._decode(self.population)
        final_fitness = self._evaluate(final_decoded)
        best_idx = np.argmin(final_fitness)
        return final_decoded[best_idx], final_fitness[best_idx], self.history
