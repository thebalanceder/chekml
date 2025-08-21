# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as cnp
from libc.math cimport pow

cdef class ClonalSelectionAlgorithm:
    cdef:
        object objective_function
        list bounds
        int population_size, generations, bits_per_dim, bit_length, dim
        double mutation_prob, clone_rate, replace_rate
        object population  # Use 'object' here
        list history

    def __init__(self, objective_function, bounds, int population_size=100, int generations=50, 
                 double mutation_prob=0.1, double clone_rate=0.1, double replace_rate=0.1, int dim=2):
        cnp.import_array()  # Initialize NumPy C API here
        self.objective_function = objective_function
        self.bounds = bounds
        self.population_size = population_size
        self.generations = generations
        self.mutation_prob = mutation_prob
        self.clone_rate = clone_rate
        self.replace_rate = replace_rate
        self.dim = dim
        self.bits_per_dim = 22
        self.bit_length = self.bits_per_dim * self.dim
        self.population = self._initialize_population()
        self.history = []

    cdef object _initialize_population(self):
        return np.random.randint(0, 2, (self.population_size, self.bit_length), dtype=np.int32)

    cdef cnp.ndarray[cnp.float64_t, ndim=2] _decode(self, object binary):
        cdef int n = binary.shape[0]
        cdef cnp.ndarray[cnp.float64_t, ndim=2] decoded = np.zeros((n, self.dim), dtype=np.float64)
        cdef int i, d
        cdef int start, end
        cdef str s
        cdef int val
        cdef double scale

        for d in range(self.dim):
            start = d * self.bits_per_dim
            end = start + self.bits_per_dim
            for i in range(n):
                s = ''.join(map(str, binary[i, start:end]))
                val = int(s, 2)
                scale = val / (2 ** self.bits_per_dim - 1)
                decoded[i, d] = self.bounds[d][0] + scale * (self.bounds[d][1] - self.bounds[d][0])
        return decoded

    cdef cnp.ndarray[cnp.float64_t, ndim=1] _evaluate(self, cnp.ndarray[cnp.float64_t, ndim=2] pop):
        cdef int i
        cdef cnp.ndarray[cnp.float64_t, ndim=1] fitness = np.zeros(pop.shape[0], dtype=np.float64)
        for i in range(pop.shape[0]):
            fitness[i] = self.objective_function(pop[i])
        return fitness

    cdef tuple _reproduce(self, object sorted_indices):
        cdef int i
        clone_sizes = np.round(self.clone_rate * self.population_size * 
                               np.linspace(1, 0.1, self.population_size)).astype(np.int32)
        clones = [np.tile(self.population[sorted_indices[i]], (clone_sizes[i], 1)) for i in range(self.population_size)]
        pcs = np.cumsum(clone_sizes)
        return np.vstack(clones), pcs

    cdef object _mutate(self, object clones):
        mutation_mask = (np.random.rand(*clones.shape) < self.mutation_prob).astype(np.int32)
        return np.bitwise_xor(clones, mutation_mask)

    cdef double _pm_cont(self, double current_pm, double initial_pm, double decay_rate, int current_iter, int max_iter):
        return initial_pm * (1 - decay_rate * current_iter / max_iter)

    cpdef tuple optimize(self):
        cdef int gen, best_index, i, num_replace
        cdef double pm = self.mutation_prob
        cdef cnp.ndarray[cnp.float64_t, ndim=2] decoded, decoded_clones
        cdef cnp.ndarray[cnp.float64_t, ndim=1] fitness, fitness_clones
        cdef object sorted_indices
        cdef object mutated_clones
        cdef list best_clones
        cdef object pcs
        cdef object final_decoded
        cdef object final_fitness
        cdef int best_idx
        cdef list decoded_position

        for gen in range(self.generations):
            decoded = self._decode(self.population)
            fitness = self._evaluate(decoded)
            sorted_indices = np.argsort(fitness)
            best_index = sorted_indices[0]
            best_fitness = fitness[best_index]
            decoded_position = self._decode(self.population[best_index:best_index+1])[0].tolist()

            self.history.append((gen, decoded_position, best_fitness))

            # Clone and mutate
            clones, pcs = self._reproduce(sorted_indices)
            pcs = np.insert(pcs, 0, 0)
            mutated_clones = self._mutate(clones)
            decoded_clones = self._decode(mutated_clones)
            fitness_clones = self._evaluate(decoded_clones)

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
