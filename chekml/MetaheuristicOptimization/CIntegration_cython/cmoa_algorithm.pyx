# cython: language_level=3
# distutils: language = c++

import numpy as np
cimport numpy as cnp
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
class CoronavirusMetamorphosisOptimizer:
    def __init__(self, objective_function, int dim, bounds, int population_size=50, int max_iter=100, 
                 double mutation_rate=0.3, double crossover_rate=0.5):
        """
        Initialize the CMOA optimizer.

        Parameters:
        - objective_function: Function to optimize (minimize).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of viruses (solutions).
        - max_iter: Maximum number of iterations.
        - mutation_rate: Probability of non-genetic mutation.
        - crossover_rate: Probability of genetic recombination.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.double)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.viruses = None  # Population of viruses (solutions)
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_viruses(self):
        """Generate initial virus population randomly."""
        self.viruses = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                        (self.population_size, self.dim))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def evaluate_viruses(self):
        """Compute fitness values for the virus population."""
        cdef cnp.ndarray[cnp.double_t, ndim=1] fitness = np.empty(self.population_size, dtype=np.double)
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.viruses[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def find_closest_virus(self, int index):
        """Find the index of the closest virus to the virus at the given index."""
        cdef cnp.ndarray[cnp.double_t, ndim=1] current_virus = self.viruses[index]
        cdef cnp.ndarray[cnp.double_t, ndim=1] distances = np.abs(np.sum(self.viruses - current_virus, axis=1))
        distances[index] = np.inf  # Exclude the virus itself
        return np.argmin(distances)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def genetic_recombination(self, int index):
        """Simulate genetic recombination by moving towards the closest virus."""
        cdef cnp.ndarray[cnp.double_t, ndim=1] new_virus = self.viruses[index].copy()
        if np.random.rand() < self.crossover_rate:
            closest_idx = self.find_closest_virus(index)
            closest_virus = self.viruses[closest_idx]
            new_virus = self.viruses[index] + np.random.rand(self.dim) * (closest_virus - self.viruses[index])
            new_virus = np.clip(new_virus, self.bounds[:, 0], self.bounds[:, 1])
        return new_virus

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def cross_activation(self, int index):
        """Simulate cross-activation by moving towards the best solution."""
        cdef cnp.ndarray[cnp.double_t, ndim=1] new_virus = self.viruses[index].copy()
        if self.best_solution is not None:
            new_virus = self.viruses[index] + np.random.rand(self.dim) * (self.best_solution - self.viruses[index])
            new_virus = np.clip(new_virus, self.bounds[:, 0], self.bounds[:, 1])
        return new_virus

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def incremental_reactivation(self, int index, int iteration):
        """Simulate incremental reactivation with rapid movement."""
        cdef cnp.ndarray[cnp.double_t, ndim=1] new_virus = self.viruses[index].copy()
        t = iteration / self.max_iter
        evolutionary_operator = np.cos(np.pi * t)  # Dynamic operator based on iteration
        if self.best_solution is not None:
            new_virus = self.viruses[index] + evolutionary_operator * np.random.rand(self.dim) * \
                        (self.best_solution - self.viruses[index])
            new_virus = np.clip(new_virus, self.bounds[:, 0], self.bounds[:, 1])
        return new_virus

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def non_genetic_mutation(self, int index):
        """Simulate non-genetic mutation by random perturbation."""
        cdef cnp.ndarray[cnp.double_t, ndim=1] new_virus = self.viruses[index].copy()
        if np.random.rand() < self.mutation_rate:
            new_virus = self.viruses[index] + np.random.uniform(-0.1, 0.1, self.dim) * \
                        (self.bounds[:, 1] - self.bounds[:, 0])
            new_virus = np.clip(new_virus, self.bounds[:, 0], self.bounds[:, 1])
        return new_virus

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def genotypic_mixing(self, int index):
        """Simulate genotypic mixing by combining information from two random viruses."""
        indices = np.random.choice(self.population_size, size=2, replace=False)
        idx1 = indices[0]
        idx2 = indices[1]
        cdef cnp.ndarray[cnp.double_t, ndim=1] new_virus = self.viruses[index] + np.random.rand(self.dim) * \
                    (self.viruses[idx1] - self.viruses[idx2])
        new_virus = np.clip(new_virus, self.bounds[:, 0], self.bounds[:, 1])
        return new_virus

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """Run the Coronavirus Metamorphosis Optimization Algorithm."""
        self.initialize_viruses()
        cdef cnp.ndarray[cnp.double_t, ndim=1] fitness
        for iteration in range(self.max_iter):
            fitness = self.evaluate_viruses()
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.viruses[min_idx].copy()
                self.best_value = fitness[min_idx]

            # Apply CMOA phases
            new_viruses = np.zeros_like(self.viruses)
            for i in range(self.population_size):
                # Genetic recombination
                new_viruses[i] = self.genetic_recombination(i)
                # Cross-activation
                new_viruses[i] = self.cross_activation(i)
                # Incremental reactivation
                new_viruses[i] = self.incremental_reactivation(i, iteration)
                # Non-genetic mutation
                new_viruses[i] = self.non_genetic_mutation(i)
                # Genotypic mixing
                new_viruses[i] = self.genotypic_mixing(i)

            self.viruses = new_viruses

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
