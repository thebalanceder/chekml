# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport INFINITY
from cpython.array cimport array

cdef class ElephantHerdingOptimizer:
    cdef object objective_function
    cdef int dim
    cdef cnp.ndarray bounds
    cdef int population_size
    cdef int max_fes
    cdef int num_clans
    cdef double alpha
    cdef double beta
    cdef int keep
    cdef int num_elephants_per_clan
    cdef cnp.ndarray population
    cdef cnp.ndarray costs
    cdef cnp.ndarray best_solution
    cdef double best_cost
    cdef int n_evaluations
    cdef list history

    def __init__(self, object objective_function, int dim, list bounds, int population_size=50, 
                 int max_fes=30000, int num_clans=5, double alpha=0.5, double beta=0.1, int keep=2):
        """
        Initialize the Elephant Herding Optimization algorithm.

        Parameters:
        - objective_function: Function to optimize (returns cost, assumes lower is better).
        - dim: Number of dimensions (variables).
        - bounds: List of (lower, upper) bounds for each dimension.
        - population_size: Total number of elephants (solutions).
        - max_fes: Maximum number of function evaluations.
        - num_clans: Number of clans to divide the population into.
        - alpha: Clan updating factor for movement towards best elephant.
        - beta: Clan center influence factor for unchanged elephants.
        - keep: Number of elite elephants to preserve (elitism).
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.double)
        self.population_size = population_size
        self.max_fes = max_fes
        self.num_clans = num_clans
        self.alpha = alpha
        self.beta = beta
        self.keep = keep
        self.num_elephants_per_clan = population_size // num_clans
        if population_size % num_clans != 0:
            raise ValueError("Population size must be divisible by number of clans")
        self.population = None
        self.costs = None
        self.best_solution = None
        self.best_cost = INFINITY
        self.n_evaluations = 0
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void initialize_population(self):
        """ Generate initial population randomly within bounds """
        cdef cnp.ndarray[cnp.double_t, ndim=2] pop = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], (self.population_size, self.dim))
        self.population = pop
        self.costs = np.empty(self.population_size, dtype=np.double)
        cdef int i
        for i in range(self.population_size):
            self.costs[i] = self.objective_function(self.population[i])
        self.n_evaluations += self.population_size
        self._update_best()
        self._clear_duplicates()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _update_best(self):
        """ Update the best solution and cost """
        cdef int min_idx = np.argmin(self.costs)
        if self.costs[min_idx] < self.best_cost:
            self.best_solution = self.population[min_idx].copy()
            self.best_cost = self.costs[min_idx]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _clear_duplicates(self):
        """ Ensure no duplicate individuals in the population """
        cdef int i, j, k
        cdef cnp.ndarray[cnp.double_t, ndim=1] chrom1, chrom2
        cdef cnp.ndarray[cnp.double_t, ndim=1] sorted_chrom1, sorted_chrom2
        for i in range(self.population_size):
            chrom1 = self.population[i]
            sorted_chrom1 = np.sort(chrom1)
            for j in range(i + 1, self.population_size):
                chrom2 = self.population[j]
                sorted_chrom2 = np.sort(chrom2)
                if np.array_equal(sorted_chrom1, sorted_chrom2):
                    k = np.random.randint(0, self.dim)
                    self.population[j, k] = np.random.uniform(self.bounds[k, 0], self.bounds[k, 1])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.intp_t, ndim=1] _sort_population(self):
        """ Sort population by cost (ascending) """
        cdef cnp.ndarray[cnp.intp_t, ndim=1] indices = np.argsort(self.costs)
        self.population = self.population[indices]
        self.costs = self.costs[indices]
        return indices

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.double_t, ndim=1] _compute_clan_center(self, cnp.ndarray[cnp.double_t, ndim=2] clan):
        """ Calculate the center of a clan """
        return np.mean(clan, axis=0)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef list _divide_into_clans(self):
        """ Divide population into clans based on fitness """
        self._sort_population()
        cdef list clans = []
        cdef int cindex, popindex = 0
        cdef cnp.ndarray[cnp.double_t, ndim=2] clan
        for cindex in range(self.num_clans):
            clan = np.empty((self.num_elephants_per_clan, self.dim), dtype=np.double)
            for i in range(self.num_elephants_per_clan):
                clan[i] = self.population[popindex]
                popindex += 1
            clans.append(clan)
        return clans

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef list _clan_updating(self, list clans):
        """ Update clans using clan updating operator """
        cdef list new_clans = []
        cdef int cindex, j, d
        cdef cnp.ndarray[cnp.double_t, ndim=2] clan, new_clan
        cdef cnp.ndarray[cnp.double_t, ndim=1] new_chrom, current, best_elephant, clan_center
        cdef double diff_sum
        for cindex in range(self.num_clans):
            clan = clans[cindex]
            new_clan = np.empty((self.num_elephants_per_clan, self.dim), dtype=np.double)
            clan_center = self._compute_clan_center(clan)
            best_elephant = clan[0]
            for j in range(self.num_elephants_per_clan):
                current = clan[j]
                new_chrom = np.empty(self.dim, dtype=np.double)
                diff_sum = 0.0
                for d in range(self.dim):
                    new_chrom[d] = current[d] + self.alpha * (best_elephant[d] - current[d]) * np.random.rand()
                    diff_sum += new_chrom[d] - current[d]
                if abs(diff_sum) < 1e-10:  # Check if new_chrom is unchanged
                    new_chrom = self.beta * clan_center
                for d in range(self.dim):
                    new_chrom[d] = min(max(new_chrom[d], self.bounds[d, 0]), self.bounds[d, 1])
                new_clan[j] = new_chrom
            new_clans.append(new_clan)
        return new_clans

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _separating_operator(self, list new_clans):
        """ Apply separating operator to the worst elephant in each clan """
        cdef int cindex
        for cindex in range(self.num_clans):
            new_clans[cindex][-1] = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef list _evaluate_clans(self, list new_clans):
        """ Evaluate new clans and ensure feasibility """
        cdef int i, j, k
        cdef cnp.ndarray[cnp.double_t, ndim=2] clan
        cdef cnp.ndarray[cnp.double_t, ndim=1] chrom1, chrom2, costs
        cdef cnp.ndarray[cnp.intp_t, ndim=1] indices
        for i in range(self.num_clans):
            clan = np.array(new_clans[i], dtype=np.double)
            # Clear duplicates
            for j in range(len(clan)):
                chrom1 = clan[j]
                sorted_chrom1 = np.sort(chrom1)
                for k in range(j + 1, len(clan)):
                    chrom2 = clan[k]
                    sorted_chrom2 = np.sort(chrom2)
                    if np.array_equal(sorted_chrom1, sorted_chrom2):
                        k = np.random.randint(0, self.dim)
                        clan[k, k] = np.random.uniform(self.bounds[k, 0], self.bounds[k, 1])
            # Evaluate costs
            costs = np.empty(len(clan), dtype=np.double)
            for j in range(len(clan)):
                costs[j] = self.objective_function(clan[j])
            self.n_evaluations += len(clan)
            # Sort clan
            indices = np.argsort(costs)
            new_clans[i] = clan[indices]
        return new_clans

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.double_t, ndim=2] _combine_clans(self, list new_clans):
        """ Combine clans into a single population """
        cdef cnp.ndarray[cnp.double_t, ndim=2] population = np.empty((self.population_size, self.dim), dtype=np.double)
        cdef int j = 0, popindex = 0, cindex
        while popindex < self.population_size and j < self.num_elephants_per_clan:
            for cindex in range(self.num_clans):
                population[popindex] = new_clans[cindex][j]
                popindex += 1
            j += 1
        return population

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _apply_elitism(self):
        """ Replace worst individuals with elite solutions """
        cdef cnp.ndarray[cnp.double_t, ndim=2] elite_chroms = self.population[:self.keep].copy()
        cdef cnp.ndarray[cnp.double_t, ndim=1] elite_costs = self.costs[:self.keep].copy()
        self._sort_population()
        self.population[-self.keep:] = elite_chroms
        self.costs[-self.keep:] = elite_costs

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple optimize(self):
        """ Run the Elephant Herding Optimization algorithm """
        cdef int gen_index, i
        cdef cnp.ndarray[cnp.double_t, ndim=2] elite_chroms
        cdef cnp.ndarray[cnp.double_t, ndim=1] elite_costs, legal_costs
        cdef double avg_cost
        self.initialize_population()
        gen_index = 1
        while self.n_evaluations < self.max_fes:
            # Save elites
            elite_chroms = self.population[:self.keep].copy()
            elite_costs = self.costs[:self.keep].copy()

            # Divide into clans
            clans = self._divide_into_clans()

            # Clan updating
            new_clans = self._clan_updating(clans)

            # Separating operator
            self._separating_operator(new_clans)

            # Evaluate clans
            new_clans = self._evaluate_clans(new_clans)

            # Combine clans
            self.population = self._combine_clans(new_clans)
            self.costs = np.empty(self.population_size, dtype=np.double)
            for i in range(self.population_size):
                self.costs[i] = self.objective_function(self.population[i])
            self.n_evaluations += self.population_size

            # Apply elitism
            self.population[-self.keep:] = elite_chroms
            self.costs[-self.keep:] = elite_costs

            # Sort and update best
            self._sort_population()
            self._update_best()

            # Compute average cost
            legal_costs = self.costs[self.costs < INFINITY]
            avg_cost = np.mean(legal_costs) if len(legal_costs) > 0 else INFINITY

            # Log results
            self.history.append((gen_index, self.best_solution.copy(), self.best_cost, avg_cost))
            print(f"Generation {gen_index}: Best Cost = {self.best_cost}, Avg Cost = {avg_cost}")

            gen_index += 1

        return self.best_solution, self.best_cost, self.history
