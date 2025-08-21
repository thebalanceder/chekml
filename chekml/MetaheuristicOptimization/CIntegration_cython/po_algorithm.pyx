# cython: language_level=3
import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport abs, sin, sqrt

# Ensure numpy C API is available
cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class PoliticalOptimizer:
    cdef public:
        object objective_function
        int dim
        cnp.ndarray bounds
        int population_size
        int parties
        int areas
        int max_iter
        double lambda_rate
        cnp.ndarray positions
        cnp.ndarray fitness
        cnp.ndarray aux_positions
        cnp.ndarray aux_fitness
        cnp.ndarray prev_positions
        cnp.ndarray prev_fitness
        cnp.ndarray a_winners
        cnp.ndarray a_winner_indices
        cnp.ndarray best_solution
        double best_value
        list history

    def __init__(self, objective_function, int dim, bounds, int population_size=64, int parties=8, int max_iter=100, double lambda_rate=1.0):
        """
        Initialize the Political Optimizer (PO).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Total number of search agents (parties * areas).
        - parties: Number of political parties.
        - max_iter: Maximum number of iterations.
        - lambda_rate: Maximum limit of party switching rate.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.float64)
        self.population_size = population_size
        self.parties = parties
        self.areas = population_size // parties
        self.max_iter = max_iter
        self.lambda_rate = lambda_rate
        self.best_value = float("inf")
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def initialize_population(self):
        """ Generate initial population randomly """
        cdef cnp.ndarray[cnp.float64_t, ndim=2] positions
        positions = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                     (self.population_size, self.dim))
        self.positions = positions
        self.aux_positions = positions.copy()
        self.prev_positions = positions.copy()
        self.fitness = np.array([self.objective_function(pos) for pos in positions], dtype=np.float64)
        self.aux_fitness = self.fitness.copy()
        self.prev_fitness = self.fitness.copy()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def election(self):
        """ Perform election phase to select constituency winners """
        cdef cnp.ndarray[cnp.float64_t, ndim=2] a_winners = np.zeros((self.areas, self.dim), dtype=np.float64)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] a_winner_indices = np.zeros(self.areas, dtype=np.int32)
        cdef int a, start_idx, end_idx, step, min_idx
        cdef cnp.ndarray[cnp.int32_t, ndim=1] constituency_indices
        cdef cnp.ndarray[cnp.float64_t, ndim=1] constituency_fitness

        for a in range(self.areas):
            start_idx = a
            end_idx = self.population_size
            step = self.areas
            constituency_indices = np.arange(start_idx, end_idx, step, dtype=np.int32)
            constituency_fitness = self.fitness[constituency_indices]
            min_idx = np.argmin(constituency_fitness)
            a_winner_indices[a] = constituency_indices[min_idx]
            a_winners[a, :] = self.positions[a_winner_indices[a], :]

        self.a_winners = a_winners
        self.a_winner_indices = a_winner_indices

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def government_formation(self):
        """ Update positions based on government formation """
        cdef int p, a, member_idx, party_leader_idx
        cdef cnp.ndarray[cnp.int32_t, ndim=1] party_indices
        cdef cnp.ndarray[cnp.float64_t, ndim=1] party_fitness
        cdef cnp.ndarray[cnp.float64_t, ndim=1] party_leader_pos, member_pos, new_pos
        cdef double new_fitness, r

        for p in range(self.parties):
            party_start = p * self.areas
            party_indices = np.arange(party_start, party_start + self.areas, dtype=np.int32)
            party_fitness = self.fitness[party_indices]
            party_leader_idx = party_indices[np.argmin(party_fitness)]
            party_leader_pos = self.positions[party_leader_idx, :]
            for a in range(self.areas):
                member_idx = party_start + a
                if member_idx != party_leader_idx:
                    member_pos = self.positions[member_idx, :]
                    r = <double>rand() / RAND_MAX
                    new_pos = member_pos + r * (party_leader_pos - member_pos)
                    new_pos = np.clip(new_pos, self.bounds[:, 0], self.bounds[:, 1])
                    new_fitness = self.objective_function(new_pos)
                    if new_fitness < self.fitness[member_idx]:
                        self.positions[member_idx, :] = new_pos
                        self.fitness[member_idx] = new_fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def election_campaign(self):
        """ Perform election campaign phase """
        cdef int p, a, member_idx, i
        cdef cnp.ndarray[cnp.float64_t, ndim=1] current_pos, prev_pos, new_pos
        cdef cnp.ndarray[cnp.float64_t, ndim=1] r
        cdef double new_fitness

        for p in range(self.parties):
            party_start = p * self.areas
            for a in range(self.areas):
                member_idx = party_start + a
                current_pos = self.positions[member_idx, :]
                prev_pos = self.prev_positions[member_idx, :]
                r = np.random.rand(self.dim)
                new_pos = current_pos + r * (current_pos - prev_pos)
                new_pos = np.clip(new_pos, self.bounds[:, 0], self.bounds[:, 1])
                new_fitness = self.objective_function(new_pos)
                if new_fitness < self.fitness[member_idx]:
                    self.positions[member_idx, :] = new_pos
                    self.fitness[member_idx] = new_fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def party_switching(self, int t):
        """ Perform party switching phase """
        cdef double psr = (1 - t * (1.0 / self.max_iter)) * self.lambda_rate
        cdef int p, a, from_idx, to_party, to_start, to_least_fit_idx
        cdef cnp.ndarray[cnp.int32_t, ndim=1] to_indices
        cdef double r

        for p in range(self.parties):
            for a in range(self.areas):
                from_idx = p * self.areas + a
                r = <double>rand() / RAND_MAX
                if r < psr:
                    to_party = rand() % self.parties
                    while to_party == p:
                        to_party = rand() % self.parties
                    to_start = to_party * self.areas
                    to_indices = np.arange(to_start, to_start + self.areas, dtype=np.int32)
                    to_least_fit_idx = to_indices[np.argmax(self.fitness[to_indices])]
                    # Swap positions
                    self.positions[[to_least_fit_idx, from_idx], :] = self.positions[[from_idx, to_least_fit_idx], :]
                    # Swap fitness
                    self.fitness[[to_least_fit_idx, from_idx]] = self.fitness[[from_idx, to_least_fit_idx]]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def parliamentarism(self):
        """ Perform parliamentarism phase """
        cdef int a, winner_idx, to_area, i
        cdef cnp.ndarray[cnp.float64_t, ndim=1] new_winner, to_winner, distance
        cdef double new_fitness
        cdef cnp.ndarray[cnp.float64_t, ndim=1] r

        for a in range(self.areas):
            new_winner = self.a_winners[a, :].copy()
            winner_idx = self.a_winner_indices[a]
            to_area = rand() % self.areas
            while to_area == a:
                to_area = rand() % self.areas
            to_winner = self.a_winners[to_area, :]
            distance = np.abs(to_winner - new_winner)
            r = 2 * np.random.rand(self.dim) - 1
            new_winner = to_winner + r * distance
            new_winner = np.clip(new_winner, self.bounds[:, 0], self.bounds[:, 1])
            new_fitness = self.objective_function(new_winner)
            if new_fitness < self.fitness[winner_idx]:
                self.positions[winner_idx, :] = new_winner
                self.fitness[winner_idx] = new_fitness
                self.a_winners[a, :] = new_winner

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """ Run the Political Optimizer """
        self.initialize_population()
        self.election()
        self.aux_fitness = self.fitness.copy()
        self.prev_fitness = self.fitness.copy()
        self.government_formation()

        self.history = []
        cdef int t, min_idx
        for t in range(self.max_iter):
            self.prev_fitness = self.aux_fitness.copy()
            self.prev_positions = self.aux_positions.copy()
            self.aux_fitness = self.fitness.copy()
            self.aux_positions = self.positions.copy()

            self.election_campaign()
            self.party_switching(t)
            self.election()
            self.government_formation()
            self.parliamentarism()

            min_idx = np.argmin(self.fitness)
            if self.fitness[min_idx] < self.best_value:
                self.best_solution = self.positions[min_idx, :].copy()
                self.best_value = self.fitness[min_idx]

            self.history.append((t, self.best_solution.copy(), self.best_value))
            print(f"Iteration {t + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
