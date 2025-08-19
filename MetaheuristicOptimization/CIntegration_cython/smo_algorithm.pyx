# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sin, sqrt, abs
from libc.stdlib cimport rand, RAND_MAX

# Type definitions for NumPy arrays
np.import_array()
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class SpiderMonkeyOptimizer:
    cdef object objective_function
    cdef int dim
    cdef np.ndarray bounds
    cdef int population_size
    cdef int max_iter
    cdef double pr
    cdef int local_leader_limit
    cdef int global_leader_limit
    cdef int max_groups
    cdef np.ndarray spider_monkeys
    cdef np.ndarray fitness
    cdef list local_leaders
    cdef list local_leader_fitness
    cdef np.ndarray global_leader
    cdef double global_leader_fitness
    cdef list groups
    cdef list local_leader_count
    cdef int global_leader_count
    cdef list history

    def __init__(self, objective_function, int dim, bounds, int population_size=50, int max_iter=100, 
                 double pr=0.1, int local_leader_limit=50, int global_leader_limit=1500, int max_groups=5):
        """
        Initialize the Spider Monkey Optimization (SMO) algorithm with Beta-Hill Climbing Optimizer (BHC) integration.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of spider monkeys (solutions).
        - max_iter: Maximum number of iterations.
        - pr: Perturbation rate for position updates.
        - local_leader_limit: Limit for local leader stagnation.
        - global_leader_limit: Limit for global leader stagnation.
        - max_groups: Maximum number of groups for fission-fusion.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.pr = pr
        self.local_leader_limit = local_leader_limit
        self.global_leader_limit = global_leader_limit
        self.max_groups = max_groups

        self.spider_monkeys = None
        self.fitness = None
        self.local_leaders = None
        self.local_leader_fitness = None
        self.global_leader = None
        self.global_leader_fitness = float("inf")
        self.groups = []
        self.local_leader_count = None
        self.global_leader_count = 0
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_population(self):
        """ Initialize spider monkey positions and groups """
        cdef np.ndarray[DTYPE_t, ndim=2] spider_monkeys = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], (self.population_size, self.dim))
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = np.zeros(self.population_size, dtype=DTYPE)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(spider_monkeys[i])
        
        self.spider_monkeys = spider_monkeys
        self.fitness = fitness
        self.groups = [list(range(self.population_size))]
        cdef int min_idx = np.argmin(fitness)
        self.local_leaders = [spider_monkeys[min_idx].copy()]
        self.local_leader_fitness = [fitness[min_idx]]
        self.local_leader_count = [0]
        self.global_leader = spider_monkeys[min_idx].copy()
        self.global_leader_fitness = fitness[min_idx]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef tuple beta_hill_climbing(self, np.ndarray[DTYPE_t, ndim=1] solution, double fitness, double delta=0.1):
        """
        Apply Beta-Hill Climbing Optimizer (BHC) to refine a solution.
        
        Parameters:
        - solution: Current solution (spider monkey position).
        - fitness: Fitness of the current solution.
        - delta: Step size for BHC.
        """
        cdef np.ndarray[DTYPE_t, ndim=1] new_solution = solution.copy()
        cdef int i
        cdef double beta
        for i in range(self.dim):
            if rand() / RAND_MAX < 0.5:
                # Approximate beta distribution (simplified for Cython)
                beta = np.random.beta(2, 5)
                new_solution[i] += delta * beta * (self.bounds[i, 1] - self.bounds[i, 0])
            else:
                new_solution[i] += delta * (2 * (rand() / RAND_MAX) - 1)
        
        for i in range(self.dim):
            if new_solution[i] < self.bounds[i, 0]:
                new_solution[i] = self.bounds[i, 0]
            elif new_solution[i] > self.bounds[i, 1]:
                new_solution[i] = self.bounds[i, 1]
        
        cdef double new_fitness = self.objective_function(new_solution)
        if new_fitness < fitness:
            return new_solution, new_fitness
        return solution, fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void local_leader_phase(self):
        """ Update spider monkey positions in Local Leader Phase (LLP) with BHC """
        cdef int group_idx, idx, rand_member
        cdef list group
        cdef np.ndarray[DTYPE_t, ndim=1] new_position
        cdef double new_fitness
        for group_idx in range(len(self.groups)):
            group = self.groups[group_idx]
            for idx in group:
                if rand() / RAND_MAX > self.pr:
                    rand_member = group[rand() % len(group)]
                    new_position = np.zeros(self.dim, dtype=DTYPE)
                    for i in range(self.dim):
                        new_position[i] = (self.spider_monkeys[idx, i] +
                                          (self.local_leaders[group_idx][i] - self.spider_monkeys[idx, i]) * (rand() / RAND_MAX) +
                                          (self.spider_monkeys[idx, i] - self.spider_monkeys[rand_member, i]) * (2 * (rand() / RAND_MAX) - 1))
                        if new_position[i] < self.bounds[i, 0]:
                            new_position[i] = self.bounds[i, 0]
                        elif new_position[i] > self.bounds[i, 1]:
                            new_position[i] = self.bounds[i, 1]
                    
                    new_position, new_fitness = self.beta_hill_climbing(new_position, self.fitness[idx])
                    if new_fitness < self.fitness[idx]:
                        self.spider_monkeys[idx] = new_position
                        self.fitness[idx] = new_fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void global_leader_phase(self):
        """ Update spider monkey positions in Global Leader Phase (GLP) with BHC """
        cdef np.ndarray[DTYPE_t, ndim=1] prob = 0.9 * (1 - self.fitness / np.max(self.fitness)) + 0.1
        cdef int i, group_idx, rand_member
        cdef np.ndarray[DTYPE_t, ndim=1] new_position
        cdef double new_fitness
        for i in range(self.population_size):
            if rand() / RAND_MAX < prob[i]:
                for group_idx in range(len(self.groups)):
                    if i in self.groups[group_idx]:
                        break
                rand_member = self.groups[group_idx][rand() % len(self.groups[group_idx])]
                new_position = np.zeros(self.dim, dtype=DTYPE)
                for j in range(self.dim):
                    new_position[j] = (self.spider_monkeys[i, j] +
                                      (self.global_leader[j] - self.spider_monkeys[i, j]) * (rand() / RAND_MAX) +
                                      (self.spider_monkeys[rand_member, j] - self.spider_monkeys[i, j]) * (2 * (rand() / RAND_MAX) - 1))
                    if new_position[j] < self.bounds[j, 0]:
                        new_position[j] = self.bounds[j, 0]
                    elif new_position[j] > self.bounds[j, 1]:
                        new_position[j] = self.bounds[j, 1]
                
                new_position, new_fitness = self.beta_hill_climbing(new_position, self.fitness[i])
                if new_fitness < self.fitness[i]:
                    self.spider_monkeys[i] = new_position
                    self.fitness[i] = new_fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void local_leader_decision(self):
        """ Update local leaders and apply BHC if no improvement """
        cdef int group_idx, idx, best_idx
        cdef list group
        cdef double min_fitness
        for group_idx in range(len(self.groups)):
            group = self.groups[group_idx]
            min_fitness = self.fitness[group[0]]
            best_idx = group[0]
            for idx in group:
                if self.fitness[idx] < min_fitness:
                    min_fitness = self.fitness[idx]
                    best_idx = idx
            
            if min_fitness < self.local_leader_fitness[group_idx]:
                self.local_leaders[group_idx] = self.spider_monkeys[best_idx].copy()
                self.local_leader_fitness[group_idx] = min_fitness
                self.local_leader_count[group_idx] = 0
            else:
                self.local_leader_count[group_idx] += 1
                
                if self.local_leader_count[group_idx] > self.local_leader_limit:
                    for idx in group:
                        self.spider_monkeys[idx], self.fitness[idx] = self.beta_hill_climbing(
                            self.spider_monkeys[idx], self.fitness[idx])
                    self.local_leader_count[group_idx] = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void global_leader_decision(self):
        """ Update global leader and split/merge groups if needed """
        cdef int min_idx = np.argmin(self.fitness)
        if self.fitness[min_idx] < self.global_leader_fitness:
            self.global_leader = self.spider_monkeys[min_idx].copy()
            self.global_leader_fitness = self.fitness[min_idx]
            self.global_leader_count = 0
        else:
            self.global_leader_count += 1
            
        if self.global_leader_count > self.global_leader_limit:
            self.global_leader_count = 0
            if len(self.groups) < self.max_groups:
                largest_group_idx = 0
                max_size = len(self.groups[0])
                for i in range(1, len(self.groups)):
                    if len(self.groups[i]) > max_size:
                        max_size = len(self.groups[i])
                        largest_group_idx = i
                largest_group = self.groups[largest_group_idx]
                if len(largest_group) > 1:
                    np.random.shuffle(largest_group)
                    split_point = len(largest_group) // 2
                    new_group1 = largest_group[:split_point]
                    new_group2 = largest_group[split_point:]
                    self.groups[largest_group_idx] = new_group1
                    self.groups.append(new_group2)
                    self.local_leaders.append(self.spider_monkeys[new_group2[0]].copy())
                    self.local_leader_fitness.append(self.fitness[new_group2[0]])
                    self.local_leader_count.append(0)
            else:
                self.groups = [list(range(self.population_size))]
                self.local_leaders = [self.global_leader.copy()]
                self.local_leader_fitness = [self.global_leader_fitness]
                self.local_leader_count = [0]

    def optimize(self):
        """ Run the Spider Monkey Optimization with BHC (SMOBHC) """
        self.initialize_population()
        cdef int iteration
        for iteration in range(self.max_iter):
            self.local_leader_phase()
            self.global_leader_phase()
            self.local_leader_decision()
            self.global_leader_decision()
            
            min_idx = np.argmin(self.fitness)
            self.history.append((iteration, self.global_leader.copy(), self.global_leader_fitness))
            print(f"Iteration {iteration + 1}: Best Value = {self.global_leader_fitness}")
        
        return self.global_leader, self.global_leader_fitness, self.history
