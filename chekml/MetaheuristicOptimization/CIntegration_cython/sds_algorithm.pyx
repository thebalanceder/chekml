# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport sqrt

# Define NumPy array types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ITYPE = np.int32
ctypedef np.int32_t ITYPE_t

cdef class StochasticDiffusionSearch:
    cdef public:
        object objective_function
        int dim
        np.ndarray bounds
        int population_size
        int max_iter
        double mutation_rate
        double mutation_scale
        double cluster_threshold
        bint context_sensitive
        np.ndarray agents
        np.ndarray activities
        np.ndarray best_solution
        double best_value
        list history
        object component_functions
        int num_components

    def __init__(self, object objective_function, int dim, bounds, int population_size=1000, 
                 int max_iter=100, double mutation_rate=0.08, double mutation_scale=4.0, 
                 double cluster_threshold=0.33, bint context_sensitive=False):
        """
        Initialize the Stochastic Diffusion Search (SDS) optimizer.

        Parameters:
        - objective_function: Function to optimize, returns a list of component values.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of agents (hypotheses).
        - max_iter: Maximum number of iterations.
        - mutation_rate: Probability of applying mutation during diffusion.
        - mutation_scale: Controls the standard deviation of mutation offset.
        - cluster_threshold: Fraction of agents required in a cluster to consider convergence.
        - context_sensitive: If True, uses context-sensitive diffusion to balance exploration.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)
        self.population_size = population_size
        self.max_iter = max_iter
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.cluster_threshold = cluster_threshold
        self.context_sensitive = context_sensitive
        self.best_value = float("inf")
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def initialize_agents(self):
        """ Generate initial hypotheses for agents randomly """
        cdef np.ndarray[DTYPE_t, ndim=2] agents
        cdef np.ndarray[ITYPE_t, ndim=1] activities
        cdef np.ndarray[DTYPE_t, ndim=1] test_input
        cdef int i, j

        # Initialize agents
        agents = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                  (self.population_size, self.dim)).astype(DTYPE)
        activities = np.zeros(self.population_size, dtype=ITYPE)

        # Determine number of components
        test_input = agents[0]
        components = self.objective_function(test_input)
        self.num_components = len(components)
        self.component_functions = lambda x, i: self.objective_function(x)[i % self.num_components]

        self.agents = agents
        self.activities = activities

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int evaluate_component(self, np.ndarray[DTYPE_t, ndim=1] hypothesis, int component_idx):
        """ Evaluate a single component function for a hypothesis """
        cdef double value, t, max_value
        value = self.component_functions(hypothesis, component_idx)
        max_value = max(1.0, abs(value))
        t = abs(value) / max_value
        return 0 if (<double>rand() / RAND_MAX) < t else 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def test_phase(self):
        """ Perform the test phase: each agent evaluates a random component function """
        cdef int i, component_idx
        cdef np.ndarray[ITYPE_t, ndim=1] activities = self.activities
        cdef np.ndarray[DTYPE_t, ndim=2] agents = self.agents

        for i in range(self.population_size):
            component_idx = rand() % self.num_components
            activities[i] = self.evaluate_component(agents[i], component_idx)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def diffusion_phase(self):
        """ Perform the diffusion phase: inactive agents recruit or pick random hypotheses """
        cdef int i, j, agent2_idx
        cdef double r
        cdef np.ndarray[DTYPE_t, ndim=2] agents = self.agents
        cdef np.ndarray[ITYPE_t, ndim=1] activities = self.activities
        cdef np.ndarray[DTYPE_t, ndim=1] new_hypothesis
        cdef np.ndarray[DTYPE_t, ndim=1] lower_bounds = self.bounds[:, 0]
        cdef np.ndarray[DTYPE_t, ndim=1] upper_bounds = self.bounds[:, 1]
        cdef double[::1] offset = np.zeros(self.dim, dtype=DTYPE)

        for i in range(self.population_size):
            if not activities[i]:
                agent2_idx = rand() % self.population_size
                if activities[agent2_idx]:
                    new_hypothesis = agents[agent2_idx].copy()
                    r = <double>rand() / RAND_MAX
                    if r < self.mutation_rate:
                        for j in range(self.dim):
                            offset[j] = np.random.normal(0, 1) / self.mutation_scale
                            new_hypothesis[j] += offset[j]
                            if new_hypothesis[j] < lower_bounds[j]:
                                new_hypothesis[j] = lower_bounds[j]
                            elif new_hypothesis[j] > upper_bounds[j]:
                                new_hypothesis[j] = upper_bounds[j]
                    agents[i] = new_hypothesis
                else:
                    for j in range(self.dim):
                        agents[i, j] = lower_bounds[j] + (upper_bounds[j] - lower_bounds[j]) * (<double>rand() / RAND_MAX)
            elif self.context_sensitive and activities[i]:
                agent2_idx = rand() % self.population_size
                if activities[agent2_idx]:
                    # Check if hypotheses are identical
                    for j in range(self.dim):
                        if agents[agent2_idx, j] != agents[i, j]:
                            break
                    else:  # Hypotheses are identical
                        activities[i] = False
                        for j in range(self.dim):
                            agents[i, j] = lower_bounds[j] + (upper_bounds[j] - lower_bounds[j]) * (<double>rand() / RAND_MAX)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def evaluate_full_objective(self, np.ndarray[DTYPE_t, ndim=1] hypothesis):
        """ Compute the full objective function value for a hypothesis """
        return sum(self.objective_function(hypothesis))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def check_convergence(self):
        """ Check if a cluster of sufficient size has formed at the best solution """
        if self.best_solution is None:
            return False
        cdef np.ndarray[DTYPE_t, ndim=2] agents = self.agents
        cdef np.ndarray[DTYPE_t, ndim=1] best_solution = self.best_solution
        cdef double dist
        cdef int i, j, cluster_size = 0
        for i in range(self.population_size):
            dist = 0.0
            for j in range(self.dim):
                dist += (agents[i, j] - best_solution[j]) ** 2
            if sqrt(dist) < 1e-3:
                cluster_size += 1
        return cluster_size / self.population_size >= self.cluster_threshold

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """ Run the Stochastic Diffusion Search optimization """
        self.initialize_agents()
        cdef np.ndarray[DTYPE_t, ndim=2] agents = self.agents
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = np.empty(self.population_size, dtype=DTYPE)
        cdef int i, iteration, min_idx
        cdef double min_fitness

        for iteration in range(self.max_iter):
            # Test phase
            self.test_phase()

            # Diffusion phase
            self.diffusion_phase()

            # Evaluate best solution
            for i in range(self.population_size):
                fitness[i] = self.evaluate_full_objective(agents[i])
            min_idx = np.argmin(fitness)
            min_fitness = fitness[min_idx]
            if min_fitness < self.best_value:
                self.best_solution = agents[min_idx].copy()
                self.best_value = min_fitness

            # Log progress
            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

            # Check halting criterion
            if self.check_convergence():
                print(f"Converged at iteration {iteration + 1}")
                break

        return self.best_solution, self.best_value, self.history
