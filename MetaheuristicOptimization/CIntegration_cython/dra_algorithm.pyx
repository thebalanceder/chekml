# cython: language_level=3
import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport cos, sin, pi, isnan

# Enable bounds checking to ensure stability
@cython.boundscheck(True)
@cython.wraparound(False)
cdef class DivineReligionsAlgorithm:
    cdef object objective_function
    cdef int dim
    cdef cnp.ndarray bounds
    cdef int population_size
    cdef int max_iter
    cdef double belief_profile_rate
    cdef double miracle_rate
    cdef double proselytism_rate
    cdef double reward_penalty_rate
    cdef int num_groups
    cdef int num_followers
    cdef cnp.ndarray belief_profiles
    cdef cnp.ndarray costs
    cdef cnp.ndarray best_solution
    cdef double best_cost
    cdef list cg_curve
    cdef public list history  # Make history accessible to Python
    cdef cnp.ndarray missionaries
    cdef cnp.ndarray followers
    cdef list groups

    def __init__(self, objective_function, int dim, bounds, int population_size=50, int max_iter=1000,
                 double belief_profile_rate=0.5, double miracle_rate=0.5, double proselytism_rate=0.9,
                 double reward_penalty_rate=0.2, int num_groups=5):
        """
        Initialize the Divine Religions Algorithm (DRA) optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of belief profiles (solutions).
        - max_iter: Maximum number of iterations.
        - belief_profile_rate: Belief Profile Consideration Rate (BPSP).
        - miracle_rate: Miracle Rate (MP).
        - proselytism_rate: Proselytism Consideration Rate (PP).
        - reward_penalty_rate: Reward or Penalty Consideration Rate (RP).
        - num_groups: Number of groups.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.double)
        self.population_size = population_size
        self.max_iter = max_iter
        self.belief_profile_rate = belief_profile_rate
        self.miracle_rate = miracle_rate
        self.proselytism_rate = proselytism_rate
        self.reward_penalty_rate = reward_penalty_rate
        self.num_groups = min(num_groups, population_size)  # Ensure num_groups <= population_size
        self.num_followers = population_size - self.num_groups
        self.best_cost = float("inf")
        self.cg_curve = []
        self.history = []  # Initialize history list
        print("Initialized history:", self.history)

    @cython.boundscheck(True)
    @cython.wraparound(False)
    def initialize_belief_profiles(self):
        """ Generate initial belief profiles randomly """
        cdef cnp.ndarray[cnp.double_t, ndim=2] belief_profiles
        cdef cnp.ndarray[cnp.double_t, ndim=1] costs
        cdef int i
        belief_profiles = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                           (self.population_size, self.dim))
        costs = np.zeros(self.population_size, dtype=np.double)
        for i in range(self.population_size):
            costs[i] = self.objective_function(belief_profiles[i])
            if isnan(costs[i]):
                costs[i] = float("inf")  # Handle NaN from objective function

        # Sort belief profiles by cost
        sort_order = np.argsort(costs)
        self.belief_profiles = belief_profiles[sort_order].copy()  # Ensure copy to avoid memory issues
        self.costs = costs[sort_order].copy()

    @cython.boundscheck(True)
    @cython.wraparound(False)
    def initialize_groups(self):
        """ Initialize groups and assign missionaries and followers """
        # Use NumPy arrays for missionaries and followers
        self.missionaries = self.belief_profiles[:self.num_groups].copy()
        self.followers = self.belief_profiles[self.num_groups:].copy()
        self.groups = [[] for _ in range(self.num_groups)]
        
        # Assign followers to groups
        cdef cnp.ndarray[cnp.int64_t, ndim=1] follower_indices
        follower_indices = np.random.choice(self.num_followers, self.num_followers, replace=False).astype(np.int64)
        cdef int idx, group_idx
        for idx in range(follower_indices.shape[0]):
            group_idx = np.random.randint(0, self.num_groups)
            self.groups[group_idx].append(self.followers[follower_indices[idx]].copy())

    @cython.boundscheck(True)
    @cython.wraparound(False)
    def miracle_operator(self):
        """ Apply Miracle Operator for exploration """
        cdef int i, j
        cdef double rand_val
        cdef cnp.ndarray[cnp.double_t, ndim=1] bp
        for i in range(self.population_size):
            rand_val = np.random.rand()
            bp = self.belief_profiles[i]
            if rand_val <= 0.5:
                for j in range(self.dim):
                    bp[j] *= cos(pi / 2) * (np.random.rand() - cos(np.random.rand()))
            else:
                for j in range(self.dim):
                    bp[j] += np.random.rand() * (bp[j] - np.round(1 ** np.random.rand()) * bp[j])
            
            # Ensure bounds
            for j in range(self.dim):
                bp[j] = min(max(bp[j], self.bounds[j, 0]), self.bounds[j, 1])
            
            new_cost = self.objective_function(bp)
            if isnan(new_cost):
                new_cost = float("inf")
            if new_cost < self.costs[i]:
                self.costs[i] = new_cost

    @cython.boundscheck(True)
    @cython.wraparound(False)
    def proselytism_operator(self, cnp.ndarray[cnp.double_t, ndim=1] leader):
        """ Apply Proselytism Operator for exploitation """
        cdef int i, j
        cdef double mean_bp, rand_val
        cdef cnp.ndarray[cnp.double_t, ndim=1] bp
        for i in range(self.population_size):
            rand_val = np.random.rand()
            bp = self.belief_profiles[i]
            if rand_val > (1 - self.miracle_rate):
                mean_bp = np.mean(bp)
                for j in range(self.dim):
                    bp[j] = (bp[j] * 0.01 + mean_bp * (1 - self.miracle_rate) +
                             (1 - mean_bp) - (np.random.rand() - 4 * sin(sin(pi * np.random.rand()))))
            else:
                for j in range(self.dim):
                    bp[j] = leader[j] * (np.random.rand() - cos(np.random.rand()))
            
            # Ensure bounds
            for j in range(self.dim):
                bp[j] = min(max(bp[j], self.bounds[j, 0]), self.bounds[j, 1])
            
            new_cost = self.objective_function(bp)
            if isnan(new_cost):
                new_cost = float("inf")
            if new_cost < self.costs[i]:
                self.costs[i] = new_cost

    @cython.boundscheck(True)
    @cython.wraparound(False)
    def reward_penalty_operator(self):
        """ Apply Reward or Penalty Operator """
        cdef int index = np.random.randint(0, self.population_size)
        cdef int j
        if np.random.rand() >= self.reward_penalty_rate:
            # Reward
            for j in range(self.dim):
                self.belief_profiles[index, j] *= (1 - np.random.randn())
        else:
            # Penalty
            for j in range(self.dim):
                self.belief_profiles[index, j] *= (1 + np.random.randn())
        
        # Ensure bounds
        for j in range(self.dim):
            self.belief_profiles[index, j] = min(max(self.belief_profiles[index, j], self.bounds[j, 0]), self.bounds[j, 1])
        
        new_cost = self.objective_function(self.belief_profiles[index])
        if isnan(new_cost):
            new_cost = float("inf")
        if new_cost < self.costs[index]:
            self.costs[index] = new_cost

    @cython.boundscheck(True)
    @cython.wraparound(False)
    def replacement_operator(self):
        """ Perform replacement between missionaries and followers in groups """
        cdef int k
        cdef cnp.ndarray[cnp.double_t, ndim=1] temp
        for k in range(self.num_groups):
            if len(self.groups[k]) > 0:
                # Swap missionary and last follower
                temp = self.missionaries[k].copy()
                self.missionaries[k] = self.groups[k][-1].copy()
                self.groups[k][-1] = temp

    @cython.boundscheck(True)
    @cython.wraparound(False)
    def optimize(self):
        """ Run the Divine Religions Algorithm """
        self.initialize_belief_profiles()
        self.initialize_groups()
        cdef int iteration, min_idx, rand_idx, rand_dim, j
        cdef double miracle_rate, new_cost
        cdef cnp.ndarray[cnp.double_t, ndim=1] leader, new_follower

        for iteration in range(self.max_iter):
            # Update miracle rate
            miracle_rate = (1 * np.random.rand()) * (1 - (<double>iteration / self.max_iter * 2)) * (1 * np.random.rand())

            # Select leader (best belief profile)
            min_idx = np.argmin(self.costs)
            leader = self.belief_profiles[min_idx].copy()

            # Create new follower
            new_follower = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)
            new_cost = self.objective_function(new_follower)
            if isnan(new_cost):
                new_cost = float("inf")

            # Belief Profile Consideration
            if np.random.rand() <= self.belief_profile_rate:
                rand_idx = np.random.randint(0, self.population_size)
                rand_dim = np.random.randint(0, self.dim)
                new_follower[rand_dim] = self.belief_profiles[rand_idx, rand_dim]

            # Exploration or Exploitation
            if np.random.rand() <= miracle_rate:
                self.miracle_operator()
            else:
                for j in range(self.dim):
                    new_follower[j] = leader[j] * (np.random.rand() - sin(np.random.rand()))
                for j in range(self.dim):
                    new_follower[j] = min(max(new_follower[j], self.bounds[j, 0]), self.bounds[j, 1])
                new_cost = self.objective_function(new_follower)
                if isnan(new_cost):
                    new_cost = float("inf")
                self.proselytism_operator(leader)

            # Update new follower cost
            if new_cost < self.costs[-1]:
                self.belief_profiles[-1] = new_follower.copy()
                self.costs[-1] = new_cost

            # Reward or Penalty
            self.reward_penalty_operator()

            # Replacement
            self.replacement_operator()

            # Update best solution
            min_idx = np.argmin(self.costs)
            if self.costs[min_idx] < self.best_cost:
                self.best_solution = self.belief_profiles[min_idx].copy()
                self.best_cost = self.costs[min_idx]

            # Store convergence curve and history
            self.cg_curve.append(self.best_cost)
            self.history.append((iteration, self.best_solution.copy(), self.best_cost))
            print(f"Iteration {iteration + 1}: Best Cost = {self.best_cost}, History length = {len(self.history)}")
        print("Final history:", self.history[:5])  # Print first 5 history entries
        return self.best_solution, self.best_cost, self.cg_curve
