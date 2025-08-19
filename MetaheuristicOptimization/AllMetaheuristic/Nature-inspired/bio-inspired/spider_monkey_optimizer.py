import numpy as np

class SpiderMonkeyOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, 
                 pr=0.1, local_leader_limit=50, global_leader_limit=1500, max_groups=5):
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
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.pr = pr
        self.local_leader_limit = local_leader_limit
        self.global_leader_limit = global_leader_limit
        self.max_groups = max_groups

        self.spider_monkeys = None  # Population of spider monkey positions
        self.fitness = None  # Fitness values
        self.local_leaders = None  # Local leader positions for each group
        self.local_leader_fitness = None  # Fitness of local leaders
        self.global_leader = None  # Global leader position
        self.global_leader_fitness = float("inf")  # Fitness of global leader
        self.groups = []  # List of groups (indices of spider monkeys)
        self.local_leader_count = None  # Counter for local leader stagnation
        self.global_leader_count = 0  # Counter for global leader stagnation
        self.history = []

    def initialize_population(self):
        """ Initialize spider monkey positions and groups """
        self.spider_monkeys = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                                (self.population_size, self.dim))
        self.fitness = np.array([self.objective_function(sm) for sm in self.spider_monkeys])
        self.groups = [list(range(self.population_size))]  # Start with one group
        self.local_leaders = [self.spider_monkeys[np.argmin(self.fitness)]]
        self.local_leader_fitness = [np.min(self.fitness)]
        self.local_leader_count = [0]
        self.global_leader = self.spider_monkeys[np.argmin(self.fitness)].copy()
        self.global_leader_fitness = np.min(self.fitness)

    def beta_hill_climbing(self, solution, fitness, delta=0.1):
        """
        Apply Beta-Hill Climbing Optimizer (BHC) to refine a solution.
        
        Parameters:
        - solution: Current solution (spider monkey position).
        - fitness: Fitness of the current solution.
        - delta: Step size for BHC.
        """
        new_solution = solution.copy()
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                # Apply beta distribution for exploration
                beta = np.random.beta(2, 5)
                new_solution[i] += delta * beta * (self.bounds[i, 1] - self.bounds[i, 0])
            else:
                # Apply hill climbing for exploitation
                new_solution[i] += delta * np.random.uniform(-1, 1)
        
        new_solution = np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])
        new_fitness = self.objective_function(new_solution)
        
        # Accept new solution if it improves fitness
        if new_fitness < fitness:
            return new_solution, new_fitness
        return solution, fitness

    def local_leader_phase(self):
        """ Update spider monkey positions in Local Leader Phase (LLP) with BHC """
        for group_idx, group in enumerate(self.groups):
            for idx in group:
                if np.random.rand() > self.pr:
                    # Update position based on local leader and another group member
                    rand_member = np.random.choice(group)
                    new_position = (self.spider_monkeys[idx] + 
                                   (self.local_leaders[group_idx] - self.spider_monkeys[idx]) * np.random.rand() +
                                   (self.spider_monkeys[idx] - self.spider_monkeys[rand_member]) * np.random.uniform(-1, 1))
                    new_position = np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])
                    
                    # Apply BHC to refine the new position
                    new_position, new_fitness = self.beta_hill_climbing(new_position, self.fitness[idx])
                    
                    # Update if better
                    if new_fitness < self.fitness[idx]:
                        self.spider_monkeys[idx] = new_position
                        self.fitness[idx] = new_fitness

    def global_leader_phase(self):
        """ Update spider monkey positions in Global Leader Phase (GLP) with BHC """
        prob = 0.9 * (1 - np.array(self.fitness) / np.max(self.fitness)) + 0.1  # Selection probability
        for i in range(self.population_size):
            if np.random.rand() < prob[i]:
                group_idx = next(idx for idx, group in enumerate(self.groups) if i in group)
                rand_member = np.random.choice(self.groups[group_idx])
                new_position = (self.spider_monkeys[i] + 
                               (self.global_leader - self.spider_monkeys[i]) * np.random.rand() +
                               (self.spider_monkeys[rand_member] - self.spider_monkeys[i]) * np.random.uniform(-1, 1))
                new_position = np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])
                
                # Apply BHC to refine the new position
                new_position, new_fitness = self.beta_hill_climbing(new_position, self.fitness[i])
                
                # Update if better
                if new_fitness < self.fitness[i]:
                    self.spider_monkeys[i] = new_position
                    self.fitness[i] = new_fitness

    def local_leader_decision(self):
        """ Update local leaders and apply BHC if no improvement """
        for group_idx, group in enumerate(self.groups):
            group_fitness = [self.fitness[i] for i in group]
            best_idx = group[np.argmin(group_fitness)]
            
            if self.fitness[best_idx] < self.local_leader_fitness[group_idx]:
                self.local_leaders[group_idx] = self.spider_monkeys[best_idx].copy()
                self.local_leader_fitness[group_idx] = self.fitness[best_idx]
                self.local_leader_count[group_idx] = 0
            else:
                self.local_leader_count[group_idx] += 1
                
                if self.local_leader_count[group_idx] > self.local_leader_limit:
                    # Apply BHC to all members in the group
                    for idx in group:
                        self.spider_monkeys[idx], self.fitness[idx] = self.beta_hill_climbing(
                            self.spider_monkeys[idx], self.fitness[idx])
                    self.local_leader_count[group_idx] = 0

    def global_leader_decision(self):
        """ Update global leader and split/merge groups if needed """
        min_idx = np.argmin(self.fitness)
        if self.fitness[min_idx] < self.global_leader_fitness:
            self.global_leader = self.spider_monkeys[min_idx].copy()
            self.global_leader_fitness = self.fitness[min_idx]
            self.global_leader_count = 0
        else:
            self.global_leader_count += 1
            
        if self.global_leader_count > self.global_leader_limit:
            self.global_leader_count = 0
            if len(self.groups) < self.max_groups:
                # Split largest group
                largest_group_idx = np.argmax([len(g) for g in self.groups])
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
                # Merge all groups into one
                self.groups = [list(range(self.population_size))]
                self.local_leaders = [self.global_leader.copy()]
                self.local_leader_fitness = [self.global_leader_fitness]
                self.local_leader_count = [0]

    def optimize(self):
        """ Run the Spider Monkey Optimization with BHC (SMOBHC) """
        self.initialize_population()
        for iteration in range(self.max_iter):
            # Local Leader Phase
            self.local_leader_phase()
            
            # Global Leader Phase
            self.global_leader_phase()
            
            # Local Leader Decision
            self.local_leader_decision()
            
            # Global Leader Decision
            self.global_leader_decision()
            
            # Update history
            min_idx = np.argmin(self.fitness)
            self.history.append((iteration, self.global_leader.copy(), self.global_leader_fitness))
            print(f"Iteration {iteration + 1}: Best Value = {self.global_leader_fitness}")
        
        return self.global_leader, self.global_leader_fitness, self.history
