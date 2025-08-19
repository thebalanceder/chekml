import numpy as np

class DivineReligionsAlgorithm:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100,
                 belief_profile_rate=0.5, miracle_rate=0.5, proselytism_rate=0.9, reward_penalty_rate=0.2,
                 num_groups=5):
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
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.belief_profile_rate = belief_profile_rate
        self.miracle_rate = miracle_rate
        self.proselytism_rate = proselytism_rate
        self.reward_penalty_rate = reward_penalty_rate
        self.num_groups = num_groups
        self.num_followers = population_size - num_groups

        self.belief_profiles = None  # Population of belief profiles
        self.best_solution = None
        self.best_cost = float("inf")
        self.cg_curve = []
        self.history = []  # To store iteration history

    def initialize_belief_profiles(self):
        """ Generate initial belief profiles randomly """
        self.belief_profiles = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                                (self.population_size, self.dim))
        self.costs = np.array([self.objective_function(bp) for bp in self.belief_profiles])

        # Sort belief profiles by cost
        sort_order = np.argsort(self.costs)
        self.belief_profiles = self.belief_profiles[sort_order]
        self.costs = self.costs[sort_order]

    def initialize_groups(self):
        """ Initialize groups and assign missionaries and followers """
        self.missionaries = self.belief_profiles[:self.num_groups]
        self.followers = self.belief_profiles[self.num_groups:]

        self.groups = [[] for _ in range(self.num_groups)]
        follower_indices = np.random.choice(range(len(self.followers)), len(self.followers), replace=False)
        
        # Assign followers to groups randomly
        for idx in follower_indices:
            group_idx = np.random.randint(0, self.num_groups)
            self.groups[group_idx].append(self.followers[idx])

    def miracle_operator(self):
        """ Apply Miracle Operator for exploration """
        for i in range(self.population_size):
            if np.random.rand() <= 0.5:
                self.belief_profiles[i] *= np.cos(np.pi / 2) * (np.random.rand() - np.cos(np.random.rand()))
            else:
                self.belief_profiles[i] += np.random.rand() * (self.belief_profiles[i] - 
                                                              np.round(1 ** np.random.rand()) * self.belief_profiles[i])
            
            # Ensure bounds
            self.belief_profiles[i] = np.clip(self.belief_profiles[i], self.bounds[:, 0], self.bounds[:, 1])
            
            new_cost = self.objective_function(self.belief_profiles[i])
            if new_cost < self.costs[i]:
                self.costs[i] = new_cost

    def proselytism_operator(self, leader):
        """ Apply Proselytism Operator for exploitation """
        for i in range(self.population_size):
            if np.random.rand() > (1 - self.miracle_rate):
                mean_bp = np.mean(self.belief_profiles[i])
                self.belief_profiles[i] = (self.belief_profiles[i] * 0.01 + 
                                          mean_bp * (1 - self.miracle_rate) + 
                                          (1 - mean_bp) - (np.random.rand() - 4 * np.sin(np.sin(np.pi * np.random.rand()))))
            else:
                self.belief_profiles[i] = leader * (np.random.rand() - np.cos(np.random.rand()))
            
            # Ensure bounds
            self.belief_profiles[i] = np.clip(self.belief_profiles[i], self.bounds[:, 0], self.bounds[:, 1])
            
            new_cost = self.objective_function(self.belief_profiles[i])
            if new_cost < self.costs[i]:
                self.costs[i] = new_cost

    def reward_penalty_operator(self):
        """ Apply Reward or Penalty Operator """
        index = np.random.randint(0, self.population_size)
        if np.random.rand() >= self.reward_penalty_rate:
            # Reward
            self.belief_profiles[index] *= (1 - np.random.randn())
        else:
            # Penalty
            self.belief_profiles[index] *= (1 + np.random.randn())
        
        # Ensure bounds
        self.belief_profiles[index] = np.clip(self.belief_profiles[index], self.bounds[:, 0], self.bounds[:, 1])
        
        new_cost = self.objective_function(self.belief_profiles[index])
        if new_cost < self.costs[index]:
            self.costs[index] = new_cost

    def replacement_operator(self):
        """ Perform replacement between missionaries and followers in groups """
        for k in range(self.num_groups):
            if len(self.groups[k]) > 0:
                # Swap missionary and last follower
                missionary = self.missionaries[k]
                follower = self.groups[k][-1]
                self.missionaries[k] = follower
                self.groups[k][-1] = missionary

    def optimize(self):
        """ Run the Divine Religions Algorithm """
        self.initialize_belief_profiles()
        self.initialize_groups()

        for iteration in range(self.max_iter):
            # Update miracle rate
            miracle_rate = (1 * np.random.rand()) * (1 - (iteration / self.max_iter * 2)) * (1 * np.random.rand())

            # Select leader (best belief profile)
            min_idx = np.argmin(self.costs)
            leader = self.belief_profiles[min_idx]

            # Create new follower
            new_follower = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)
            new_cost = self.objective_function(new_follower)

            # Belief Profile Consideration
            if np.random.rand() <= self.belief_profile_rate:
                rand_idx = np.random.randint(0, self.population_size)
                rand_dim = np.random.randint(0, self.dim)
                new_follower[rand_dim] = self.belief_profiles[rand_idx][rand_dim]

            # Exploration or Exploitation
            if np.random.rand() <= miracle_rate:
                self.miracle_operator()
            else:
                new_follower = leader * (np.random.rand() - np.sin(np.random.rand()))
                new_follower = np.clip(new_follower, self.bounds[:, 0], self.bounds[:, 1])
                new_cost = self.objective_function(new_follower)
                self.proselytism_operator(leader)

            # Update new follower cost
            if new_cost < self.costs[-1]:
                self.belief_profiles[-1] = new_follower
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
            print(f"Iteration {iteration + 1}: Best Cost = {self.best_cost}")

        return self.best_solution, self.best_cost, self.cg_curve

# Example usage:
if __name__ == "__main__":
    # Define a simple objective function (e.g., Sphere function)
    def sphere_function(x):
        return np.sum(x ** 2)

    # Define problem parameters
    dim = 30
    bounds = [(-100, 100)] * dim  # Bounds for each dimension
    population_size = 50
    max_iter = 1000

    # Initialize and run DRA
    dra = DivineReligionsAlgorithm(
        objective_function=sphere_function,
        dim=dim,
        bounds=bounds,
        population_size=population_size,
        max_iter=max_iter
    )
    best_solution, best_cost, cg_curve = dra.optimize()
    print(f"Best Solution: {best_solution}")
    print(f"Best Cost: {best_cost}")
